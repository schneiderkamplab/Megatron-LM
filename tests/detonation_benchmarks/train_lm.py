"""Train a causal LM with Megatron-FSDP on a hybrid-sharded mesh, logging to wandb.

Parameters are sharded within a node (dp_shard = GPUs per node) and replicated
across nodes (dp_outer = one rank per node); the DeToNATION replicator runs
across dp_outer. ``--strategy none`` is the plain hybrid-sharded baseline,
where Megatron-FSDP all-reduces gradients across dp_outer natively.

Data is a flat uint32 token file (see tokenize_corpus.py). The last
``--val-tokens`` tokens are held out for validation; training windows are
sampled from the rest with a fixed seed, so every strategy sees the same
batches.

Usage (one node per 8 GPUs, launched by slurm_lumi_train_lm.sh):
    python train_lm.py --model allenai/OLMo-2-1124-7B --data tokens.bin \\
        --strategy random --rate 0.0625 --wandb-project detonation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

STRATEGIES = ("none", "full", "full_sync", "demo", "slicing", "striding", "random")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="HF model id; its config is trained from scratch")
    p.add_argument("--init-pretrained", action="store_true",
                   help="start from the model's pretrained weights instead of a fresh init")
    p.add_argument("--data", required=True, help="Flat uint32 token file")
    p.add_argument("--loss-mask", default=None,
                   help="uint8 file parallel to --data; only tokens marked 1 are loss targets "
                        "(see prepare_sft.py)")
    p.add_argument("--val-tokens", type=int, default=20_000_000)
    p.add_argument("--seq-length", type=int, default=2048)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--train-iters", type=int, default=5000)
    p.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.04)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--outer-sharding", choices=("no_shard", "optim"), default="no_shard")
    p.add_argument("--strategy", choices=STRATEGIES, default="none")
    p.add_argument("--rate", type=float, default=0.0625, help="random/slicing/striding")
    p.add_argument("--topk", type=int, default=4, help="demo")
    p.add_argument("--chunk", type=int, default=64, help="demo/slicing/striding")
    p.add_argument("--decay", type=float, default=0.999)
    p.add_argument("--max-delay", type=int, default=4,
                   help="random: steps a result may arrive late; 0 = synchronous random/demo")
    p.add_argument("--sign", action="store_true", help="optimizer sees sign(replicated grad)")
    p.add_argument("--wait-ms", type=float, default=0.0,
                   help="async replicators: results that have arrived when the step's "
                        "gradients are ready apply in that step, waiting up to this many "
                        "ms for the rest (0, the default: no waiting); negative gives the "
                        "delayed mode (every result one step late)")
    p.add_argument("--profile-steps", type=int, nargs=2, default=None, metavar=("START", "STOP"),
                   help="on rank 0, record communication bytes and a profiler trace for "
                        "steps [START, STOP) and write a summary to --profile-dir")
    p.add_argument("--profile-dir", default="comm_profile")
    p.add_argument("--local-metrics", action="store_true",
                   help="reduce train/val loss within each node only and log node 0's, "
                        "so logging never synchronizes nodes")
    p.add_argument("--save-dir", default=None,
                   help="save the final model here as a Hugging Face checkpoint (bf16)")
    p.add_argument("--ckpt-dir", default=None,
                   help="training checkpoints (model, optimizer, replicator, wandb run); "
                        "training resumes from the latest one found here")
    p.add_argument("--ckpt-interval", type=int, default=1000,
                   help="steps between training checkpoints (with --ckpt-dir)")
    p.add_argument("--seed", type=int, default=0, help="model init and data order")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-seqs", type=int, default=512, help="validation sequences, split over ranks")
    p.add_argument("--wandb-entity", default=None, help="wandb team or user; default: the key's")
    p.add_argument("--wandb-project", default="detonation-megatron")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    return p.parse_args()


def build_hybrid_mesh(world_size: int, gpus_per_node: int):
    """2D mesh (dp_outer = nodes, dp_shard = GPUs per node), in Slurm block rank order."""
    if world_size % gpus_per_node:
        raise ValueError(f"world size {world_size} not divisible by {gpus_per_node} GPUs/node")
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(world_size // gpus_per_node, gpus_per_node),
        mesh_dim_names=("dp_outer", "dp_shard"),
    )
    mesh[("dp_outer", "dp_shard")]._flatten("hsdp")
    return mesh, mesh["hsdp"].get_group()


def build_model(model_id: str, seed: int, grad_checkpointing: bool, pretrained: bool):
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)  # Identical init on every rank before sharding.
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    else:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    if grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    unit_types = []
    for module in model.modules():
        name = type(module).__name__
        if ("DecoderLayer" in name or name.endswith("Block")) and type(module) not in unit_types:
            unit_types.append(type(module))
    return model, unit_types


class TokenData:
    """Train/validation windows from a flat token file; the tail is held out for validation."""

    def __init__(self, path: str, seq_length: int, val_tokens: int, mask_path: str | None = None):
        self.tokens = np.memmap(path, dtype=np.uint32, mode="r")
        self.mask = None
        if mask_path is not None:
            self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
            if len(self.mask) != len(self.tokens):
                raise ValueError(f"{mask_path}: length differs from {path}")
        self.seq_length = seq_length
        self.train_end = len(self.tokens) - val_tokens
        if self.train_end < 100 * (seq_length + 1):
            raise ValueError(f"{path}: {len(self.tokens)} tokens is too few")

    def _window(self, start: int):
        """(input_ids, labels) of one window; masked-out targets are -100."""
        end = start + self.seq_length + 1
        window = torch.from_numpy(np.asarray(self.tokens[start:end], dtype=np.int64))
        labels = window[1:].clone()
        if self.mask is not None:
            keep = torch.from_numpy(np.asarray(self.mask[start + 1 : end], dtype=np.bool_))
            labels[~keep] = -100
        return window[:-1], labels

    def _stack(self, starts):
        inputs, labels = zip(*(self._window(int(s)) for s in starts))
        return torch.stack(inputs), torch.stack(labels)

    def train_batch(self, step: int, rank: int, batch_size: int, seed: int):
        rng = np.random.default_rng([seed, rank, step])
        starts = rng.integers(0, self.train_end - self.seq_length - 1, size=batch_size)
        return self._stack(starts)

    def val_batches(self, num_seqs: int, rank: int, world_size: int, batch_size: int):
        """Fixed, non-overlapping validation windows; each rank takes every world_size-th."""
        stride = self.seq_length + 1
        num_seqs = min(num_seqs, (len(self.tokens) - self.train_end) // stride)
        mine = list(range(rank, num_seqs, world_size))
        for i in range(0, len(mine), batch_size):
            yield self._stack(self.train_end + j * stride for j in mine[i : i + batch_size])


def lm_loss_sum(model, input_ids, labels):
    """(summed loss, number of target tokens); labels of -100 are ignored."""
    logits = model(input_ids).logits
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    return loss, (labels != -100).sum()


def lm_loss(model, input_ids, labels):
    # A window may contain no target tokens under a loss mask; it then adds 0.
    loss, count = lm_loss_sum(model, input_ids, labels)
    return loss / count.clamp(min=1)


@torch.no_grad()
def evaluate(model, data, args, rank, world_size, device, group=None):
    """Mean per-token validation loss over the ranks of `group` (default: all ranks)."""
    model.eval()
    total = torch.zeros(2, device=device, dtype=torch.float64)  # (loss sum, num tokens)
    for input_ids, labels in data.val_batches(args.eval_seqs, rank, world_size,
                                              args.micro_batch_size):
        loss, count = lm_loss_sum(model, input_ids.to(device), labels.to(device))
        total[0] += loss.double()
        total[1] += count
    model.train()
    dist.all_reduce(total, group=group)
    return (total[0] / total[1]).item()


def save_hf_checkpoint(model, model_id: str, save_dir: str, rank: int):
    """Gather the sharded parameters and write a bf16 Hugging Face checkpoint
    (config, weights, tokenizer) from rank 0. Collective: call on every rank."""
    from torch.distributed.tensor import DTensor
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        gather_uneven_dtensor_to_full_tensor,
    )

    full_state = {}
    for name, value in model.state_dict().items():
        if isinstance(value, DTensor):
            value = gather_uneven_dtensor_to_full_tensor(value, torch.device("cpu"))
            value = value.to_local() if isinstance(value, DTensor) else value
        if rank == 0 and isinstance(value, torch.Tensor):
            full_state[name.removeprefix("module.")] = value.detach().to("cpu", torch.bfloat16)
    if rank == 0:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(model_id)
        with torch.device("meta"):
            hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        missing, unexpected = hf_model.load_state_dict(full_state, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith("rotary_emb.inv_freq")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]}, "
                               f"unexpected {unexpected[:5]}")
        hf_model.save_pretrained(save_dir, safe_serialization=True)
        AutoTokenizer.from_pretrained(model_id).save_pretrained(save_dir)
        print(f"saved {len(full_state)} tensors to {save_dir}", flush=True)
    dist.barrier()


def save_training_checkpoint(ckpt_dir: str, step: int, model, optimizer, meta: dict, rank: int):
    """Write a resumable checkpoint for `step` completed steps, then point
    ckpt_dir/latest at it and delete the previous one. Collective."""
    import torch.distributed.checkpoint as dcp

    path = os.path.join(ckpt_dir, f"step_{step:07d}")
    dcp.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()},
             checkpoint_id=path)
    # Replicator state differs between ranks (each node has its own residual).
    if model.replicator is not None:
        torch.save(model.replicator.state_dict(), os.path.join(path, f"replicator_{rank}.pt"))
    dist.barrier()
    if rank == 0:
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({**meta, "step": step}, f)
        latest = os.path.join(ckpt_dir, "latest")
        previous = open(latest).read().strip() if os.path.exists(latest) else None
        with open(latest + ".tmp", "w") as f:
            f.write(os.path.basename(path))
        os.replace(latest + ".tmp", latest)
        if previous and previous != os.path.basename(path):
            shutil.rmtree(os.path.join(ckpt_dir, previous), ignore_errors=True)
        print(f"checkpoint: step {step} -> {path}", flush=True)
    dist.barrier()


def load_training_checkpoint(ckpt_dir: str, model, optimizer, rank: int, device):
    """Load ckpt_dir/latest into model, optimizer and replicator; return its
    metadata, or None if there is no checkpoint. Collective."""
    import torch.distributed.checkpoint as dcp

    latest = os.path.join(ckpt_dir, "latest")
    if not os.path.exists(latest):
        return None
    path = os.path.join(ckpt_dir, open(latest).read().strip())
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.load(state, checkpoint_id=path)
    model.load_state_dict(state["model"], strict=False)  # Also installs the weights.
    optimizer.load_state_dict(state["optimizer"])
    if model.replicator is not None:
        model.replicator.load_state_dict(
            torch.load(os.path.join(path, f"replicator_{rank}.pt"), map_location=device))
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    if rank == 0:
        print(f"resumed from {path} (step {meta['step']}); replication results in flight "
              f"at that point are lost", flush=True)
    return meta


def lr_at(step: int, args) -> float:
    warmup = max(1, int(args.warmup_frac * args.train_iters))
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, args.train_iters - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    from megatron.core.distributed.fsdp.src.megatron_fsdp import (
        MixedPrecisionPolicy,
        fully_shard_model,
        fully_shard_optimizer,
    )

    mesh, hybrid_group = build_hybrid_mesh(world_size, args.gpus_per_node)
    num_nodes = world_size // args.gpus_per_node
    model, unit_types = build_model(args.model, args.seed, args.grad_checkpointing,
                                    args.init_pretrained)
    num_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    model = fully_shard_model(
        model,
        device_mesh=mesh,
        dp_shard_dim="dp_shard",
        dp_outer_dim="dp_outer",
        hybrid_fsdp_group=hybrid_group,
        outer_dp_sharding_strategy=args.outer_sharding,
        zero_dp_strategy="optim_grads_params",
        fsdp_unit_modules=unit_types,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        sync_model_each_microbatch=True,
        # Reduce gradients in fp32; bf16 lacks the precision for cross-rank averaging.
        mixed_precision_policy=MixedPrecisionPolicy(main_grads_dtype=torch.float32),
        device=device,
        replication_strategy=args.strategy,
        replication_decay=args.decay,
        replication_topk=args.topk,
        replication_chunk=args.chunk,
        replication_rate=args.rate,
        replication_seed=42,
        replication_max_delay=args.max_delay,
        replication_wait_ms=args.wait_ms if args.wait_ms >= 0 else None,
        replication_sign=args.sign,
    )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    optimizer = fully_shard_optimizer(optimizer)

    # Node-local metrics: reduce over this node's GPUs only; node 0 (rank 0) logs.
    if args.local_metrics:
        metric_group = mesh["dp_shard"].get_group()
        metric_rank, metric_size = local_rank, args.gpus_per_node
    else:
        metric_group, metric_rank, metric_size = None, rank, world_size

    data = TokenData(args.data, args.seq_length, args.val_tokens, args.loss_mask)
    tokens_per_step = world_size * args.micro_batch_size * args.seq_length

    resumed = None
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        resumed = load_training_checkpoint(args.ckpt_dir, model, optimizer, rank, device)
    start_step = resumed["step"] if resumed else 0

    wandb_run = None
    if rank == 0:
        import wandb

        wandb_run = wandb.init(
            # A resumed run continues the same wandb run.
            id=resumed["wandb_id"] if resumed else None,
            resume="allow" if resumed else None,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or args.strategy,
            tags=args.wandb_tags,
            config={
                **vars(args),
                "num_nodes": num_nodes,
                "world_size": world_size,
                "num_params": num_params,
                "tokens_per_step": tokens_per_step,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
        print(f"{args.model}: {num_params / 1e9:.2f}B params, {num_nodes} nodes x "
              f"{args.gpus_per_node} GPUs, strategy={args.strategy}, "
              f"{tokens_per_step} tokens/step", flush=True)

    def log(metrics, step):
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    profiler = None
    if args.profile_steps is not None and rank == 0:
        from comm_profile import StepProfiler

        profiler = StepProfiler(
            *args.profile_steps, args.profile_dir,
            inter_node_groups=[mesh["dp_outer"].get_group(), hybrid_group], rank=rank,
        )

    loss_sum = torch.zeros((), device=device)
    window_start = time.time()
    step_times = resumed["step_times"] if resumed else []
    train_losses = resumed["train_losses"] if resumed else []
    for step in range(start_step, args.train_iters):
        if step % args.eval_interval == 0:
            eval_start = time.time()
            val_loss = evaluate(model, data, args, metric_rank, metric_size, device, metric_group)
            log({"val/loss": val_loss}, step)
            if rank == 0:
                print(f"step {step:6d}  val_loss {val_loss:.4f}", flush=True)
            window_start += time.time() - eval_start  # Keep eval out of step times.

        if profiler is not None:
            profiler.step_begin(step)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        input_ids, labels = data.train_batch(step, rank, args.micro_batch_size, args.seed)
        loss = lm_loss(model, input_ids.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_sum += loss.detach()
        if profiler is not None:
            profiler.step_end(step, {"strategy": args.strategy, "max_delay": args.max_delay,
                                     "num_nodes": num_nodes, "num_params": num_params})

        if (step + 1) % args.log_interval == 0:
            # Also synchronizes (all ranks, or this node), so the timing below is real.
            dist.all_reduce(loss_sum, group=metric_group)
            train_loss = loss_sum.item() / (metric_size * args.log_interval)
            loss_sum.zero_()
            step_time = (time.time() - window_start) / args.log_interval
            step_times.append(step_time)
            window_start = time.time()
            if not math.isfinite(train_loss):
                raise RuntimeError(f"non-finite train loss at step {step}")
            train_losses.append(train_loss)
            # How replication results landed on this rank (asynchronous replicators).
            repl_stats = model.replicator.pop_stats() if model.replicator is not None else {}
            applied = sum(repl_stats.values())
            log({
                **{f"repl/{k}_frac": v / applied for k, v in repl_stats.items() if applied},
                "train/loss": train_loss,
                "train/lr": lr_at(step, args),
                "train/tokens": (step + 1) * tokens_per_step,
                "perf/step_time_sec": step_time,
                "perf/tokens_per_sec": tokens_per_step / step_time,
                "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            }, step + 1)
            if rank == 0:
                print(f"step {step + 1:6d}  loss {train_loss:.4f}  "
                      f"{step_time:.3f}s/step  lr {lr_at(step, args):.2e}", flush=True)

        if (args.ckpt_dir and (step + 1) % args.ckpt_interval == 0
                and step + 1 < args.train_iters):
            ckpt_start = time.time()
            save_training_checkpoint(args.ckpt_dir, step + 1, model, optimizer, {
                "wandb_id": wandb_run.id if wandb_run is not None else None,
                "step_times": step_times, "train_losses": train_losses,
            }, rank)
            window_start += time.time() - ckpt_start  # Keep checkpointing out of step times.

    val_loss = evaluate(model, data, args, metric_rank, metric_size, device, metric_group)
    log({"val/loss": val_loss}, args.train_iters)
    if wandb_run is not None:
        steady = step_times[len(step_times) // 10:] or step_times
        wandb_run.summary["final_val_loss"] = val_loss
        wandb_run.summary["median_step_time_sec"] = float(np.median(steady))
        # Training time without evaluation and startup, and the end-of-training loss.
        wandb_run.summary["train_hours"] = sum(step_times) * args.log_interval / 3600
        wandb_run.summary["final_train_loss"] = float(np.mean(train_losses[-10:]))
    if rank == 0:
        print(f"done  val_loss {val_loss:.4f}", flush=True)
    if args.save_dir:
        # The gather needs fresh communication buffers; with the optimizer
        # state still resident the GPUs can be too full to allocate them.
        optimizer.state.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        save_hf_checkpoint(model, args.model, args.save_dir, rank)
    if wandb_run is not None:
        wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
