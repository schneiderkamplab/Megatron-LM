"""Run a single benchmark: one model x one replicator strategy.

Usage:
    torchrun --nproc-per-node=2 run_benchmark.py \
        --config configs/experiments/smoke_test.yaml \
        --model-idx 0 \
        --replicator-idx 1

Uses the native Megatron-FSDP ``fully_shard_model`` / ``fully_shard_optimizer``
APIs so the DeToNation replicator is exercised through the real gradient
reduction pipeline (``GradReducePipeline`` -> ``outer_fsdp_group_grad_reduce``).

Supports two model modes:
  - hf_id == "fake": in-process FakeModel (~4.2M params, no download)
  - hf_id == "<HF model id>": loaded via transformers directly (no AutoBridge)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim import AdamW

# Add this directory to path for config_parser import.
sys.path.insert(0, str(Path(__file__).parent))
from config_parser import parse_config  # noqa: E402

# ---------------------------------------------------------------------------
# Fake model: tiny transformer-like model (no HuggingFace download).
# ---------------------------------------------------------------------------

class FakeMLP(torch.nn.Module):
    """Small MLP mimicking a decoder-layer FFN."""

    def __init__(self, hidden_size: int = 512, intermediate_size: int = 1024):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FakeModel(torch.nn.Module):
    """Embedding + 2 MLP layers + output projection (~4.2M params)."""

    def __init__(self, vocab_size: int = 1024, hidden_size: int = 512,
                 intermediate_size: int = 1024):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.layer0 = FakeMLP(hidden_size, intermediate_size)
        self.layer1 = FakeMLP(hidden_size, intermediate_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.layer0(x)
        x = self.layer1(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Model loading.
# ---------------------------------------------------------------------------

def load_fake_model(device: torch.device):
    """Build the in-process fake model."""
    model = FakeModel(vocab_size=1024, hidden_size=512, intermediate_size=1024)
    model = model.to(device)
    fsdp_unit_modules = [FakeMLP]
    vocab_size = 1024
    return model, fsdp_unit_modules, vocab_size


def load_hf_model(hf_id: str, device: torch.device):
    """Load a HuggingFace model directly via transformers (no AutoBridge).

    Auto-detects the decoder layer class for FSDP unit modules.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = model.to(device)

    # Auto-detect decoder layer class for fsdp_unit_modules.
    fsdp_unit_modules = []
    for module in model.modules():
        cls_name = type(module).__name__
        if "DecoderLayer" in cls_name or cls_name.endswith("Block"):
            if type(module) not in fsdp_unit_modules:
                fsdp_unit_modules.append(type(module))

    vocab_size = config.vocab_size
    return model, fsdp_unit_modules, vocab_size


# ---------------------------------------------------------------------------
# Distributed init.
# ---------------------------------------------------------------------------

def init_distributed():
    """Initialize torch.distributed and set the CUDA device."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------

def run_benchmark(model_cfg, replicator_cfg, train_cfg, output_dir, rank, world_size):
    """Run a training benchmark with the given replicator strategy.

    Uses ``fully_shard_model`` + ``fully_shard_optimizer`` so the replicator
    is wired through the real ``GradReducePipeline`` standalone-replication
    path (``outer_fsdp_group_grad_reduce=True`` even without HSDP).
    """
    from megatron.core.distributed.fsdp.src.megatron_fsdp import (
        MixedPrecisionPolicy,
        fully_shard_model,
        fully_shard_optimizer,
    )

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    # Load model (fake or real HF).
    if model_cfg.hf_id == "fake":
        model, fsdp_unit_modules, vocab_size = load_fake_model(device)
    else:
        model, fsdp_unit_modules, vocab_size = load_hf_model(model_cfg.hf_id, device)

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        model_desc = "FakeModel" if model_cfg.hf_id == "fake" else model_cfg.hf_id
        print(f"  Model: {model_desc} (~{total_params / 1e6:.1f}M params)")
        print(f"  FSDP unit modules: {[m.__name__ for m in fsdp_unit_modules]}")

    # Fully-shard the model with the replicator strategy.
    model = fully_shard_model(
        model,
        zero_dp_strategy="optim_grads_params",
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        sync_model_each_microbatch=True,
        fsdp_unit_modules=fsdp_unit_modules,
        replication_strategy=replicator_cfg.strategy,
        replication_decay=replicator_cfg.decay,
        replication_topk=replicator_cfg.topk,
        replication_chunk=replicator_cfg.chunk,
        replication_rate=replicator_cfg.rate,
        replication_seed=replicator_cfg.seed,
        mixed_precision_policy=MixedPrecisionPolicy(),
        device=device,
    )

    # Optimizer on the FSDP-managed parameters.
    optimizer = AdamW(model.parameters(), lr=1e-4)
    optimizer = fully_shard_optimizer(optimizer)

    # Random token data.
    seq_length = model_cfg.seq_length
    batch_size = train_cfg.micro_batch_size

    # Training loop.
    start_time = time.time()
    losses: list[float] = []

    for iteration in range(train_cfg.train_iters):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        labels = input_ids.clone()

        logits = model(input_ids)
        if hasattr(logits, "logits"):
            logits = logits.logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        losses.append(loss_val)

        if iteration % 10 == 0 and rank == 0:
            print(f"    iter {iteration:4d}  loss={loss_val:.4f}")

    elapsed = time.time() - start_time

    metrics = {
        "model": model_cfg.name,
        "hf_id": model_cfg.hf_id,
        "strategy": replicator_cfg.strategy,
        "train_iters": train_cfg.train_iters,
        "total_time_sec": round(elapsed, 2),
        "throughput_iters_per_sec": round(train_cfg.train_iters / elapsed, 2)
        if elapsed > 0 else 0,
        "final_loss": round(losses[-1], 6) if losses else None,
        "world_size": world_size,
        "model_params_millions": round(total_params / 1e6, 2),
    }

    if rank == 0:
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics written to {metrics_path}")
        print(json.dumps(metrics, indent=2))

    return metrics


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a single DeToNATION benchmark with Megatron-FSDP"
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--model-idx", type=int, default=0)
    parser.add_argument("--replicator-idx", type=int, default=0)
    parser.add_argument("--hardware-idx", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    exp_cfg = parse_config(args.config)
    model_cfg = exp_cfg.models[args.model_idx]
    replicator_cfg = exp_cfg.replicators[args.replicator_idx]
    hw_cfg = exp_cfg.hardware[args.hardware_idx]

    if args.output_dir is None:
        output_dir = Path(
            f"results/{exp_cfg.name}/{model_cfg.name}/{replicator_cfg.strategy}"
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank, world_size, local_rank = init_distributed()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"DeToNATION Benchmark (Megatron-FSDP)")
        print(f"{'=' * 60}")
        print(f"  Experiment:  {exp_cfg.name}")
        print(f"  Model:       {model_cfg.name} ({model_cfg.hf_id})")
        print(f"  Replicator:  {replicator_cfg.strategy}")
        print(f"  Hardware:    {hw_cfg.nodes} nodes x {hw_cfg.gpus_per_node} GPUs")
        print(f"  World size:  {world_size}")
        print(f"  Output:      {output_dir}")
        print(f"{'=' * 60}\n")

    try:
        run_benchmark(model_cfg, replicator_cfg, exp_cfg.train, output_dir, rank, world_size)
        if rank == 0:
            print(f"\n  Benchmark '{replicator_cfg.strategy}' completed successfully.\n")
    except Exception:
        if rank == 0:
            print(f"\n  Benchmark '{replicator_cfg.strategy}' FAILED.\n")
            traceback.print_exc()
        dist.barrier()
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
