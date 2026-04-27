"""Run a single benchmark: one model x one replicator strategy.

Usage:
    torchrun --nproc-per-node=4 run_benchmark.py \
        --config configs/experiments/smoke_test.yaml \
        --model-idx 0 \
        --replicator-idx 1

The script uses native Megatron-LM APIs to load models from HuggingFace and
Megatron-LM's DistributedDataParallelConfig for replicator settings.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Add this directory to path for config_parser import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config_parser import parse_config


class MockDataset(Dataset):
    """Simple mock dataset for benchmarking."""

    def __init__(self, seq_length, num_samples=1000):
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "tokens": torch.randint(0, 1000, (self.seq_length,)),
            "attention_mask": torch.ones(self.seq_length),
            "position_ids": torch.arange(self.seq_length),
        }


def initialize_model_parallel(model_cfg):
    """Initialize Megatron model parallel groups."""
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=model_cfg.tp,
        pipeline_model_parallel_size=model_cfg.pp,
    )


def build_model(model_cfg):
    """Build a HuggingFace model using native Megatron APIs."""
    # Import directly from the module file since it's not exported from __init__.py
    from megatron.core.models.huggingface.module import AutoHuggingFaceModel
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.enums import ModelType

    # Create a minimal config
    config = TransformerConfig(
        num_layers=1,  # Minimal for benchmarking
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        tensor_model_parallel_size=model_cfg.tp,
        pipeline_model_parallel_size=model_cfg.pp,
    )

    # Add huggingface model path to config
    # AutoHuggingFaceModel expects this attribute
    config.huggingface_model_name_or_path = model_cfg.hf_id

    # Build the model - AutoHuggingFaceModel works with ANY HF model
    model = AutoHuggingFaceModel(config)

    # Add model_type attribute required by Megatron's training infrastructure
    # This is expected by get_model_type() in pipeline_parallel/schedules.py
    model.model_type = ModelType.encoder_or_decoder

    return model, config


def setup_ddp(model, config, replicator_cfg):
    """Set up DistributedDataParallel with replication strategy."""
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )

    ddp_config = DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        replication_strategy=replicator_cfg.strategy,
        replication_topk=replicator_cfg.topk,
        replication_chunk=replicator_cfg.chunk,
        replication_decay=replicator_cfg.decay,
        replication_rate=replicator_cfg.rate,
        replication_seed=replicator_cfg.seed,
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )

    model = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=model,
    )

    return model


def create_forward_step_func(model_cfg):
    """Create forward step function for training."""

    def forward_step_func(data_iterator, model):
        """Forward step function that computes model output and returns loss function."""

        def loss_func(output_tensor):
            """Simple loss function - output_tensor is now a tensor, not HF model output."""
            # Compute mock loss from the tensor output
            loss = output_tensor.float().mean()

            # Reduce loss across data parallel ranks
            reduced_loss = loss.clone()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(reduced_loss)
                reduced_loss = reduced_loss / torch.distributed.get_world_size()
            return reduced_loss, {"lm loss": reduced_loss}

        # Get batch from data iterator
        try:
            batch = next(data_iterator)
        except StopIteration:
            # Reinitialize iterator if exhausted
            return None

        tokens = batch["tokens"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        # Forward pass - handle both causal LM and encoder models
        try:
            # Try causal LM style (GPT-2, Llama, etc.) with labels for loss
            model_output = model(
                input_ids=tokens,
                attention_mask=attention_mask,
                labels=tokens,
            )
        except Exception as e:
            try:
                # Fallback: encoder style (BERT, etc.) or models that don't accept labels
                model_output = model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                )
            except Exception as e2:
                # Last resort: pass as positional args (for AutoHuggingFaceModel)
                model_output = model(tokens, attention_mask=attention_mask)

        # Extract the right tensor for Megatron's training loop
        # Megatron expects a tensor, not a BaseModelOutput object
        if hasattr(model_output, 'logits'):
            output_tensor = model_output.logits
        elif hasattr(model_output, 'last_hidden_state'):
            output_tensor = model_output.last_hidden_state
        elif isinstance(model_output, dict):
            # Handle dictionary outputs
            output_tensor = model_output.get('logits', model_output.get('last_hidden_state', list(model_output.values())[0]))
        else:
            # Assume it's already a tensor
            output_tensor = model_output

        return output_tensor, loss_func

    return forward_step_func


def run_benchmark(model_cfg, replicator_cfg, train_cfg, output_dir):
    """Run benchmark using native Megatron training loop."""
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # Initialize model parallel
    initialize_model_parallel(model_cfg)
    model_parallel_cuda_manual_seed(123)

    # Build model
    model, config = build_model(model_cfg)
    model = model.cuda()

    # Setup DDP
    model = setup_ddp(model, config, replicator_cfg)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Setup data
    dataset = MockDataset(seq_length=model_cfg.seq_length)
    dataloader = DataLoader(dataset, batch_size=train_cfg.micro_batch_size, shuffle=True)

    # Get forward backward function
    forward_backward_func = get_forward_backward_func()
    forward_step_func = create_forward_step_func(model_cfg)

    # Training loop
    start_time = time.time()

    for iteration in range(train_cfg.train_iters):
        optimizer.zero_grad()

        data_iterator = iter(dataloader)

        # Run forward/backward
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=1,
            seq_length=model_cfg.seq_length,
            micro_batch_size=train_cfg.micro_batch_size,
            decoder_seq_length=model_cfg.seq_length,
            forward_only=False,
        )

        # Finalize gradients and update
        finalize_model_grads([model])
        optimizer.step()

        if iteration % 10 == 0:
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                print(f"Iteration {iteration}: Losses: {losses_reduced}")

    elapsed = time.time() - start_time

    # Write metrics
    metrics = {
        "total_time_sec": round(elapsed, 2),
        "train_iters": train_cfg.train_iters,
        "strategy": replicator_cfg.strategy,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"\nBenchmark complete. Metrics saved to {metrics_path}")
        print(json.dumps(metrics, indent=2))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run a single DeToNATION benchmark")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--model-idx", type=int, default=0, help="Index into models list")
    parser.add_argument("--replicator-idx", type=int, default=0, help="Index into replicators list")
    parser.add_argument("--hardware-idx", type=int, default=0, help="Index into hardware list")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for metrics and logs (default: auto-generated)",
    )
    args = parser.parse_args()

    # Parse experiment config
    exp_cfg = parse_config(args.config)

    model_cfg = exp_cfg.models[args.model_idx]
    replicator_cfg = exp_cfg.replicators[args.replicator_idx]
    hw_cfg = exp_cfg.hardware[args.hardware_idx]

    # Set output directory
    if args.output_dir is None:
        output_dir = Path(
            f"results/{exp_cfg.name}/{model_cfg.name}/{replicator_cfg.strategy}"
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"Model: {model_cfg.name} ({model_cfg.hf_id})")
        print(f"Replicator: {replicator_cfg.strategy}")
        print(f"Hardware: {hw_cfg.nodes} nodes x {hw_cfg.gpus_per_node} GPUs")
        print(f"Output: {output_dir}")

    # Run benchmark with native Megatron APIs
    run_benchmark(model_cfg, replicator_cfg, exp_cfg.train, output_dir)


if __name__ == "__main__":
    main()
