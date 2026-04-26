"""Run a single benchmark: one model x one replicator strategy.

Usage:
    torchrun --nproc-per-node=4 run_benchmark.py \
        --config configs/experiments/smoke_test.yaml \
        --model-idx 0 \
        --replicator-idx 1

The script uses Megatron-Bridge to load models from HuggingFace and
Megatron-LM's DistributedDataParallelConfig for replicator settings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add this directory to path for config_parser import
sys.path.insert(0, str(Path(__file__).parent))

from config_parser import parse_config


def build_bridge_config(model_cfg, replicator_cfg, train_cfg):
    """Build a Megatron-Bridge ConfigContainer with the given model and replicator settings.

    Returns a ConfigContainer ready for pretrain().
    """
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import ConfigContainer

    # Load model via Bridge
    provider = AutoBridge.from_hf_pretrained(model_cfg.hf_id).to_megatron_provider(
        load_weights=False
    )
    provider.tensor_model_parallel_size = model_cfg.tp
    provider.pipeline_model_parallel_size = model_cfg.pp

    # Build config container with defaults
    cfg = _default_config_container()
    cfg.model = provider

    # Training settings
    cfg.train.train_iters = train_cfg.train_iters
    cfg.train.global_batch_size = train_cfg.global_batch_size
    cfg.train.micro_batch_size = train_cfg.micro_batch_size

    # DDP / replicator settings
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.replication_strategy = replicator_cfg.strategy
    cfg.ddp.replication_topk = replicator_cfg.topk
    cfg.ddp.replication_chunk = replicator_cfg.chunk
    cfg.ddp.replication_decay = replicator_cfg.decay
    cfg.ddp.replication_rate = replicator_cfg.rate
    cfg.ddp.replication_seed = replicator_cfg.seed

    # Dataset: use mock data (no real dataset needed)
    cfg.dataset.blend = None
    cfg.dataset.seq_length = model_cfg.seq_length

    return cfg


def _default_config_container():
    """Create a ConfigContainer with minimal defaults for benchmarking."""
    from megatron.bridge.training.config import (
        ConfigContainer,
        CheckpointConfig,
        DistributedInitConfig,
        LoggerConfig,
        RNGConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.optimizer import OptimizerConfig

    cfg = ConfigContainer(
        rng=RNGConfig(),
        train=TrainingConfig(),
        ddp=DistributedDataParallelConfig(),
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(),
        checkpoint=CheckpointConfig(),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        dist=DistributedInitConfig(),
    )
    # Minimal training defaults for benchmarking
    cfg.train.log_interval = 1
    cfg.train.eval_iters = 0
    cfg.train.eval_interval = None
    cfg.checkpoint.save_interval = None  # Don't save checkpoints during benchmarks
    return cfg


def run_benchmark(cfg, output_dir):
    """Run pretrain and collect metrics.

    Calls Bridge's pretrain() which runs the full training loop.
    After completion, collects timing metrics from TensorBoard logs.
    """
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.gpt_step import forward_step

    # Set up output directory for logs
    cfg.train.tensorboard_dir = str(output_dir / "tb_logs")

    start_time = time.time()
    pretrain(cfg, forward_step)
    elapsed = time.time() - start_time

    # Write summary metrics
    metrics = {
        "total_time_sec": round(elapsed, 2),
        "train_iters": cfg.train.train_iters,
        "strategy": cfg.ddp.replication_strategy,
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

    # Build config and run
    cfg = build_bridge_config(model_cfg, replicator_cfg, exp_cfg.train)
    run_benchmark(cfg, output_dir)


if __name__ == "__main__":
    main()
