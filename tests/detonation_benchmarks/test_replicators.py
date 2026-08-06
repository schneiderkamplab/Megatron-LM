#!/usr/bin/env python3
"""Standalone test for DeToNation replicators on 2 GPUs.

Exercises each replicator strategy (none, full, demo, slicing, striding,
random) with a small fake model — no HuggingFace download needed.

Usage:
    torchrun --nproc-per-node=2 tests/detonation_benchmarks/test_replicators.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist

# Add the replicators package to path so we can import directly.
REPL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "megatron", "core", "distributed", "fsdp", "src", "megatron_fsdp", "replicators",
)
REPL_DIR = os.path.abspath(REPL_DIR)
sys.path.insert(0, os.path.dirname(REPL_DIR))

from replicators import get_replicator  # noqa: E402

# ---------------------------------------------------------------------------
# Fake model: a tiny 2-layer MLP with Qwen-like architecture stubs.
# ---------------------------------------------------------------------------

class FakeQwenMLP(torch.nn.Module):
    """Small MLP that mimics a Qwen-style decoder layer's FFN.

    Parameters have sizes chosen to be divisible by common chunk sizes
    (64, 128) so DCT/slicing/striding replicators work cleanly.
    """

    def __init__(self, hidden_size: int = 512, intermediate_size: int = 1024):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FakeQwenModel(torch.nn.Module):
    """Tiny model with an embedding + 2 MLP layers + output projection.

    Total params ~2M. Gradients are computed via a simple MSE loss
    against random targets.
    """

    def __init__(self, vocab_size: int = 1024, hidden_size: int = 512,
                 intermediate_size: int = 1024, seq_length: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.layer0 = FakeQwenMLP(hidden_size, intermediate_size)
        self.layer1 = FakeQwenMLP(hidden_size, intermediate_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.layer0(x)
        x = self.layer1(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def get_param_buffers(model: torch.nn.Module, device: torch.device):
    """Flatten all trainable parameters into a single 1-D buffer.

    Returns:
        (buffer, param_info) where buffer is a 1-D tensor on `device`
        and param_info maps bucket_id -> (numel, slice).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    flat = torch.cat([p.data.view(-1) for p in params]).clone()

    # Gradient buffer: same layout but for gradients.
    grad_flat = torch.zeros_like(flat)

    bucket_sizes = {0: flat.numel()}
    return flat, grad_flat, bucket_sizes, params


def run_replicator_test(
    strategy: str,
    model: torch.nn.Module,
    world_size: int,
    rank: int,
    device: torch.device,
    num_steps: int = 5,
    **kwargs,
):
    """Test a single replicator strategy.

    Simulates the Megatron FSDP pipeline's interaction with a replicator:
      1. Create replicator via get_replicator()
      2. init() with bucket sizes, process group, etc.
      3. For each step:
         a. pre_step()
         b. Write random gradients to the gradient buffer
         c. replicate_bucket_group()
         d. Wait on the returned event
         e. post_step()
    """
    # Flatten model parameters into a gradient buffer.
    _, grad_buffer, bucket_sizes, _ = get_param_buffers(model, device)

    # Build the replicator.
    replicator = get_replicator(
        strategy=strategy,
        compression_decay=kwargs.get("decay", 0.999),
        compression_topk=kwargs.get("topk", 32),
        compression_chunk=kwargs.get("chunk", 64),
        compression_rate=kwargs.get("rate", 0.1),
        seed=kwargs.get("seed", 42),
    )

    # Initialize.
    lr = 1e-4
    lr_provider = lambda: lr

    pg = dist.group.WORLD
    replicator.init(
        process_group=pg,
        bucket_sizes=bucket_sizes,
        dtype=torch.float32,
        device=device,
        lr_provider=lr_provider,
    )

    # Create a stream for communication.
    comm_stream = torch.cuda.Stream()

    # Simulate training steps.
    for step in range(num_steps):
        replicator.pre_step()

        # Write deterministic-but-varied "gradients" to the buffer.
        # Each rank writes different values to ensure all_reduce changes them.
        torch.manual_seed(42 + step * 100 + rank)
        grad_buffer.copy_(torch.randn_like(grad_buffer) * 0.01)

        def get_buffer_fn(bucket_id, _buf=grad_buffer):
            return _buf

        event = replicator.replicate_bucket_group(
            bucket_group=[0],
            get_buffer_fn=get_buffer_fn,
            stream=comm_stream,
        )

        # Wait for the event if one was returned.
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        torch.cuda.synchronize()

        # Check the gradient buffer is not all-zeros (after step 0).
        if step > 0:
            nonzero = (grad_buffer.abs() > 0).any().item()
            if not nonzero:
                raise RuntimeError(
                    f"Step {step}: gradient buffer is all zeros after replication"
                )

        replicator.post_step()

    # Final: resolve any remaining pending async operations.
    if hasattr(replicator, '_pending') and replicator._pending:
        for bucket_id in list(replicator._pending.keys()):
            # Force-wait on any pending work.
            pending = replicator._pending[bucket_id]
            if isinstance(pending, tuple):
                for item in pending:
                    if hasattr(item, 'wait'):
                        item.wait()
                    elif isinstance(item, dist.Work):
                        item.wait()

    return True


def main():
    # Init distributed.
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")

    # Build fake model.
    model = FakeQwenModel(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        seq_length=64,
    ).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"DeToNation Replicator Test — 2 GPU")
        print(f"{'='*60}")
        print(f"Model: FakeQwenModel (~{total_params/1e6:.1f}M params)")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    # Test each strategy.
    strategies = [
        ("none", {}),
        ("full", {}),
        ("demo", {"topk": 32, "chunk": 64, "decay": 0.999}),
        ("slicing", {"rate": 0.1, "chunk": 64, "decay": 0.999}),
        ("striding", {"rate": 0.1, "chunk": 64, "decay": 0.999}),
        ("random", {"rate": 0.1, "decay": 0.999, "seed": 42}),
    ]

    results = {}

    for strategy, kwargs in strategies:
        if rank == 0:
            print(f"  Testing '{strategy}'...", end=" ", flush=True)
        try:
            # Rebuild model each time so gradients are fresh.
            test_model = FakeQwenModel(
                vocab_size=1024,
                hidden_size=512,
                intermediate_size=1024,
                seq_length=64,
            ).to(device)

            run_replicator_test(
                strategy=strategy,
                model=test_model,
                world_size=world_size,
                rank=rank,
                device=device,
                num_steps=5,
                **kwargs,
            )
            results[strategy] = "PASS"
            if rank == 0:
                print("PASS")
        except Exception as e:
            results[strategy] = f"FAIL: {e}"
            if rank == 0:
                print(f"FAIL: {e}")
                traceback.print_exc()

        # Barrier between strategies.
        dist.barrier()

    # Summary.
    if rank == 0:
        passed = sum(1 for v in results.values() if v == "PASS")
        total = len(results)
        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed")
        for s, r in results.items():
            status = "✅" if r == "PASS" else "❌"
            print(f"  {status} {s}: {r}")
        print(f"{'='*60}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
