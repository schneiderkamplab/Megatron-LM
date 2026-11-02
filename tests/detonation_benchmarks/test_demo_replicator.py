#!/usr/bin/env python3
"""Multi-GPU test for synchronous vs. one-step-delayed DeMoBucketReplicator.

Every rank uses the same gradients, so the averaged update equals the
locally compressed one and can be computed with the replicator's own DCT and
top-k helpers. Synchronous mode must apply step s's update in step s; async
mode in step s + 1 (zeros in step 0).

Usage:
    torchrun --nproc-per-node=2 tests/detonation_benchmarks/test_demo_replicator.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

REPL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "megatron", "core", "distributed", "fsdp", "src", "megatron_fsdp", "replicators",
    )
)
sys.path.insert(0, os.path.dirname(REPL_DIR))

from replicators.compression_utils import (  # noqa: E402
    DCTBufferCompress,
    DCTBufferTransform,
)
from replicators.demo import DeMoBucketReplicator  # noqa: E402

NUMEL = 4096
TOPK, CHUNK, DECAY, LR = 4, 64, 0.9, 0.5
NUM_STEPS = 6


def make_grad(step: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(step)
    return torch.randn(NUMEL, generator=g).to(device)


def reference_updates(device):
    """The compressed update transmitted at each step (identical on all ranks)."""
    transform = DCTBufferTransform({0: NUMEL}, CHUNK, torch.float32, device)
    delta = torch.zeros(NUMEL, device=device)
    updates = []
    for step in range(NUM_STEPS):
        delta.mul_(DECAY).add_(make_grad(step, device), alpha=LR)
        encoded = transform.encode(0, delta)
        idx, val = DCTBufferCompress.compress(encoded, TOPK, chunk_size=transform.chunk_sizes[0])
        sent = transform.decode(
            0, DCTBufferCompress.decompress(idx, val, NUMEL, device, torch.float32)
        )
        delta.sub_(sent)
        updates.append(sent.clone())
    return updates


def run(synchronous: bool, device):
    repl = DeMoBucketReplicator(compression_decay=DECAY, compression_topk=TOPK,
                                compression_chunk=CHUNK, synchronous=synchronous)
    repl.init(dist.group.WORLD, {0: NUMEL}, torch.float32, device, lambda: LR)
    stream = torch.cuda.Stream()
    outputs = []
    for step in range(NUM_STEPS):
        grad = make_grad(step, device)
        stream.wait_stream(torch.cuda.current_stream())
        event = repl.replicate_bucket_group([0], lambda _: grad, stream)
        event.synchronize()
        torch.cuda.synchronize()
        outputs.append(grad.clone())
    return outputs


def check(name, outputs, expected, rank):
    for step, (out, exp) in enumerate(zip(outputs, expected)):
        if not torch.allclose(out, exp, atol=1e-5, rtol=1e-4):
            err = (out - exp).abs().max().item()
            raise AssertionError(f"[{name}] rank {rank} step {step}: max err {err:.3e}")
    if rank == 0:
        print(f"  PASS {name}")


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    updates = reference_updates(device)
    check("synchronous demo applies step s in step s", run(True, device), updates, rank)
    delayed = [torch.zeros(NUMEL, device=device)] + updates[:-1]
    check("async demo applies step s in step s + 1", run(False, device), delayed, rank)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("All demo replicator tests passed.")


if __name__ == "__main__":
    main()
