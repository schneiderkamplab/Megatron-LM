#!/usr/bin/env python3
"""Multi-GPU test for FullBucketReplicator's three modes.

Synchronous applies the averaged gradient in the same step; asynchronous
applies the previous step's average (one-step delay); adaptive (finalize_step)
applies the current step's average when the collective completes in time, and
keeps the local gradient for the step whenever it does not.

Usage:
    torchrun --nproc-per-node=2 tests/detonation_benchmarks/test_full_replicator.py
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

from replicators.full import FullBucketReplicator  # noqa: E402

NUMEL = 4096


class DelayedWork:
    """Wraps a Work so is_completed() is False until `ready_step` is reached."""

    def __init__(self, work, ready_step, clock):
        self.work = work
        self.ready_step = ready_step
        self.clock = clock

    def is_completed(self):
        return self.clock["s"] >= self.ready_step and self.work.is_completed()

    def wait(self):
        return self.work.wait()


def make_grad(rank: int, step: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1000 * step + rank)
    return torch.randn(NUMEL, generator=g).to(device)


def make_avg(world_size: int, step: int, device) -> torch.Tensor:
    return torch.stack([make_grad(r, step, device) for r in range(world_size)]).mean(0)


def run(delays, rank, world_size, device, synchronous=False, wait_s=0.0):
    repl = FullBucketReplicator(synchronous=synchronous)
    repl.init(dist.group.WORLD, {0: NUMEL}, torch.float32, device, lambda: 0.0)

    original_all_reduce = dist.all_reduce
    clock = {"s": 0}

    def delayed_all_reduce(*args, **kwargs):
        work = original_all_reduce(*args, **kwargs)
        if kwargs.get("async_op"):
            step = clock["s"]
            return DelayedWork(work, step + delays[step], clock)
        return work

    stream = torch.cuda.Stream()
    outputs = []
    dist.all_reduce = delayed_all_reduce
    try:
        for step in range(len(delays)):
            clock["s"] = step
            grad = make_grad(rank, step, device)
            stream.wait_stream(torch.cuda.current_stream())
            repl.replicate_bucket_group([0], lambda _: grad, stream).synchronize()
            if wait_s > 0:
                repl.finalize_step([0], lambda _: grad, wait_s)
            torch.cuda.synchronize()
            outputs.append(grad.clone())
    finally:
        dist.all_reduce = original_all_reduce
        torch.cuda.synchronize()
    return outputs


def expected_adaptive(delays, rank, world_size, device):
    """Reference for the adaptive mode: fresh average when the collective
    completes within its step, stale average when one is pending from before,
    the local gradient after a fresh step, zero only at the very start."""
    out = []
    pending = None
    fresh = False
    for step, delay in enumerate(delays):
        if pending is not None:
            value = make_avg(world_size, pending, device)
            pending = None
        elif fresh:
            value = make_grad(rank, step, device)
            fresh = False
        else:
            value = torch.zeros(NUMEL, device=device)
        pending = step
        if delay == 0:  # Completes within the step: finalize applies it.
            value = make_avg(world_size, step, device)
            pending = None
            fresh = True
        out.append(value)
    return out


def check(name, outputs, expected, rank):
    for step, (out, exp) in enumerate(zip(outputs, expected)):
        if not torch.allclose(out, exp, atol=1e-5, rtol=1e-5):
            err = (out - exp).abs().max().item()
            raise AssertionError(f"[{name}] rank {rank} step {step}: max err {err:.3e}")
    if rank == 0:
        print(f"  PASS {name}")


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    n = 8
    avg = [make_avg(world_size, s, device) for s in range(n)]
    zero = torch.zeros(NUMEL, device=device)

    outputs = run([0] * n, rank, world_size, device, synchronous=True)
    check("synchronous", outputs, avg, rank)

    outputs = run([1] * n, rank, world_size, device)
    check("asynchronous one-step delay", outputs, [zero] + avg[:-1], rank)

    outputs = run([0] * n, rank, world_size, device, wait_s=0.3)
    check("adaptive: all results in time", outputs,
          expected_adaptive([0] * n, rank, world_size, device), rank)

    delays = [0, 1, 0, 0, 2, 0, 1, 0]
    outputs = run(delays, rank, world_size, device, wait_s=0.3)
    check("adaptive: mixed same-step and late", outputs,
          expected_adaptive(delays, rank, world_size, device), rank)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("All full replicator tests passed.")


if __name__ == "__main__":
    main()
