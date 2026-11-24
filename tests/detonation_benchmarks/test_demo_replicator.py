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


def run(synchronous: bool, device, wait_s=None, delays=None):
    """Replicated gradient per step. With `delays`, the gathers posted at step s
    only report completion from step s + delays[s] on."""
    repl = DeMoBucketReplicator(compression_decay=DECAY, compression_topk=TOPK,
                                compression_chunk=CHUNK, synchronous=synchronous)
    repl.init(dist.group.WORLD, {0: NUMEL}, torch.float32, device, lambda: LR)
    original_all_gather = dist.all_gather
    clock = {"s": 0}

    def delayed_all_gather(*args, **kwargs):
        work = original_all_gather(*args, **kwargs)
        if kwargs.get("async_op") and delays is not None:
            return DelayedWork(work, clock["s"] + delays[clock["s"]], clock)
        return work

    stream = torch.cuda.Stream()
    outputs = []
    dist.all_gather = delayed_all_gather
    try:
        for step in range(NUM_STEPS):
            clock["s"] = step
            grad = make_grad(step, device)
            stream.wait_stream(torch.cuda.current_stream())
            event = repl.replicate_bucket_group([0], lambda _: grad, stream)
            event.synchronize()
            if wait_s is not None:
                # Let delay-0 collectives finish, as a network that keeps up with
                # the backward pass would.
                torch.cuda.synchronize()
                repl.finalize_step([0], lambda _: grad, wait_s)
            torch.cuda.synchronize()
            outputs.append(grad.clone())
    finally:
        dist.all_gather = original_all_gather
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
    # With finalize_step the gathers complete within the step, so adaptive
    # async demo matches the synchronous result.
    check("adaptive demo applies step s in step s", run(False, device, wait_s=2.0),
          updates, rank)
    # Late gathers are added at the next step, on top of that step's own
    # update if it is in time; every update is applied exactly once.
    delays = [0, 1, 0, 1, 1, 0]
    expected = [torch.zeros(NUMEL, device=device) for _ in range(NUM_STEPS)]
    for step, delay in enumerate(delays):
        if step + delay < NUM_STEPS:
            expected[step + delay] += updates[step]
    check("adaptive demo with late gathers", run(False, device, wait_s=0.3, delays=delays),
          expected, rank)
    check("adaptive demo, no wait, with late gathers",
          run(False, device, wait_s=0.0, delays=delays), expected, rank)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("All demo replicator tests passed.")


if __name__ == "__main__":
    main()
