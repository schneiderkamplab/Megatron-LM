#!/usr/bin/env python3
"""Multi-GPU test for the step-tagged RandomBucketReplicator.

Checks that delayed results land on the indices of the step they were sent
for, that results older than max_delay are dropped, and that a result whose
step tag differs across ranks is dropped.

Usage:
    torchrun --nproc-per-node=2 tests/detonation_benchmarks/test_random_replicator.py
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

import replicators.random as random_module  # noqa: E402
from replicators.random import RandomBucketReplicator  # noqa: E402

NUMEL = 4096
RATE = 0.125
DECAY = 0.9
LR = 0.5


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


def simulate(world_size, num_steps, delays, max_delay, device, seed=42):
    """Reference: per-step expected replicated gradient on every rank.

    The result sent at step s completes at step s + delays[s]. Collectives
    complete in issue order, so a result is applied at the first step where it
    and every earlier result have completed, unless it is more than max_delay
    steps old by then, in which case it is dropped.
    """
    repl = RandomBucketReplicator(compression_decay=DECAY, compression_rate=RATE,
                                  seed=seed, max_delay=max_delay)
    repl._device = device
    repl._rng = torch.Generator(device=device)
    deltas = [torch.zeros(NUMEL, device=device) for _ in range(world_size)]
    expected = [torch.zeros(NUMEL, device=device) for _ in range(num_steps)]
    in_flight = []  # (sent_step, values), oldest first
    for step in range(num_steps):
        while in_flight and max_delay > 0:
            sent_step, values = in_flight[0]
            if step - sent_step > max_delay:
                in_flight.pop(0)
            elif step >= sent_step + delays[sent_step]:
                in_flight.pop(0)
                expected[step] += values
            else:
                break
        idx, num_valid = repl._select_indices(0, step, NUMEL)
        idx = idx.long()
        valid = idx[:num_valid]
        sent = torch.zeros(NUMEL, device=device)
        for r in range(world_size):
            deltas[r].mul_(DECAY).add_(make_grad(r, step, device), alpha=LR)
            sent[valid] += deltas[r][valid] / world_size
            deltas[r][idx] = 0
        if max_delay == 0:
            expected[step] += sent  # Synchronous: applied in the step it was sent.
        else:
            in_flight.append((step, sent))
    return expected


def run(delays, max_delay, rank, world_size, device, step_offset=0):
    num_steps = len(delays)
    repl = RandomBucketReplicator(compression_decay=DECAY, compression_rate=RATE,
                                  max_delay=max_delay)
    repl.init(dist.group.WORLD, {0: NUMEL}, torch.float32, device, lambda: LR)
    repl._steps[0] += step_offset

    original_all_reduce = dist.all_reduce
    clock = {"s": 0}

    def delayed_all_reduce(*args, **kwargs):
        step = clock["s"]
        return DelayedWork(original_all_reduce(*args, **kwargs), step + delays[step], clock)

    stream = torch.cuda.Stream()
    outputs = []
    dist.all_reduce = delayed_all_reduce
    try:
        for step in range(num_steps):
            clock["s"] = step
            grad = make_grad(rank, step, device)
            stream.wait_stream(torch.cuda.current_stream())
            repl.replicate_bucket_group([0], lambda _: grad, stream).synchronize()
            torch.cuda.synchronize()
            outputs.append(grad.clone())
    finally:
        dist.all_reduce = original_all_reduce
        torch.cuda.synchronize()
    return outputs


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

    # Indices are identical on every rank.
    repl = RandomBucketReplicator(compression_rate=RATE)
    repl.init(dist.group.WORLD, {0: NUMEL}, torch.float32, device, lambda: LR)
    idx, num_valid = repl._select_indices(0, 7, NUMEL)
    idx = idx.long()
    gathered = [torch.empty_like(idx) for _ in range(world_size)]
    dist.all_gather(gathered, idx)
    assert all(torch.equal(g, idx) for g in gathered), "ranks selected different indices"
    assert idx.numel() == num_valid == int(RATE * NUMEL), (idx.numel(), num_valid)
    assert idx.unique().numel() == idx.numel(), "indices must be distinct"
    if rank == 0:
        print("  PASS indices agree across ranks, fixed length")

    n = 12
    cases = {
        "synchronous (max_delay=0)": ([0] * n, 0),
        "one-step delay": ([1] * n, 4),
        "mixed delays within max_delay": ([1, 3, 2, 1, 4, 1, 2, 1, 1, 3, 1, 1], 4),
        "late results dropped": ([1, 6, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1], 4),
    }
    for name, (delays, max_delay) in cases.items():
        expected = simulate(world_size, n, delays, max_delay, device)
        outputs = run(delays, max_delay, rank, world_size, device)
        check(name, outputs, expected, rank)

    # Undersampling forces the zero-padding path.
    original_sigmas = random_module._OVERSAMPLE_SIGMAS
    random_module._OVERSAMPLE_SIGMAS = -6.0
    try:
        _, num_valid = repl._select_indices(0, 0, NUMEL)
        assert num_valid < int(RATE * NUMEL), "padding path not exercised"
        expected = simulate(world_size, n, [1] * n, 4, device)
        outputs = run([1] * n, 4, rank, world_size, device)
        check("zero-padded payloads", outputs, expected, rank)
    finally:
        random_module._OVERSAMPLE_SIGMAS = original_sigmas

    # Ranks disagree about the step: tags mismatch, every result is dropped.
    if world_size > 1:
        outputs = run([1] * n, 4, rank, world_size, device, step_offset=rank)
        check("mismatched tags dropped", outputs,
              [torch.zeros(NUMEL, device=device)] * n, rank)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("All random replicator tests passed.")


if __name__ == "__main__":
    main()
