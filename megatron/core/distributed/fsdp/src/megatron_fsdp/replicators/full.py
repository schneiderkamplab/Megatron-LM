# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from .base import BucketReplicator

__all__ = ["FullBucketReplicator"]


class FullBucketReplicator(BucketReplicator):
    """Full gradient replication via async all-reduce.

    Sends the complete sharded gradient buffer across the inter-node group
    using an asynchronous all_reduce. Results from the previous step are
    resolved before posting new communications (one-step-delay pattern).

    With synchronous=True the all_reduce blocks and the result is applied in
    the same step, through the same clone/copy path as the asynchronous mode,
    so the two modes differ only in the one-step delay. It is then numerically
    equivalent to FullSyncBucketReplicator (which reduces in place without the
    staging copies) and to the standard Megatron-LM DP-Outer all_reduce.
    """

    def __init__(self, synchronous: bool = False):
        self.synchronous = synchronous
        self._process_group: Optional[dist.ProcessGroup] = None
        self._world_size: int = 1
        self._pending: Dict[int, Tuple[dist.Work, torch.Tensor]] = {}
        # Buckets whose result was already applied in the step it was sent
        # (by finalize_step), so the next replicate call must not reapply or
        # zero-fill them.
        self._fresh: Set[int] = set()

    def init(
        self,
        process_group: dist.ProcessGroup,
        bucket_sizes: Dict[int, int],
        dtype: torch.dtype,
        device: torch.device,
        lr_provider: Callable[[], float],
    ) -> None:
        self._process_group = process_group
        self._world_size = dist.get_world_size(process_group)

    def pre_step(self) -> None:
        pass

    def post_step(self) -> None:
        pass

    def wait_pending(self, bucket_id: int) -> bool:
        # Resolution is handled inline in replicate_bucket_group.
        return bucket_id in self._pending

    def finalize_step(
        self,
        bucket_ids: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        deadline_s: float,
    ) -> None:
        """Apply this step's all-reduce results if they complete in time, making
        the replicator synchronous whenever the network keeps up. A bucket that
        misses the deadline keeps the local gradient for this step and its
        global result is applied at the next step as usual."""
        if self.synchronous or self._world_size <= 1:
            return
        end = time.monotonic() + deadline_s
        remaining = [b for b in bucket_ids if b in self._pending]
        while remaining:
            still_pending = []
            for bucket_id in remaining:
                handle, comm_buffer = self._pending[bucket_id]
                if handle.is_completed():
                    handle.wait()  # Orders the current stream after the collective.
                    del self._pending[bucket_id]
                    get_buffer_fn(bucket_id).copy_(comm_buffer)
                    self._fresh.add(bucket_id)
                else:
                    still_pending.append(bucket_id)
            remaining = still_pending
            if remaining and time.monotonic() >= end:
                break

    def replicate_bucket_group(
        self,
        bucket_group: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> Optional[torch.cuda.Event]:
        if self._world_size <= 1:
            return None

        event = None
        with torch.cuda.stream(stream):
            for bucket_id in bucket_group:
                grad = get_buffer_fn(bucket_id)

                # Clone the CURRENT gradient before overwriting the buffer
                # with the previous step's result.
                comm_buffer = grad.clone()

                if self.synchronous:
                    dist.all_reduce(
                        comm_buffer, op=dist.ReduceOp.AVG, group=self._process_group
                    )
                    grad.copy_(comm_buffer)
                    continue

                # Resolve previous step's async all_reduce.
                had_pending = bucket_id in self._pending
                if had_pending:
                    handle, prev_buffer = self._pending.pop(bucket_id)
                    handle.wait()
                    # Copy the averaged result into the gradient buffer
                    # for the optimizer to use.
                    grad.copy_(prev_buffer)

                # Post new async all_reduce on the current gradient.
                handle = dist.all_reduce(
                    comm_buffer,
                    op=dist.ReduceOp.AVG,
                    group=self._process_group,
                    async_op=True,
                )
                self._pending[bucket_id] = (handle, comm_buffer)

                if not had_pending:
                    if bucket_id in self._fresh:
                        # Last step's result was applied in that same step, so
                        # there is nothing pending. Keep the local (DP-Shard
                        # mean) gradient as the estimate; finalize_step
                        # replaces it with the global result if it arrives in
                        # time.
                        self._fresh.discard(bucket_id)
                    else:
                        # First step: no result available, zero-fill.
                        grad.zero_()

            event = stream.record_event()

        return event
