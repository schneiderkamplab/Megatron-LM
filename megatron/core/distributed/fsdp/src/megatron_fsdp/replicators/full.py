# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .base import BucketReplicator

__all__ = ["FullBucketReplicator"]


class FullBucketReplicator(BucketReplicator):
    """Full gradient replication via async all-reduce.

    Sends the complete sharded gradient buffer across the inter-node group
    using an asynchronous all_reduce. Results from the previous step are
    resolved before posting new communications (one-step-delay pattern);
    with finalize_step, a result that completes within its own step is
    applied in that step instead. Either way each result is applied once.

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
        self._stats = {"fresh": 0, "late": 0, "dropped": 0}

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

    def pop_stats(self) -> Dict[str, int]:
        stats, self._stats = self._stats, {"fresh": 0, "late": 0, "dropped": 0}
        return stats

    def finalize_step(
        self,
        bucket_ids: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        deadline_s: float,
    ) -> None:
        """Add this step's all-reduce results to the gradient if they complete
        in time, making the replicator synchronous whenever the network keeps
        up. A result that misses the deadline is added at the next step
        instead, so every result is applied exactly once."""
        if self.synchronous or self._world_size <= 1:
            return
        # Cutoff: when the step's gradients exist on the GPU, which the CPU
        # usually reaches early. The replication collectives run on their own
        # streams, so this does not wait for them.
        torch.cuda.current_stream().synchronize()
        end = time.monotonic() + deadline_s
        remaining = [b for b in bucket_ids if b in self._pending]
        while remaining:
            still_pending = []
            for bucket_id in remaining:
                handle, comm_buffer = self._pending[bucket_id]
                if handle.is_completed():
                    handle.wait()  # Orders the current stream after the collective.
                    del self._pending[bucket_id]
                    get_buffer_fn(bucket_id).add_(comm_buffer)
                    self._stats["fresh"] += 1
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

                # Clone the CURRENT gradient before the buffer is reused for
                # the results that arrive.
                comm_buffer = grad.clone()

                if self.synchronous:
                    dist.all_reduce(
                        comm_buffer, op=dist.ReduceOp.AVG, group=self._process_group
                    )
                    grad.copy_(comm_buffer)
                    continue

                # The gradient buffer becomes the sum of the results that
                # arrive: a late one from the previous step now, and this
                # step's own if finalize_step sees it complete in time.
                grad.zero_()
                if bucket_id in self._pending:
                    handle, prev_buffer = self._pending.pop(bucket_id)
                    handle.wait()
                    grad.add_(prev_buffer)
                    self._stats["late"] += 1

                handle = dist.all_reduce(
                    comm_buffer,
                    op=dist.ReduceOp.AVG,
                    group=self._process_group,
                    async_op=True,
                )
                self._pending[bucket_id] = (handle, comm_buffer)

            event = stream.record_event()

        return event
