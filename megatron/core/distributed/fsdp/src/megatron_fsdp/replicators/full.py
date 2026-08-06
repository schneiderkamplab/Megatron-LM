# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .base import BucketReplicator

__all__ = ["FullBucketReplicator"]


class FullBucketReplicator(BucketReplicator):
    """Full gradient replication via async all-reduce.

    Sends the complete sharded gradient buffer across the inter-node group
    using an asynchronous all_reduce. Results from the previous step are
    resolved before posting new communications (one-step-delay pattern).

    This replicator serves as the correctness baseline — it should produce
    numerically identical results to the standard Megatron-LM DP-Outer
    all_reduce.
    """

    def __init__(self):
        self._process_group: Optional[dist.ProcessGroup] = None
        self._world_size: int = 1
        self._pending: Dict[int, Tuple[dist.Work, torch.Tensor]] = {}

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
                    # First step: no result available, zero-fill.
                    grad.zero_()

            event = stream.record_event()

        return event
