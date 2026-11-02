# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist

from .base import BucketReplicator

__all__ = ["FullSyncBucketReplicator"]


class FullSyncBucketReplicator(BucketReplicator):
    """Full gradient replication via a synchronous all-reduce (diagnostic).

    Same as FullBucketReplicator but without the one-step delay: the averaged
    gradient is used in the step it was computed. Training through this
    replicator should match the native DP-Outer all-reduce, which isolates the
    cost of the delay from the cost of the replicator integration.
    """

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

    def wait_pending(self, bucket_id: int) -> bool:
        return False

    def replicate_bucket_group(
        self,
        bucket_group: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> Optional[torch.cuda.Event]:
        if self._world_size <= 1:
            return None
        with torch.cuda.stream(stream):
            for bucket_id in bucket_group:
                dist.all_reduce(
                    get_buffer_fn(bucket_id), op=dist.ReduceOp.AVG, group=self._process_group
                )
            return stream.record_event()
