# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist

from .base import BucketReplicator

__all__ = ["NoOpBucketReplicator"]


class NoOpBucketReplicator(BucketReplicator):
    """Passthrough replicator that does nothing.

    Used when replication_strategy='none'. The standard Megatron-LM
    DP-Outer communication runs unchanged when this replicator is active.
    All methods are no-ops.
    """

    def init(
        self,
        process_group: dist.ProcessGroup,
        bucket_sizes: Dict[int, int],
        dtype: torch.dtype,
        device: torch.device,
        lr_provider: Callable[[], float],
    ) -> None:
        pass

    def pre_step(self) -> None:
        pass

    def post_step(self) -> None:
        pass

    def replicate_bucket_group(
        self,
        bucket_group: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> Optional[torch.cuda.Event]:
        # Return None so the caller uses its own standard DP-Outer logic.
        return None

    def wait_pending(self, bucket_id: int) -> bool:
        return False
