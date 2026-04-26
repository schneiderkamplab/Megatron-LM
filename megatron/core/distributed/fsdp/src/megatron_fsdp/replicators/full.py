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

    def wait_pending(self, bucket_id: int, -> Tuple[bool, Optional[torch.Tensor]]:
        """Resolve pending async all_reduce from the previous step.


        Returns:
            Tuple of (True, result_buffer) if resolved, else (False, None).
            result_buffer can be copied into the gradient buffer.
        """
        if bucket_id not in self._pending:
            return False, (None, None)

        handle, result_buffer = self._pending.pop(bucket_id)
        handle.wait()
        return True, result_buffer

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
                # Resolve previous step's async result
                had_pending = self.wait_pending(bucket_id)

                grad = get_buffer_fn(bucket_id)

                if had_pending:
                    # Copy the resolved result back into the gradient buffer
                    _, result = list(self._pending.items())[0][1] if bucket_id in self._pending else (None, None)
                    # result was already the grad buffer (in-place all_reduce)
                    pass

                # Post new async all_reduce (in-place on a copy so we don't
                # corrupt the current gradient buffer that the optimizer may read)
                comm_buffer = grad.clone()
                handle = dist.all_reduce(
                    comm_buffer,
                    op=dist.ReduceOp.AVG,
                    group=self._process_group,
                    async_op=True,
                )
                self._pending[bucket_id] = (handle, comm_buffer)

            event = stream.record_event()

        return event
