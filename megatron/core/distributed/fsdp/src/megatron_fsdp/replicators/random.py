# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .base import BucketReplicator, DeltaBufferManager

__all__ = ["RandomBucketReplicator"]


class RandomBucketReplicator(BucketReplicator):
    """Random sampling replication.

    For each bucket, randomly selects a fraction of elements from the delta
    buffer and asynchronously all_reduces them across nodes. On the next step,
    the result is scattered back. Introduces a one-step delay.
    """

    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_rate: float = 0.1,
        seed: int = 42,
    ):
        if compression_rate <= 0:
            raise ValueError("compression_rate must be positive")
        if compression_rate > 1:
            raise ValueError("compression_rate > 1.0 is not supported")

        self.compression_decay = compression_decay
        self.compression_rate = compression_rate
        self.seed = seed

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
        self._lr_provider = lr_provider
        self._device = device
        self._dtype = dtype

        self._delta_manager = DeltaBufferManager(bucket_sizes, dtype, device)
        self._bucket_sizes = bucket_sizes

        self._rng = torch.Generator(device=device).manual_seed(self.seed)

        # Pre-allocate reusable buffers per bucket.
        self._result_bufs: Dict[int, torch.Tensor] = {}   # full-size, zeroed each step
        self._mask_bufs: Dict[int, torch.Tensor] = {}    # bool, refilled each step
        for bucket_id, numel in bucket_sizes.items():
            if numel == 0:
                continue
            self._result_bufs[bucket_id] = torch.zeros(numel, dtype=dtype, device=device)
            self._mask_bufs[bucket_id] = torch.zeros(numel, dtype=torch.bool, device=device)

        # Pending: bucket_id -> (handle, comm_buffer, mask)
        # mask is stored as a clone (small bool) since _mask_bufs is reused.
        self._pending: Dict[
            int, Tuple[dist.Work, torch.Tensor, torch.Tensor]
        ] = {}

    def pre_step(self) -> None:
        pass

    def post_step(self) -> None:
        pass

    def wait_pending(self, bucket_id: int) -> bool:
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
                lr = self._lr_provider()

                delta = self._delta_manager.update_delta(
                    bucket_id, grad, lr, self.compression_decay
                )

                had_pending = bucket_id in self._pending
                if had_pending:
                    handle, comm_buf, prev_mask = self._pending.pop(bucket_id)
                    handle.wait()
                    result = self._result_bufs[bucket_id]
                    result.zero_()
                    result[prev_mask] = comm_buf
                    grad.copy_(result)

                if self.compression_rate >= 1.0:
                    grad.copy_(delta)
                    delta.zero_()
                    continue

                # Generate random mask: torch.rand < rate is O(n), no topk.
                mask = self._mask_bufs[bucket_id]
                mask.copy_(
                    torch.rand(
                        delta.numel(), generator=self._rng, device=self._device
                    ) < self.compression_rate
                )

                # Gather selected elements, zero them from delta.
                compressed = delta[mask].clone()
                delta[mask] = 0

                handle = dist.all_reduce(
                    compressed,
                    op=dist.ReduceOp.AVG,
                    group=self._process_group,
                    async_op=True,
                )
                self._pending[bucket_id] = (handle, compressed, mask.clone())

                if not had_pending:
                    grad.zero_()

            event = stream.record_event()

        return event
