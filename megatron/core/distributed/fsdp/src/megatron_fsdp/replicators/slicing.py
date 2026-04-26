# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist

from .base import BucketReplicator, DeltaBufferManager
from .compression_utils import _get_smaller_split

__all__ = ["SlicingBucketReplicator"]


class SlicingBucketReplicator(BucketReplicator):
    """Cyclic chunk-slicing replication.

    Divides each bucket's delta buffer into chunks and cyclically slices
    a fraction of those chunks per step. The selected slices are all_reduced
    (synchronous) across nodes. This provides bandwidth savings proportional
    to (1 - compression_rate).
    """

    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_rate: float = 0.1,
        compression_chunk: int = 64,
    ):
        if compression_rate <= 0:
            raise ValueError("compression_rate must be positive")
        if compression_rate > 1:
            raise ValueError("compression_rate > 1.0 is not supported")

        self.compression_decay = compression_decay
        self.compression_rate = compression_rate
        self.compression_chunk = compression_chunk

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

        self._delta_manager = DeltaBufferManager(bucket_sizes, dtype, device)
        self._counter = 0

        # Pre-compute chunk sizes for each bucket
        self._chunk_sizes: Dict[int, int] = {}
        for bucket_id, numel in bucket_sizes.items():
            self._chunk_sizes[bucket_id] = _get_smaller_split(
                numel, self.compression_chunk
            )

    def pre_step(self) -> None:
        pass

    def post_step(self) -> None:
        pass

    def wait_pending(self, bucket_id: int) -> bool:
        # Synchronous — nothing to wait on
        return False

    def replicate_bucket_group(
        self,
        bucket_group: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> Optional[torch.cuda.Event]:
        event = None
        with torch.cuda.stream(stream):
            for bucket_id in bucket_group:
                grad = get_buffer_fn(bucket_id)
                lr = self._lr_provider()

                # Update delta
                delta = self._delta_manager.update_delta(
                    bucket_id, grad, lr, self.compression_decay
                )

                if self._world_size <= 1 or self.compression_rate >= 1.0:
                    grad.copy_(delta)
                    delta.zero_()
                    continue

                chunk_size = self._chunk_sizes[bucket_id]
                if chunk_size == 0 or delta.numel() == 0:
                    continue

                numel = delta.numel()
                assert numel % chunk_size == 0
                num_chunks = numel // chunk_size
                delta_2d = delta.view(num_chunks, -1)

                # Select cyclic slice of chunks
                chunks_per_step = max(int(self.compression_rate * num_chunks), 1)
                begin = (self._counter * chunks_per_step) % num_chunks
                end = begin + chunks_per_step

                if end <= num_chunks:
                    compressed = delta_2d[begin:end].clone()
                    mask = torch.zeros(num_chunks, dtype=torch.bool, device=self._device)
                    mask[begin:end] = True
                else:
                    wrap = end - num_chunks
                    compressed = torch.cat(
                        [delta_2d[begin:], delta_2d[:wrap]], dim=0
                    ).clone()
                    mask = torch.zeros(num_chunks, dtype=torch.bool, device=self._device)
                    mask[begin:] = True
                    mask[:wrap] = True

                # Zero out transmitted chunks from delta
                delta_2d[mask] = 0

                # Synchronous all_reduce of compressed data
                dist.all_reduce(
                    compressed, op=dist.ReduceOp.AVG, group=self._process_group
                )

                # Scatter result back into gradient buffer
                result = torch.zeros_like(delta_2d)
                if end <= num_chunks:
                    result[begin:end] = compressed
                else:
                    split = num_chunks - begin
                    result[begin:] = compressed[:split]
                    result[:wrap] = compressed[split:]

                grad.copy_(result.view(-1))

            self._counter += 1
            event = stream.record_event()

        return event
