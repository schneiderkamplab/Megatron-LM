# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .base import BucketReplicator, DeltaBufferManager
from .compression_utils import DCTBufferCompress, DCTBufferTransform

__all__ = ["DeMoBucketReplicator"]


class DeMoBucketReplicator(BucketReplicator):
    """DeMo (Decoupled Momentum) replication with DCT compression.

    For each bucket, maintains a delta buffer:
        delta = decay * delta + lr * grad
    Then DCT-encodes the delta, selects top-k coefficients per chunk,
    and asynchronously all_gathers sparse indices/values across nodes.

    On the next step, the gathered sparse data from all nodes is decoded
    to produce the replicated gradient. This introduces a one-step delay
    (identical to the original DeToNation design).
    """

    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
    ):
        if compression_topk <= 0:
            raise ValueError("compression_topk must be positive")
        if compression_chunk <= 0:
            raise ValueError("compression_chunk must be positive")
        if compression_decay < 0:
            raise ValueError("Negative compression_decay is not supported")
        if compression_decay >= 1:
            raise ValueError("compression_decay >= 1.0 is not supported")

        self.compression_decay = compression_decay
        self.compression_topk = compression_topk
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
        self._dtype = dtype

        # Delta buffers
        self._delta_manager = DeltaBufferManager(bucket_sizes, dtype, device)

        # DCT transform (one set of basis matrices per unique chunk size)
        self._transform = DCTBufferTransform(
            buffer_sizes=bucket_sizes,
            compression_chunk=self.compression_chunk,
            dtype=dtype,
            device=device,
        )

        # Pending async operations: bucket_id -> (idx_handle, val_handle,
        # idx_gather_bufs, val_gather_bufs, result_buffer, bucket_id_for_numel)
        self._pending: Dict[
            int, Tuple[dist.Work, dist.Work, list, list, torch.Tensor, int]
        ] = {}

    def pre_step(self) -> None:
        pass

    def post_step(self) -> None:
        pass

    def wait_pending(self, bucket_id: int) -> bool:
        if bucket_id not in self._pending:
            return False

        idx_handle, val_handle, idx_bufs, val_bufs, result_buf, numel = self._pending.pop(
            bucket_id
        )
        idx_handle.wait()
        val_handle.wait()

        # Batch decompress: average all nodes' sparse data
        idx_long = [b.to(torch.int64) for b in idx_bufs]
        averaged = DCTBufferCompress.batch_decompress(
            idx_long, val_bufs, numel, self._device, self._dtype
        )

        # Inverse DCT to get the replicated gradient
        result_buf.copy_(
            self._transform.decode(bucket_id, averaged)
        )
        return True

    def replicate_bucket_group(
        self,
        bucket_group: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> Optional[torch.cuda.Event]:
        if self._world_size <= 1:
            # Single node: just use the delta directly
            event = None
            with torch.cuda.stream(stream):
                for bucket_id in bucket_group:
                    grad = get_buffer_fn(bucket_id)
                    delta = self._delta_manager.update_delta(
                        bucket_id, grad, self._lr_provider(), self.compression_decay
                    )
                    grad.copy_(delta)
                    delta.zero_()
            return event

        event = None
        with torch.cuda.stream(stream):
            for bucket_id in bucket_group:
                grad = get_buffer_fn(bucket_id)
                lr = self._lr_provider()

                # Update delta
                delta = self._delta_manager.update_delta(
                    bucket_id, grad, lr, self.compression_decay
                )

                # Resolve previous step's result
                had_pending = bucket_id in self._pending
                if had_pending:
                    idx_handle, val_handle, idx_bufs, val_bufs, result_buf, numel = self._pending.pop(
                        bucket_id
                    )
                    idx_handle.wait()
                    val_handle.wait()

                    # Batch decompress
                    idx_long = [b.to(torch.int64) for b in idx_bufs]
                    averaged = DCTBufferCompress.batch_decompress(
                        idx_long, val_bufs, numel, self._device, self._dtype
                    )
                    # Write decoded result into the gradient buffer
                    grad.copy_(
                        self._transform.decode(bucket_id, averaged)
                    )

                # DCT encode the delta
                encoded = self._transform.encode(bucket_id, delta)

                # Top-k compress (per-chunk)
                sparse_idx, sparse_val = DCTBufferCompress.compress(
                    encoded, self.compression_topk,
                    chunk_size=self._transform.chunk_sizes[bucket_id],
                )
                sparse_idx = sparse_idx.to(torch.int32)

                # Estimate transmitted gradient and subtract from delta
                decompressed = DCTBufferCompress.decompress(
                    sparse_idx, sparse_val, encoded.numel(),
                    self._device, self._dtype,
                )
                transmit_grad = self._transform.decode(bucket_id, decompressed)
                delta.sub_(transmit_grad)

                # Post async all_gather for indices and values
                numel = encoded.numel()
                idx_gather_buf = [
                    torch.zeros_like(sparse_idx) for _ in range(self._world_size)
                ]
                val_gather_buf = [
                    torch.zeros_like(sparse_val) for _ in range(self._world_size)
                ]
                idx_handle = dist.all_gather(
                    idx_gather_buf, sparse_idx,
                    group=self._process_group, async_op=True,
                )
                val_handle = dist.all_gather(
                    val_gather_buf, sparse_val,
                    group=self._process_group, async_op=True,
                )

                # Store result buffer for the NEXT step's wait_pending
                # For the first step (no pending result), we'll zero-fill
                self._pending[bucket_id] = (
                    idx_handle, val_handle,
                    idx_gather_buf, val_gather_buf,
                    grad,  # reuse grad buffer for result
                    numel,
                )

                if not had_pending:
                    # First step: no result available, zero-fill
                    grad.zero_()

            event = stream.record_event()

        return event
