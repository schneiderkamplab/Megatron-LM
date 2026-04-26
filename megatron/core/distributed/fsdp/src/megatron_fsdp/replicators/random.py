# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .base import BucketReplicator, DeltaBufferManager

__all__ = ["RandomBucketReplicator"]


class RandomBucketReplicator(BucketReplicator):
    """Random permutation sampling replication.

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

        # Random generator for per-step permutations
        self._rng = torch.Generator(device="cpu").manual_seed(self.seed)

        # Pending async: bucket_id -> (handle, comm_buffer, indices, numel)
        self._pending: Dict[
            int, Tuple[dist.Work, torch.Tensor, torch.Tensor, int]
        ] = {}

        # Pre-generate permutations for each unique bucket size
        self._permutations: Dict[int, torch.Tensor] = {}

    def pre_step(self) -> None:
        """Generate new random permutations for this step."""
        max_size = max(self._bucket_sizes.values()) if self._bucket_sizes else 0
        if max_size == 0:
            return

        rand_score = torch.rand(max_size, generator=self._rng)
        self._permutations = {}
        for bucket_id, numel in self._bucket_sizes.items():
            if numel == 0:
                continue
            k = max(int(self.compression_rate * numel), 1)
            k = min(k, numel)
            self._permutations[bucket_id] = torch.topk(
                rand_score[:numel], k=k, largest=False
            ).indices.to(device=self._device)

    def post_step(self) -> None:
        pass

    def wait_pending(self, bucket_id: int) -> bool:
        if bucket_id not in self._pending:
            return False

        handle, comm_buffer, indices, numel = self._pending.pop(bucket_id)
        handle.wait()

        # Scatter result back into the gradient buffer
        grad = self._delta_manager.deltas[bucket_id]  # temporary target
        result = torch.zeros(numel, dtype=self._dtype, device=self._device)
        result[indices] = comm_buffer

        # Copy into the actual gradient buffer (stored as the result target)
        # This is handled by replicate_bucket_group which has access to get_buffer_fn
        self._last_result = (bucket_id, result)
        return True

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

                # Resolve pending result from previous step
                had_pending = bucket_id in self._pending
                if had_pending:
                    handle, comm_buffer, indices, numel = self._pending.pop(bucket_id)
                    handle.wait()
                    # Scatter result into gradient buffer
                    result = torch.zeros(numel, dtype=self._dtype, device=self._device)
                    result[indices] = comm_buffer
                    grad.copy_(result)

                # Update delta
                delta = self._delta_manager.update_delta(
                    bucket_id, grad, lr, self.compression_decay
                )

                if self._world_size <= 1 or self.compression_rate >= 1.0:
                    grad.copy_(delta)
                    delta.zero_()
                    continue

                if bucket_id not in self._permutations:
                    continue

                # Random sample from delta
                indices = self._permutations[bucket_id]
                compressed = delta[indices].clone()

                # Zero out sampled elements in delta
                mask = torch.zeros(delta.numel(), dtype=torch.bool, device=self._device)
                mask[indices] = True
                delta[~mask] = 0

                # Async all_reduce
                handle = dist.all_reduce(
                    compressed,
                    op=dist.ReduceOp.AVG,
                    group=self._process_group,
                    async_op=True,
                )
                self._pending[bucket_id] = (handle, compressed, indices, delta.numel())

                if not had_pending:
                    # First step: no result, zero-fill
                    grad.zero_()

            event = stream.record_event()

        return event
