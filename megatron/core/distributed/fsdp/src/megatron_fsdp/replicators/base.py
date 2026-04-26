# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist


class DeltaBufferManager:
    """Manages per-bucket delta tensors for compression-based replicators.

    Delta buffers match the sharded gradient buffer layout: one tensor per bucket,
    same dtype and device as the corresponding gradient buffer.
    """

    def __init__(
        self,
        bucket_sizes: Dict[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Args:
            bucket_sizes: Mapping from bucket_id to the number of elements in the
                sharded gradient buffer for that bucket.
            dtype: Data type for delta buffers.
            device: Device for delta buffers.
        """
        self.deltas: Dict[int, torch.Tensor] = {}
        for bucket_id, numel in bucket_sizes.items():
            self.deltas[bucket_id] = torch.zeros(numel, dtype=dtype, device=device)

    def update_delta(
        self,
        bucket_id: int,
        grad: torch.Tensor,
        lr: float,
        decay: float,
    ) -> torch.Tensor:
        """Update and return the delta for a bucket.

        Computes: delta = decay * delta + lr * grad

        Args:
            bucket_id: Bucket identifier.
            grad: Sharded gradient tensor (must be 1-D contiguous).
            lr: Current learning rate.
            decay: Delta decay factor.

        Returns:
            The updated delta tensor.
        """
        delta = self.deltas[bucket_id]
        if decay != 1.0:
            delta.mul_(decay)
        delta.add_(grad, alpha=lr)
        return delta

    def get_delta(self, bucket_id: int) -> torch.Tensor:
        """Return the delta tensor for a bucket."""
        return self.deltas[bucket_id]

    def zero_delta(self, bucket_id: int) -> None:
        """Zero out the delta for a bucket."""
        self.deltas[bucket_id].zero_()


class BucketReplicator(ABC):
    """Abstract base class for bucket-level gradient replicators.

    A BucketReplicator replaces the standard DP-Outer (inter-node) communication
    in Megatron-LM's GradRed
    ucePipeline with a custom strategy that may include
    compression, delta accumulation, and asynchronous communication.

    The replicator operates on bucket groups (lists of bucket IDs) and receives
    gradient data via a callable (get_buffer_fn) rather than directly holding
    references to gradient buffers. This decouples the replicator from the
    buffer management internals.
    """

    @abstractmethod
    def init(
        self,
        process_group: dist.ProcessGroup,
        bucket_sizes: Dict[int, int],
        dtype: torch.dtype,
        device: torch.device,
        lr_provider: Callable[[], float],
    ) -> None:
        """Initialize the replicator.

        Called once after construction, once the buffer layout is known.

        Args:
            process_group: The process group for inter-node communication
                (either HSDP outer group or standalone replication group).
            bucket_sizes: Mapping from bucket_id to number of elements in the
                sharded gradient buffer for that bucket.
            dtype: Data type for internal buffers (delta, communication).
            device: Device for internal buffers.
            lr_provider: Callable that returns the current learning rate.
        """
        pass

    def pre_step(self) -> None:
        """Called at the beginning of each training step, before gradient reduction."""
        pass

    def post_step(self) -> None:
        """Called after optimizer.step() completes."""
        pass

    @abstractmethod
    def replicate_bucket_group(
        self,
        bucket_group: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> Optional[torch.cuda.Event]:
        """Replicate gradients across the inter-node group for a bucket group.

        This is called from the DP-Outer phase of
        GradReducePipeline._bucket_group_gradient_reduce(). The replicator
        should handle inter-node gradient exchange, possibly with compression.

        For async replicators: resolve pending results from the previous step
        first, then post new async communications. Return a CUDA event that
        will be used to synchronize the pipeline. On the first step when no
        result is available yet, the gradient buffer should be zero-filled
        (the caller handles this).

        Args:
            bucket_group: List of bucket IDs being reduced together.
            get_buffer_fn: Callable that returns the sharded gradient tensor
                for a given bucket_id. The returned tensor is 1-D.
            stream: CUDA stream to use for communication operations.

        Returns:
            Optional CUDA event to wait on for completion, or None if the
            operation is synchronous or the caller should use its own event.
        """
        pass

    @abstractmethod
    def wait_pending(self, bucket_id: int) -> bool:
        """Wait for any pending async replication result for the given bucket.

        Called at the start of replicate_bucket_group() to resolve results
        from the previous step before posting new communications.

        Args:
            bucket_id: The bucket to check for pending results.

        Returns:
            True if a replicated gradient was applied to the buffer.
        """
        pass
