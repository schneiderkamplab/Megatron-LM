# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from .base import BucketReplicator, DeltaBufferManager
from .no_op import NoOpBucketReplicator

__all__ = [
    "BucketReplicator",
    "DeltaBufferManager",
    "NoOpBucketReplicator",
    "FullBucketReplicator",
    "DeMoBucketReplicator",
    "SlicingBucketReplicator",
    "StridingBucketReplicator",
    "RandomBucketReplicator",
    "get_replicator",
]


def get_replicator(
    strategy: str = "none",
    *,
    compression_decay: float = 0.999,
    compression_topk: int = 32,
    compression_chunk: int = 64,
    compression_rate: float = 0.1,
    seed: int = 42,
) -> BucketReplicator:
    """Factory function to create a BucketReplicator by strategy name.

    Args:
        strategy: Replication strategy name. Valid values:
            'none' - No replication (passthrough).
            'full' - Full gradient all-reduce.
            'demo' - DeMo DCT compression.
            'slicing' - Cyclic chunk slicing.
            'striding' - Strided sampling.
            'random' - Random permutation sampling.
        compression_decay: Delta decay factor (default 0.999).
        compression_topk: Top-k for DeMo (default 32).
        compression_chunk: Chunk size for DeMo/slicing/striding (default 64).
        compression_rate: Fraction for slicing/striding/random (default 0.1).
        seed: Random seed for random replicator (default 42).

    Returns:
        A BucketReplicator instance.
    """
    if strategy == "none":
        return NoOpBucketReplicator()
    elif strategy == "full":
        from .full import FullBucketReplicator

        return FullBucketReplicator()
    elif strategy == "demo":
        from .demo import DeMoBucketReplicator

        return DeMoBucketReplicator(
            compression_decay=compression_decay,
            compression_topk=compression_topk,
            compression_chunk=compression_chunk,
        )
    elif strategy == "slicing":
        from .slicing import SlicingBucketReplicator

        return SlicingBucketReplicator(
            compression_decay=compression_decay,
            compression_rate=compression_rate,
            compression_chunk=compression_chunk,
        )
    elif strategy == "striding":
        from .striding import StridingBucketReplicator

        return StridingBucketReplicator(
            compression_decay=compression_decay,
            compression_rate=compression_rate,
            compression_chunk=compression_chunk,
        )
    elif strategy == "random":
        from .random import RandomBucketReplicator

        return RandomBucketReplicator(
            compression_decay=compression_decay,
            compression_rate=compression_rate,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown replication strategy: '{strategy}'. "
            f"Valid values: 'none', 'full', 'demo', 'slicing', 'striding', 'random'."
        )
