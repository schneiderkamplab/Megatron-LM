# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from .base import BucketReplicator, DeltaBufferManager
from .no_op import NoOpBucketReplicator

__all__ = [
    "BucketReplicator",
    "DeltaBufferManager",
    "NoOpBucketReplicator",
    "FullBucketReplicator",
    "FullSyncBucketReplicator",
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
    max_delay: int = 4,
) -> BucketReplicator:
    """Factory function to create a BucketReplicator by strategy name.

    Args:
        strategy: Replication strategy name. Valid values:
            'none' - No replication (passthrough).
            'full' - Full gradient all-reduce.
            'full_sync' - Full gradient all-reduce without the one-step delay (diagnostic).
            'demo' - DeMo DCT compression.
            'slicing' - Cyclic chunk slicing.
            'striding' - Strided sampling.
            'random' - Random permutation sampling.
        compression_decay: Delta decay factor (default 0.999).
        compression_topk: Top-k for DeMo (default 32).
        compression_chunk: Chunk size for DeMo/slicing/striding (default 64).
        compression_rate: Fraction for slicing/striding/random (default 0.1).
        seed: Random seed for random replicator (default 42).
        max_delay: Steps an async random-replicator result may lag before it is
            dropped (default 4). 0 makes the full, random and demo replicators synchronous.

    Returns:
        A BucketReplicator instance.
    """
    if strategy == "none":
        return NoOpBucketReplicator()
    elif strategy == "full":
        from .full import FullBucketReplicator

        return FullBucketReplicator(synchronous=max_delay == 0)
    elif strategy == "full_sync":
        from .full_sync import FullSyncBucketReplicator

        return FullSyncBucketReplicator()
    elif strategy == "demo":
        from .demo import DeMoBucketReplicator

        return DeMoBucketReplicator(
            compression_decay=compression_decay,
            compression_topk=compression_topk,
            compression_chunk=compression_chunk,
            synchronous=max_delay == 0,
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
            max_delay=max_delay,
        )
    else:
        raise ValueError(
            f"Unknown replication strategy: '{strategy}'. "
            f"Valid values: 'none', 'full', 'full_sync', 'demo', 'slicing', 'striding', 'random'."
        )
