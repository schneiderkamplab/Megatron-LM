"""YAML experiment config parser for DeToNATION benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


VALID_STRATEGIES = ("none", "full", "demo", "slicing", "striding", "random")


@dataclass
class TrainConfig:
    train_iters: int = 100
    global_batch_size: int = 32
    micro_batch_size: int = 1


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    tp: int = 1
    pp: int = 1
    seq_length: int = 2048


@dataclass
class ReplicatorConfig:
    strategy: str
    topk: int = 32
    chunk: int = 64
    decay: float = 0.999
    rate: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown replication strategy: '{self.strategy}'. "
                f"Valid values: {VALID_STRATEGIES}"
            )


@dataclass
class HardwareConfig:
    nodes: int = 1
    gpus_per_node: int = 4


@dataclass
class ExperimentConfig:
    name: str
    train: TrainConfig
    models: list[ModelConfig]
    replicators: list[ReplicatorConfig]
    hardware: list[HardwareConfig]


def parse_config(path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment config file into an ExperimentConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    train = TrainConfig(**raw.get("train", {}))

    models = [ModelConfig(**m) for m in raw["models"]]
    replicators = [ReplicatorConfig(**r) for r in raw["replicators"]]
    hardware = [HardwareConfig(**h) for h in raw.get("hardware", [{"nodes": 1, "gpus_per_node": 4}])]

    return ExperimentConfig(
        name=raw["name"],
        train=train,
        models=models,
        replicators=replicators,
        hardware=hardware,
    )
