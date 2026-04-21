"""Tests for the experiment config parser."""
import tempfile
import os
import pytest
import yaml

from config_parser import ExperimentConfig, ModelConfig, ReplicatorConfig, HardwareConfig, parse_config


def test_parse_minimal_config():
    """Parser should load a minimal YAML config with all required fields."""
    config_yaml = {
        "name": "test_run",
        "train": {"train_iters": 10, "global_batch_size": 4, "micro_batch_size": 1},
        "models": [
            {"name": "gpt2", "hf_id": "openai-community/gpt2", "tp": 1, "pp": 1, "seq_length": 1024}
        ],
        "replicators": [
            {"strategy": "none"},
            {"strategy": "demo", "topk": 32, "chunk": 64, "decay": 0.999},
        ],
        "hardware": [
            {"nodes": 1, "gpus_per_node": 2}
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_yaml, f)
        f.flush()
        cfg = parse_config(f.name)
    os.unlink(f.name)

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "test_run"
    assert cfg.train.train_iters == 10
    assert len(cfg.models) == 1
    assert cfg.models[0].hf_id == "openai-community/gpt2"
    assert len(cfg.replicators) == 2
    assert cfg.replicators[1].strategy == "demo"
    assert cfg.replicators[1].topk == 32
    assert len(cfg.hardware) == 1
    assert cfg.hardware[0].gpus_per_node == 2


def test_replicator_defaults():
    """ReplicatorConfig should have sensible defaults for all optional fields."""
    r = ReplicatorConfig(strategy="demo")
    assert r.topk == 32
    assert r.chunk == 64
    assert r.decay == 0.999
    assert r.rate == 0.1
    assert r.seed == 42


def test_invalid_strategy_raises():
    """Parser should reject unknown replicator strategies."""
    config_yaml = {
        "name": "bad",
        "train": {"train_iters": 10, "global_batch_size": 4, "micro_batch_size": 1},
        "models": [{"name": "x", "hf_id": "x", "tp": 1, "pp": 1, "seq_length": 128}],
        "replicators": [{"strategy": "invalid_strategy"}],
        "hardware": [{"nodes": 1, "gpus_per_node": 1}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_yaml, f)
        f.flush()
    with pytest.raises(ValueError, match="Unknown replication strategy"):
        parse_config(f.name)
    os.unlink(f.name)


def test_model_config_required_fields():
    """ModelConfig should require hf_id, tp, pp, seq_length."""
    with pytest.raises(TypeError):
        ModelConfig(name="x")
