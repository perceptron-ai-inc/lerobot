"""P2.12/P2.17: reject unsupported intent in metadata preflight, not at first save."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, NoReturn

import pytest

pytest.importorskip("transformers", reason="requires the parent-owned ISAAC CPU environment")

from lerobot.configs.default import DatasetConfig, EMAConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _source_package
from tests.policies.perceptron_isaac.test_portable_model_repo import _native_portable_package

Intent = Literal["periodic_save", "final_save_only", "publish_final", "publish_checkpoint", "ema"]
INTENTS: tuple[Intent, ...] = (
    "periodic_save",
    "final_save_only",
    "publish_final",
    "publish_checkpoint",
    "ema",
)


def _train_config(
    config: PerceptronIsaacConfig | ACTConfig, output: Path, intent: Intent | None
) -> TrainPipelineConfig:
    config.push_to_hub = intent == "publish_final"
    config.repo_id = "fixture/never-uploaded" if intent in {"publish_final", "publish_checkpoint"} else None
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/never-downloaded", video_backend="pyav"),
        policy=config,
        output_dir=output,
        steps=2,
        save_checkpoint=intent in {"periodic_save", "final_save_only", "publish_checkpoint"},
        save_freq=0 if intent == "final_save_only" else 1,
        save_checkpoint_to_hub=intent == "publish_checkpoint",
        ema=EMAConfig(enable=intent == "ema"),
        env_eval_freq=0,
        eval_steps=0,
    )
    cfg.parallelism.resolve(1)
    return cfg


def _forbid_allocation_or_training(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("unsupported intent reached allocation/training before metadata refusal")


@pytest.mark.parametrize("intent", INTENTS)
def test_raw_export_training_intent_is_refused_by_real_config_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: Intent
) -> None:
    policy_path, config, _ = _native_portable_package(tmp_path)
    config.pretrained_path = policy_path
    cfg = _train_config(config, tmp_path / "run", intent)
    monkeypatch.setattr(sys, "argv", ["lerobot-train"])
    monkeypatch.setattr(PerceptronIsaacPolicy, "__init__", _forbid_allocation_or_training)

    with pytest.raises(
        ValueError, match=r"(?i)(raw|portable).*(unsupported|not supported)|unsupported.*(raw|portable)"
    ):
        cfg.validate()

    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize("intent", INTENTS)
def test_standard_train_refuses_raw_intent_before_engine_model_or_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: Intent
) -> None:
    pytest.importorskip("datasets", reason="the maintained training entry point requires lerobot[dataset]")
    import lerobot.scripts.lerobot_train as train_module

    policy_path, config, _ = _native_portable_package(tmp_path)
    config.pretrained_path = policy_path
    cfg = _train_config(config, tmp_path / "run", intent)
    monkeypatch.setattr(sys, "argv", ["lerobot-train"])
    # Tripwires only: real config validation executes. No fake loader or save acceptance.
    monkeypatch.setattr(train_module, "make_accelerator", _forbid_allocation_or_training)
    monkeypatch.setattr(train_module, "make_policy", _forbid_allocation_or_training)
    monkeypatch.setattr(train_module, "update_policy", _forbid_allocation_or_training)
    monkeypatch.setattr(PerceptronIsaacPolicy, "__init__", _forbid_allocation_or_training)

    with pytest.raises(
        ValueError, match=r"(?i)(raw|portable).*(unsupported|not supported)|unsupported.*(raw|portable)"
    ):
        train_module.train(cfg)

    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()


def test_raw_explicit_no_save_training_preflight_and_inference_remain_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path, config, _ = _native_portable_package(tmp_path)
    config.pretrained_path = policy_path
    cfg = _train_config(config, tmp_path / "run", None)
    monkeypatch.setattr(sys, "argv", ["lerobot-train"])

    cfg.validate()
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)

    assert policy._isaac_model is not None
    assert not cfg.save_checkpoint and not config.push_to_hub and not cfg.ema.enable
    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize(
    "intent", ["periodic_save", "final_save_only", "publish_final", "publish_checkpoint"]
)
def test_source_bound_native_save_intents_remain_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: Intent
) -> None:
    source = tmp_path / "source"
    config = _source_package(source)
    config.pretrained_path = source
    cfg = _train_config(config, tmp_path / "run", intent)
    monkeypatch.setattr(sys, "argv", ["lerobot-train"])
    monkeypatch.setattr(PerceptronIsaacPolicy, "__init__", _forbid_allocation_or_training)

    cfg.validate()

    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize("intent", INTENTS)
def test_ordinary_policy_training_intents_remain_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: Intent
) -> None:
    cfg = _train_config(ACTConfig(device="cpu"), tmp_path / "run", intent)
    monkeypatch.setattr(sys, "argv", ["lerobot-train"])

    cfg.validate()

    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize("policy_type", ["isaac", "act"])
def test_normal_hub_defaults_and_missing_repo_validation_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy_type: str
) -> None:
    config = PerceptronIsaacConfig(device="cpu") if policy_type == "isaac" else ACTConfig(device="cpu")
    assert config.push_to_hub is True
    assert config.repo_id is None
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/unused", video_backend="pyav"),
        policy=config,
        output_dir=tmp_path / "run",
    )
    cfg.parallelism.resolve(1)
    monkeypatch.setattr(sys, "argv", ["lerobot-train"])

    with pytest.raises(ValueError, match="repo_id.*missing"):
        cfg.validate()

    assert config.push_to_hub is True
    assert config.repo_id is None
