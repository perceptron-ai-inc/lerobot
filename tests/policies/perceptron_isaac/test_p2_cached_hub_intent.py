"""Cached Hub IDs must reach the same metadata-only intent gate as local packages."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("transformers", reason="requires the parent-owned offline ISAAC CPU environment")

from huggingface_hub import constants, hf_hub_download, snapshot_download

from lerobot.configs import policies as policy_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from tests.policies.perceptron_isaac.test_p2_training_intent import (
    INTENTS,
    Intent,
    _forbid_allocation_or_training,
    _train_config,
)
from tests.policies.perceptron_isaac.test_portable_model_repo import _native_portable_package


@dataclass(frozen=True)
class CachedPolicy:
    repo_id: str
    snapshot: Path


def _cache_policy(
    root: Path, config: PerceptronIsaacConfig | ACTConfig, monkeypatch: pytest.MonkeyPatch
) -> CachedPolicy:
    """Use the real Hub cache readers; only their IO environment is redirected offline."""
    cache = root / "hub"
    repo = cache / "models--fixture--cached-policy"
    revision = "a" * 40
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    config.save_pretrained(snapshot)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))

    def offline_config_download(*, repo_id: str, filename: str, **kwargs: Any) -> str:
        # The config caller still supplies legacy no-op options removed by Hub 1.x.
        assert kwargs.pop("proxies", None) is None
        assert kwargs.pop("resume_download", None) is None
        kwargs["local_files_only"] = True
        return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)

    # CLI config loading has no local_files_only flag. Its IO leaf still calls the
    # maintained Hub API, with an explicit offline boundary even if a client was cached.
    monkeypatch.setattr(policy_configs, "hf_hub_download", offline_config_download)
    cached = CachedPolicy("fixture/cached-policy", snapshot)
    assert Path(hf_hub_download(cached.repo_id, "config.json", local_files_only=True)).parent == snapshot
    assert Path(snapshot_download(cached.repo_id, local_files_only=True)) == snapshot
    return cached


def _cache_raw_policy(root: Path, monkeypatch: pytest.MonkeyPatch, intent: Intent | None) -> CachedPolicy:
    source = root / "source"
    source.mkdir()
    policy_path, config, _ = _native_portable_package(source)
    # Materialize the other maintained raw layout: package-local relative hf_model.
    config.hf_model_path = "hf_model"
    _train_config(config, root / "unused", intent)
    cached = _cache_policy(root, config, monkeypatch)
    for asset in policy_path.iterdir():
        asset.rename(cached.snapshot / asset.name)
    policy_path.rmdir()
    (source / "model").rename(cached.snapshot / "hf_model")
    return cached


def _hub_cli(monkeypatch: pytest.MonkeyPatch, cached: CachedPolicy) -> None:
    monkeypatch.setattr(sys, "argv", ["lerobot-train", f"--policy.path={cached.repo_id}"])


@pytest.mark.parametrize("intent", INTENTS)
def test_cached_hub_raw_intent_refused_before_accelerator_policy_or_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: Intent
) -> None:
    pytest.importorskip("datasets", reason="requires the maintained stock training entry point")
    import lerobot.scripts.lerobot_train as train_module

    cached = _cache_raw_policy(tmp_path, monkeypatch, intent)
    before = {path: path.read_bytes() for path in cached.snapshot.rglob("*") if path.is_file()}
    cfg = _train_config(ACTConfig(device="cpu"), tmp_path / "run", intent)
    _hub_cli(monkeypatch, cached)
    # Real CLI resolution, config migration and validation run. No fake validator or loader.
    monkeypatch.setattr(train_module, "make_accelerator", _forbid_allocation_or_training)
    monkeypatch.setattr(train_module, "make_policy", _forbid_allocation_or_training)
    monkeypatch.setattr(train_module, "update_policy", _forbid_allocation_or_training)
    monkeypatch.setattr(PerceptronIsaacPolicy, "__init__", _forbid_allocation_or_training)

    with pytest.raises(
        ValueError, match=r"(?i)(raw|portable).*(unsupported|not supported)|unsupported.*(raw|portable)"
    ):
        train_module.train(cfg)

    assert isinstance(cfg.policy, PerceptronIsaacConfig)
    assert cfg.policy.pretrained_path == Path(cached.repo_id)
    assert cfg.policy.hf_model_path == "hf_model", "preflight must not rewrite serialized relative paths"
    assert {path: path.read_bytes() for path in cached.snapshot.rglob("*") if path.is_file()} == before
    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()


def test_cached_hub_raw_explicit_no_save_and_real_inference_stay_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = _cache_raw_policy(tmp_path, monkeypatch, None)
    before = {path: path.read_bytes() for path in cached.snapshot.rglob("*") if path.is_file()}
    cfg = _train_config(ACTConfig(device="cpu"), tmp_path / "run", None)
    _hub_cli(monkeypatch, cached)

    cfg.validate()
    assert isinstance(cfg.policy, PerceptronIsaacConfig)
    assert cfg.policy.pretrained_path == Path(cached.repo_id)
    assert cfg.policy.hf_model_path == "hf_model"
    assert not cfg.save_checkpoint and not cfg.policy.push_to_hub and not cfg.ema.enable
    policy = PerceptronIsaacPolicy.from_pretrained(cached.repo_id, local_files_only=True)

    assert policy._isaac_model is not None
    assert policy.config.artifact_kind == "neutral_debug"
    assert {path: path.read_bytes() for path in cached.snapshot.rglob("*") if path.is_file()} == before
    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize("intent", INTENTS)
def test_ordinary_cached_hub_intents_keep_real_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intent: Intent
) -> None:
    config = ACTConfig(device="cpu")
    cfg = _train_config(config, tmp_path / "run", intent)
    cached = _cache_policy(tmp_path, config, monkeypatch)
    _hub_cli(monkeypatch, cached)

    cfg.validate()

    assert isinstance(cfg.policy, ACTConfig)
    assert cfg.policy.pretrained_path == Path(cached.repo_id)
    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()


def test_ordinary_cached_hub_defaults_still_require_publish_repo_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ACTConfig(device="cpu")
    assert config.push_to_hub and config.repo_id is None
    cached = _cache_policy(tmp_path, config, monkeypatch)
    cfg = _train_config(ACTConfig(device="cpu"), tmp_path / "run", None)
    _hub_cli(monkeypatch, cached)

    with pytest.raises(ValueError, match="repo_id.*missing"):
        cfg.validate()

    assert isinstance(cfg.policy, ACTConfig)
    assert cfg.policy.push_to_hub and cfg.policy.repo_id is None
    loaded = PreTrainedConfig.from_pretrained(cached.repo_id, local_files_only=True)
    assert isinstance(loaded, ACTConfig)
    assert loaded.push_to_hub and loaded.repo_id is None
    assert cfg.output_dir is not None
    assert not cfg.output_dir.exists()
