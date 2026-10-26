"""Finalize real tiny native ISAAC saves once, after every package writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.common.train_utils import save_checkpoint
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import (
    _load_native_fixture,
    _source_package,
)


@pytest.fixture
def native_policy(tmp_path: Path) -> PerceptronIsaacPolicy:
    """Load the real native model with explicit test-only reduced geometry, not a topology double."""
    config = _source_package(tmp_path / "source")
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = _load_native_fixture(config)
    return policy


def _record_real_finalizations(
    policy: PerceptronIsaacPolicy, monkeypatch: pytest.MonkeyPatch
) -> list[frozenset[str]]:
    """Observe finalizer entry without bypassing its real manifest and digest writes."""
    original = policy.finalize_pretrained_package
    inventories: list[frozenset[str]] = []

    def record(save_directory: str | Path) -> None:
        root = Path(save_directory)
        inventories.append(frozenset(path.name for path in root.iterdir()))
        original(save_directory)

    monkeypatch.setattr(policy, "finalize_pretrained_package", record)
    return inventories


def test_save_checkpoint_finalizes_native_package_once_after_all_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_policy: PerceptronIsaacPolicy
) -> None:
    finalizations = _record_real_finalizations(native_policy, monkeypatch)
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), policy=native_policy.config)
    cfg.parallelism.resolve(1)
    # Real, empty pipelines isolate serialization order from ISAAC rendering and normalization.
    preprocessor = PolicyProcessorPipeline[dict[str, object], dict[str, object]](
        steps=[], name="policy_preprocessor"
    )
    postprocessor = PolicyProcessorPipeline[dict[str, object], dict[str, object]](
        steps=[], name="policy_postprocessor"
    )
    checkpoint = tmp_path / "checkpoint"

    save_checkpoint(
        checkpoint,
        step=7,
        cfg=cfg,
        policy=native_policy,
        optimizer=torch.optim.Adam(native_policy.parameters()),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    saved = checkpoint / PRETRAINED_MODEL_DIR
    sidecars = {"train_config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    assert len(finalizations) == 1, "Training saves must not finalize before writing their sidecars"
    assert sidecars <= finalizations[0]
    manifest = json.loads((saved / "mk1_trained_package_manifest.json").read_text())
    for filename in sidecars:
        assert manifest["files_sha256"][filename] == file_sha256(saved / filename)
    PerceptronIsaacPolicy._verify_saved_package(saved)


def test_standalone_native_save_finalizes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_policy: PerceptronIsaacPolicy
) -> None:
    """Control: moving ownership to HubMixin must preserve complete standalone saves."""
    finalizations = _record_real_finalizations(native_policy, monkeypatch)
    saved = tmp_path / "standalone"

    native_policy.save_pretrained(saved)

    assert len(finalizations) == 1
    assert (saved / "mk1_trained_package_manifest.json").is_file()
    PerceptronIsaacPolicy._verify_saved_package(saved)


def test_native_save_can_defer_finalization_until_sidecars_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_policy: PerceptronIsaacPolicy
) -> None:
    finalizations = _record_real_finalizations(native_policy, monkeypatch)
    saved = tmp_path / "deferred"

    native_policy.save_pretrained(saved, finalize_package=False)

    assert finalizations == [], "Deferred saves must leave finalization to the package writer"
    assert (saved / "hf_model" / "model.safetensors.index.json").is_file()
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), policy=native_policy.config)
    cfg.save_pretrained(saved)
    native_policy.finalize_pretrained_package(saved)

    assert len(finalizations) == 1
    assert "train_config.json" in finalizations[0]
    manifest = json.loads((saved / "mk1_trained_package_manifest.json").read_text())
    assert manifest["files_sha256"]["train_config.json"] == file_sha256(saved / "train_config.json")
    PerceptronIsaacPolicy._verify_saved_package(saved)
