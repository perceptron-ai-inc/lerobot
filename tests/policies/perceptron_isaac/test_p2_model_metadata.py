"""P2.02: model counts are not architecture; authenticated historical bytes remain binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest

pytest.importorskip("transformers", reason="requires the parent-owned ISAAC CPU environment")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.policies.perceptron_isaac.checkpoint_integrity import canonical_json_sha256, file_sha256
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    Mk1CheckpointContractError,
    validate_mk1_checkpoint,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.trained_package import (
    VerifiedTrainedTensorStorage,
    finalize_canonical_trained_package,
)
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _source_package
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _config,
    _write_checkpoint,
    _write_portable_isaac05_checkpoint,
)
from tests.policies.perceptron_isaac.test_trained_storage_contract import (
    _render_metadata,
    reduced_loader,  # noqa: F401 -- shared fixture changes geometry permission only
)

Family = Literal["isaac05", "legacy_mk1"]
MANIFEST = "mk1_trained_package_manifest.json"
DIGEST_FIELD = "mk1_trained_package_manifest_sha256"


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def _legacy_authenticated_package(root: Path, *, version: Literal[2, 3] = 3) -> PerceptronIsaacConfig:
    """Stamp historical v2/v3 formats independently of the evolving production writer.

    The count is present in both original policy JSON and manifest. Their full hashes,
    including that count, must be verified before any runtime compatibility projection.
    This creates a synthetic signed-format fixture, not public weights or a hash waiver.
    """
    _source_package(root)
    config_path = root / "config.json"
    payload = json.loads(config_path.read_text())
    payload["trained_steps"] = 100_000
    payload[DIGEST_FIELD] = None
    manifest = {
        "schema": f"perceptron_isaac_mk1_canonical_trained_package_manifest_v{version}",
        "artifact_kind": payload["artifact_kind"],
        "trained_steps": 100_000,
        "mk1_source_model_import_sha256": payload["mk1_source_model_import_sha256"],
        "model_storage": "canonical_inner",
        **({"metadata": {"tensor_storage_dtypes": ["F32"]}} if version == 3 else {}),
        "config_sha256": canonical_json_sha256(payload),
        "files_sha256": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in _files(root).items()
            if name not in {"config.json", MANIFEST}
        },
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    payload[DIGEST_FIELD] = file_sha256(root / MANIFEST)
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    loaded = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    assert isinstance(loaded, PerceptronIsaacConfig)
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(loaded, root)
    return loaded


@pytest.mark.parametrize("family", ["isaac05", "legacy_mk1"])
def test_missing_model_count_is_the_only_relaxed_source_gate(tmp_path: Path, family: Family) -> None:
    root = tmp_path / "model"
    if family == "isaac05":
        expected, raw, _ = _write_portable_isaac05_checkpoint(root)
        artifact_key = "isaac05_artifact"
        allowed_dtypes = frozenset({"F32"})
    else:
        raw = _config(test_geometry=True, mtp_present=False)
        expected = _write_checkpoint(root, config=raw)
        artifact_key = "genesis_artifact"
        allowed_dtypes = frozenset({"BF16"})
    before = _files(root)
    assert raw[artifact_key].pop("trained_steps") == 100_000
    (root / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    original_bytes = _files(root)
    assert {name for name in before if before[name] != original_bytes[name]} == {"config.json"}

    contract, inventory = validate_mk1_checkpoint(
        root, allow_test_only_reduced_geometry=True, allowed_storage_dtypes=allowed_dtypes
    )

    assert contract.artifact.artifact_kind == expected.artifact.artifact_kind == "trained_policy"
    assert contract.text == expected.text
    assert contract.vector_encoder == expected.vector_encoder
    assert contract.action_expert == expected.action_expert
    assert contract.expected_tensor_shapes() == expected.expected_tensor_shapes()
    assert set(inventory.tensors) == set(expected.expected_tensor_shapes())
    assert _files(root) == original_bytes, "validation must not invent a count or rewrite source bytes"


@pytest.mark.parametrize("omit_count", [False, True], ids=["count_present", "count_absent"])
@pytest.mark.parametrize("defect", ["kind", "geometry", "artifact_schema"])
def test_count_removal_does_not_accept_unsupported_architecture_or_identity(
    defect: str, omit_count: bool
) -> None:
    raw = _config(test_geometry=False)
    if omit_count:
        del raw["genesis_artifact"]["trained_steps"]
    if defect == "kind":
        raw["genesis_artifact"]["artifact_kind"] = "unknown"
        message = "artifact_kind"
    elif defect == "geometry":
        raw["text_config"]["hidden_size"] = 17
        message = "production MK1|hidden_size"
    else:
        raw["genesis_artifact"]["schema_version"] = 2
        message = "schema_version"
    with pytest.raises(Mk1CheckpointContractError, match=message):
        Mk1CheckpointContract.parse_allowlisted(raw)


def test_unsaved_config_does_not_invent_one_model_step() -> None:
    payload = PerceptronIsaacConfig(device="cpu").to_dict()
    assert payload.get("trained_steps") is None


def test_neutral_debug_kind_does_not_require_a_fake_zero_count() -> None:
    config = PerceptronIsaacConfig(device="cpu", artifact_kind="neutral_debug")
    assert config.artifact_kind == "neutral_debug"
    assert config.to_dict().get("trained_steps") is None


@pytest.mark.parametrize("artifact_kind", ["trained_policy", "neutral_debug"])
def test_historical_policy_count_is_retired_only_in_runtime_projection(
    tmp_path: Path, artifact_kind: str
) -> None:
    config = PerceptronIsaacConfig(device="cpu", artifact_kind=artifact_kind)
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw["trained_steps"] = 100_000 if artifact_kind == "trained_policy" else 0
    path.write_text(json.dumps(raw))
    before = path.read_bytes()

    loaded = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(loaded, PerceptronIsaacConfig)
    assert loaded.artifact_kind == artifact_kind
    assert "trained_steps" not in loaded.to_dict()
    assert path.read_bytes() == before


def test_historical_nested_training_config_retires_only_model_count(tmp_path: Path) -> None:
    from lerobot.configs.default import DatasetConfig

    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/unused", video_backend="pyav"),
        policy=PerceptronIsaacConfig(device="cpu", push_to_hub=False),
        steps=123,
        save_freq=17,
    )
    config.save_pretrained(tmp_path)
    path = tmp_path / TRAIN_CONFIG_NAME
    raw = json.loads(path.read_text())
    raw["policy"]["trained_steps"] = 100_000
    path.write_text(json.dumps(raw))
    before = path.read_bytes()

    loaded = TrainPipelineConfig.from_pretrained(path, local_files_only=True)

    assert loaded.steps == 123 and loaded.save_freq == 17
    assert isinstance(loaded.policy, PerceptronIsaacConfig)
    assert "trained_steps" not in loaded.policy.to_dict()
    assert path.read_bytes() == before


def test_new_package_writer_needs_no_model_step_count(tmp_path: Path) -> None:
    root = tmp_path / "package"
    _source_package(root)
    config_path = root / "config.json"
    raw = json.loads(config_path.read_text())
    raw.pop("trained_steps", None)
    config_path.write_text(json.dumps(raw))
    loaded = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    assert isinstance(loaded, PerceptronIsaacConfig)
    source_bytes = (root / "hf_model" / "config.json").read_bytes()

    finalize_canonical_trained_package(
        loaded, root, mk1_source_model_import_sha256=loaded.mk1_source_model_import_sha256
    )

    manifest = json.loads((root / MANIFEST).read_text())
    assert "trained_steps" not in manifest
    assert manifest["artifact_kind"] == "trained_policy"
    assert manifest["mk1_source_model_import_sha256"] == loaded.mk1_source_model_import_sha256
    assert (root / "hf_model" / "config.json").read_bytes() == source_bytes
    assert isinstance(
        PerceptronIsaacPolicy._verify_trained_package_manifest(loaded, root), VerifiedTrainedTensorStorage
    )


@pytest.mark.parametrize("version", [2, 3])
def test_historical_count_bearing_manifest_verifies_without_rewriting(
    tmp_path: Path, version: Literal[2, 3]
) -> None:
    root = tmp_path / "package"
    loaded = _legacy_authenticated_package(root, version=version)
    original_bytes = _files(root)

    storage = PerceptronIsaacPolicy._verify_trained_package_manifest(loaded, root)

    if version == 3:
        assert isinstance(storage, VerifiedTrainedTensorStorage)
        assert storage.dtypes == frozenset({"F32"})
    else:
        assert storage is None, "v2 must not gain authenticated-v3 storage privileges"
    assert _files(root) == original_bytes


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("target", ["policy_count", "manifest_count", "source_count", "source_identity"])
def test_historical_count_bytes_are_not_excluded_from_authentication(
    tmp_path: Path, target: str, version: Literal[2, 3]
) -> None:
    root = tmp_path / "package"
    loaded = _legacy_authenticated_package(root, version=version)
    PerceptronIsaacPolicy._verify_trained_package_manifest(loaded, root)
    if target == "policy_count":
        path = root / "config.json"
        raw = json.loads(path.read_text())
        raw["trained_steps"] += 1
    elif target == "manifest_count":
        path = root / MANIFEST
        raw = json.loads(path.read_text())
        raw["trained_steps"] += 1
    elif target == "source_count":
        path = root / "hf_model" / "config.json"
        raw = json.loads(path.read_text())
        raw["isaac05_artifact"]["trained_steps"] += 1
    else:
        path = root / "hf_model" / "source_mk1_model_import.json"
        raw = {"fixture": "different-source"}
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="digest mismatch|identity disagrees"):
        PerceptronIsaacPolicy._verify_trained_package_manifest(loaded, root)


@pytest.mark.usefixtures("reduced_loader")
def test_backbone_identity_does_not_compare_model_step_counts(tmp_path: Path) -> None:
    root = tmp_path / "package"
    config = _source_package(root)
    _render_metadata(config, root / "native_render_metadata.json")
    # Inner source retains its historical 100000; an unsaved policy has no matching model count.
    raw = json.loads((root / "config.json").read_text())
    raw.pop("trained_steps", None)
    (root / "config.json").write_text(json.dumps(raw))
    loaded = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    assert isinstance(loaded, PerceptronIsaacConfig)
    finalize_canonical_trained_package(
        loaded, root, mk1_source_model_import_sha256=loaded.mk1_source_model_import_sha256
    )
    original_bytes = _files(root)

    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    policy._load_backbone()

    assert policy._isaac_model is not None
    assert policy.config.artifact_kind == "trained_policy"
    assert _files(root) == original_bytes
