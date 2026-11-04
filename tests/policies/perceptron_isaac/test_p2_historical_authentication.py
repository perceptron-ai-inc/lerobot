"""Historical model-count migration must preserve each format's actual trust boundary.

Synthetic complete legacy packages use the maintained tiny tensor geometry, not public
weights. Only the import reader's geometry permission is widened; hashes, inventories,
provenance, tokenizer validation and the public policy reader remain real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import pytest

pytest.importorskip("transformers", reason="requires the parent-owned offline ISAAC CPU environment")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.perceptron_isaac import mk1_checkpoint_contract
from lerobot.policies.perceptron_isaac.checkpoint_integrity import canonical_json_sha256, file_sha256
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    validate_mk1_checkpoint,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _config as _policy_config
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config, _write_checkpoint
from tests.policies.perceptron_isaac.test_trained_storage_contract import _render_metadata

HistoricalFormat = Literal["v1", "import", "neutral"]
FORMATS: tuple[HistoricalFormat, ...] = ("v1", "import", "neutral")
TRAINED_MANIFEST = "mk1_trained_package_manifest.json"
NEUTRAL_MANIFEST = "mk1_neutral_debug_manifest.json"
IMPORT_RECORD = "mk1_model_import.json"
DIGEST_FIELD = "mk1_trained_package_manifest_sha256"


class ArtifactIdentity(TypedDict):
    schema_version: int
    artifact_kind: str
    trained_steps: NotRequired[int]


class ImportedShard(TypedDict):
    filename: str
    source_sha256: str
    output_sha256: str
    tensor_count: int


class VocabRecord(TypedDict):
    schema: str
    algorithm: str
    token_count: int
    tokenizer_sha256: str
    vocab_sha256: str


class ImportRecord(TypedDict):
    schema: str
    algorithm: str
    backbone_family: str
    rmsnorm_weight_convention: str
    artifact: ArtifactIdentity
    source_files_sha256: dict[str, str]
    files_sha256: dict[str, str]
    qwen35_vocab: VocabRecord
    shards: list[ImportedShard]


@dataclass(frozen=True)
class HistoricalPackage:
    root: Path
    expected_contract: Mk1CheckpointContract


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture(autouse=True)
def isolate_vocab_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Public loading publishes a retained vocab path; restore it between tests.
    monkeypatch.setenv("QWEN35_VOCAB_PATH", "")


@pytest.fixture
def reduced_import_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    read = mk1_checkpoint_contract.read_mk1_config

    def read_reduced(
        model_dir: str | Path, *, allow_test_only_reduced_geometry: bool = False
    ) -> Mk1CheckpointContract:
        return read(model_dir, allow_test_only_reduced_geometry=True)

    monkeypatch.setattr(mk1_checkpoint_contract, "read_mk1_config", read_reduced)


def _stamp_v1(root: Path) -> None:
    """Historical v1 delegates the inner tree to its byte-copy import record."""
    payload = _json(root / "config.json")
    payload[DIGEST_FIELD] = None
    manifest = {
        "schema": "perceptron_isaac_mk1_trained_package_manifest_v1",
        "artifact_kind": payload["artifact_kind"],
        "mk1_model_import_sha256": payload["mk1_model_import_sha256"],
        **({"trained_steps": payload["trained_steps"]} if "trained_steps" in payload else {}),
        "config_sha256": canonical_json_sha256(payload),
        "files_sha256": {
            name: file_sha256(root / name)
            for name in _bytes(root)
            if name not in {"config.json", TRAINED_MANIFEST} and not name.startswith("hf_model/")
        },
    }
    _write_json(root / TRAINED_MANIFEST, manifest)
    payload[DIGEST_FIELD] = file_sha256(root / TRAINED_MANIFEST)
    _write_json(root / "config.json", payload)


def _rebind_import(root: Path, record: object, format: HistoricalFormat) -> None:
    _write_json(root / "hf_model" / IMPORT_RECORD, record)
    payload = _json(root / "config.json")
    payload["mk1_model_import_sha256"] = file_sha256(root / "hf_model" / IMPORT_RECORD)
    _write_json(root / "config.json", payload)
    if format == "v1":
        _stamp_v1(root)


def _historical_package(root: Path, format: HistoricalFormat, *, count_present: bool) -> HistoricalPackage:
    root.mkdir()
    raw = _config(test_geometry=True, mtp_present=False)
    artifact: ArtifactIdentity = {
        "schema_version": 1,
        "artifact_kind": "neutral_debug" if format == "neutral" else "trained_policy",
    }
    if count_present:
        artifact["trained_steps"] = 0 if format == "neutral" else 100_000
    raw["genesis_artifact"] = artifact
    # Preserve a real full base-vocabulary import without allocating a production backbone.
    raw["text_config"].update(vocab_size=256279, bos_token_id=248044, eos_token_id=248044)
    raw.update(
        image_token_id=248056, video_token_id=248057, vision_start_token_id=248053, vision_end_token_id=248054
    )
    raw["genesis_coord_tokens"].update(offset=248320, size=1001)
    expected = _write_checkpoint(root / "hf_model", config=raw)
    model = root / "hf_model"
    vocab = {f"fixture-token-{index}": index for index in range(248044)}
    tokenizer = {
        "model": {"vocab": vocab},
        "added_tokens": [
            {"content": token, "id": index, "special": True}
            for token, index in (
                ("<|endoftext|>", 248044),
                ("<|vision_start|>", 248053),
                ("<|vision_end|>", 248054),
                ("<|image_pad|>", 248056),
                ("<|video_pad|>", 248057),
            )
        ],
    }
    _write_json(model / "tokenizer.json", tokenizer)
    _write_json(model / "vocab.json", vocab)
    (root / "isaac_deployment_adapter.json").write_text("{}\n")
    config = _policy_config(
        hf_model_path="hf_model",
        artifact_kind=artifact["artifact_kind"],
        vector_max_states=4,
        max_action_dim=4,
        max_action_horizon=3,
        native_render_metadata_path="native_render_metadata.json",
        deployment_adapter_sha256=file_sha256(root / "isaac_deployment_adapter.json"),
        train_storage_fp32=True,
    )
    _render_metadata(config, root / "native_render_metadata.json")
    metadata = _json(root / "native_render_metadata.json")
    metadata["mharmony_reserved_token_groups"] = [{"name": "coord", "offset": 248320, "size": 1001}]
    _write_json(root / "native_render_metadata.json", metadata)
    config.save_pretrained(root)
    payload = _json(root / "config.json")
    if count_present:
        payload["trained_steps"] = artifact["trained_steps"]
    _write_json(root / "config.json", payload)
    if format == "neutral":
        files = {name: file_sha256(root / name) for name in _bytes(root)}
        manifest = {
            "schema_version": 1,
            "artifact_kind": "neutral_debug",
            **({"trained_steps": 0} if count_present else {}),
            "source_config_sha256": files["hf_model/config.json"],
            "source_index_sha256": files["hf_model/model.safetensors.index.json"],
            "model_materialization": {path.name: "copy" for path in model.iterdir()},
            "files_sha256": files,
        }
        _write_json(root / NEUTRAL_MANIFEST, manifest)
    else:
        files = {path.name: file_sha256(path) for path in model.iterdir()}
        source_files = {name: digest for name, digest in files.items() if name != "vocab.json"}
        weight_map = _json(model / "model.safetensors.index.json")["weight_map"]
        record: ImportRecord = {
            "schema": "perceptron_isaac_mk1_model_import_v2",
            "algorithm": "byte_for_byte_safetensors_copy",
            "backbone_family": "mk1_qwen3_6_moe",
            "rmsnorm_weight_convention": "zero_centered_1_plus_weight",
            "artifact": artifact,
            "source_files_sha256": source_files,
            "files_sha256": files,
            "qwen35_vocab": {
                "schema": "perceptron_isaac_qwen35_vocab_v1",
                "algorithm": "extract_tokenizer_json_model_vocab",
                "token_count": len(vocab),
                "tokenizer_sha256": files["tokenizer.json"],
                "vocab_sha256": files["vocab.json"],
            },
            "shards": [
                {
                    "filename": name,
                    "source_sha256": source_files[name],
                    "output_sha256": files[name],
                    "tensor_count": sum(value == name for value in weight_map.values()),
                }
                for name in sorted(set(weight_map.values()))
            ],
        }
        _rebind_import(root, record, format)
    return HistoricalPackage(root, expected)


def _read_config(root: Path) -> PerceptronIsaacConfig:
    loaded = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    assert isinstance(loaded, PerceptronIsaacConfig)
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(loaded, root)
    return loaded


def _read_actual(format: HistoricalFormat, root: Path) -> PerceptronIsaacConfig:
    if format == "import":
        loaded = _read_config(root)
        PerceptronIsaacPolicy._verify_mk1_model_import(loaded, root)
        return loaded
    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert policy._isaac_model is None, "historical metadata validation must not allocate weights"
    return policy.config


@pytest.mark.usefixtures("reduced_import_geometry")
@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("count_present", [True, False], ids=["count-bearing", "count-free"])
def test_complete_historical_readers_preserve_bytes_identity_geometry_and_inventory(
    tmp_path: Path, format: HistoricalFormat, count_present: bool
) -> None:
    package = _historical_package(tmp_path / "package", format, count_present=count_present)
    root = package.root
    before = _bytes(root)

    loaded = _read_actual(format, root)
    contract, inventory = validate_mk1_checkpoint(root / "hf_model", allow_test_only_reduced_geometry=True)

    assert contract == package.expected_contract
    assert contract.artifact.artifact_kind == loaded.artifact_kind
    assert loaded.artifact_kind == ("neutral_debug" if format == "neutral" else "trained_policy")
    assert (loaded.vector_max_states, loaded.max_action_dim, loaded.max_action_horizon) == (4, 4, 3)
    assert inventory.keys == frozenset(contract.expected_tensor_shapes())
    assert all(header.dtype == "BF16" for header in inventory.tensors.values())
    for shard in inventory.shards:
        inventory.verify_shard_identity(shard)
    assert "trained_steps" not in loaded.to_dict()
    source_artifact = _json(root / "hf_model/config.json")["genesis_artifact"]
    assert ("trained_steps" in source_artifact) == count_present
    if format == "neutral":
        manifest = _json(root / NEUTRAL_MANIFEST)
        assert manifest["files_sha256"] == {
            name: file_sha256(root / name) for name in before if name != NEUTRAL_MANIFEST
        }
        assert manifest["source_config_sha256"] == file_sha256(root / "hf_model/config.json")
        assert manifest["source_index_sha256"] == file_sha256(root / "hf_model/model.safetensors.index.json")
    else:
        record = _json(root / "hf_model" / IMPORT_RECORD)
        assert record["artifact"] == source_artifact, "retired counts still bind exact historical identity"
        assert loaded.mk1_model_import_sha256 == file_sha256(root / "hf_model" / IMPORT_RECORD)
        assert record["files_sha256"] == {
            path.name: file_sha256(path)
            for path in (root / "hf_model").iterdir()
            if path.name != IMPORT_RECORD
        }
        assert record["source_files_sha256"] == {
            name: digest for name, digest in record["files_sha256"].items() if name != "vocab.json"
        }
        assert (
            _json(root / "hf_model/vocab.json") == _json(root / "hf_model/tokenizer.json")["model"]["vocab"]
        )
        if format == "v1":
            manifest = _json(root / TRAINED_MANIFEST)
            payload = _json(root / "config.json")
            assert payload[DIGEST_FIELD] == file_sha256(root / TRAINED_MANIFEST)
            payload[DIGEST_FIELD] = None
            assert manifest["config_sha256"] == canonical_json_sha256(payload)
            assert manifest["files_sha256"] == {
                name: file_sha256(root / name)
                for name in before
                if name not in {"config.json", TRAINED_MANIFEST} and not name.startswith("hf_model/")
            }
    assert _bytes(root) == before


@pytest.mark.usefixtures("reduced_import_geometry")
@pytest.mark.parametrize("count_present", [True, False], ids=["count-bearing", "count-free"])
def test_historical_neutral_debug_still_refuses_both_action_apis(tmp_path: Path, count_present: bool) -> None:
    package = _historical_package(tmp_path / "package", "neutral", count_present=count_present)
    before = _bytes(package.root)
    policy = PerceptronIsaacPolicy.from_pretrained(package.root, local_files_only=True)

    with pytest.raises(RuntimeError, match="rejects neutral_debug"):
        policy.select_action({})
    with pytest.raises(RuntimeError, match="rejects neutral_debug"):
        policy.predict_action_chunk({})

    assert policy._isaac_model is None
    assert _bytes(package.root) == before


@pytest.mark.usefixtures("reduced_import_geometry")
@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("count_present", [True, False], ids=["count-bearing", "count-free"])
@pytest.mark.parametrize("defect", ["unlisted", "missing", "changed-bytes"])
def test_historical_readers_keep_exhaustive_inventory_and_file_hashes(
    tmp_path: Path, format: HistoricalFormat, count_present: bool, defect: str
) -> None:
    package = _historical_package(tmp_path / "package", format, count_present=count_present)
    root = package.root
    # Import owns hf_model; v1 owns the outer tree; neutral owns the complete tree.
    target = root / "hf_model" if format == "import" else root
    if defect == "unlisted":
        (target / "unlisted.json").write_text("{}\n")
        message = "inventory disagrees"
    else:
        path = target / ("vocab.json" if format == "import" else "native_render_metadata.json")
        if defect == "missing":
            path.unlink()
            # Public path resolution is also a real fail-closed boundary.
            message = "inventory disagrees|does not exist"
        else:
            path.write_text(path.read_text() + " ")
            message = "digest mismatch"
    before = _bytes(root)

    with pytest.raises(RuntimeError, match=message):
        _read_actual(format, root)

    assert _bytes(root) == before


@pytest.mark.usefixtures("reduced_import_geometry")
@pytest.mark.parametrize("format", ["v1", "import"])
@pytest.mark.parametrize("count_present", [True, False], ids=["count-bearing", "count-free"])
@pytest.mark.parametrize(
    "defect", ["artifact-count", "artifact-kind", "source-provenance", "shard-count", "geometry"]
)
def test_authenticated_import_still_binds_exact_source_identity_and_geometry(
    tmp_path: Path, format: HistoricalFormat, count_present: bool, defect: str
) -> None:
    package = _historical_package(tmp_path / "package", format, count_present=count_present)
    root = package.root
    record = _json(root / "hf_model" / IMPORT_RECORD)
    if defect == "artifact-count":
        record["artifact"]["trained_steps"] = 100_001
        message = "inner config artifact identity disagrees"
    elif defect == "artifact-kind":
        record["artifact"]["artifact_kind"] = "neutral_debug"
        message = "import artifact identity disagrees"
    elif defect == "source-provenance":
        record["source_files_sha256"]["config.json"] = "0" * 64
        message = "byte-copy provenance disagrees"
    elif defect == "shard-count":
        record["shards"][0]["tensor_count"] += 1
        message = "shard record disagrees"
    else:
        payload = _json(root / "config.json")
        payload["max_action_horizon"] += 1
        _write_json(root / "config.json", payload)
        message = "outer checkpoint maxima disagree"
    # Authenticate the invalid declaration honestly, so refusal is semantic, not stale hashes.
    _rebind_import(root, record, format)
    before = _bytes(root)

    with pytest.raises(RuntimeError, match=message):
        _read_actual(format, root)

    assert _bytes(root) == before


@pytest.mark.usefixtures("reduced_import_geometry")
@pytest.mark.parametrize("target", ["policy-count", "manifest-count", "inner-count", "record-count"])
def test_v1_historical_count_bytes_are_still_authenticated(tmp_path: Path, target: str) -> None:
    package = _historical_package(tmp_path / "package", "v1", count_present=True)
    root = package.root
    if target == "policy-count":
        path, key = root / "config.json", "trained_steps"
    elif target == "manifest-count":
        path, key = root / TRAINED_MANIFEST, "trained_steps"
    elif target == "inner-count":
        path, key = root / "hf_model/config.json", "genesis_artifact"
    else:
        path, key = root / "hf_model" / IMPORT_RECORD, "artifact"
    payload = _json(path)
    if key == "trained_steps":
        payload[key] += 1
    else:
        payload[key]["trained_steps"] += 1
    _write_json(path, payload)
    before = _bytes(root)

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _read_actual("v1", root)

    assert _bytes(root) == before


@pytest.mark.usefixtures("reduced_import_geometry")
def test_authenticated_v1_outer_count_is_not_source_model_compatibility(tmp_path: Path) -> None:
    package = _historical_package(tmp_path / "package", "v1", count_present=True)
    root = package.root
    payload = _json(root / "config.json")
    payload["trained_steps"] = 7
    _write_json(root / "config.json", payload)
    _stamp_v1(root)
    before = _bytes(root)

    loaded = _read_actual("v1", root)

    assert "trained_steps" not in loaded.to_dict()
    assert _json(root / TRAINED_MANIFEST)["trained_steps"] == 7
    assert _json(root / "hf_model" / IMPORT_RECORD)["artifact"]["trained_steps"] == 100_000
    assert _bytes(root) == before


@pytest.mark.parametrize("target", ["policy-count", "inner-count"])
def test_neutral_historical_count_in_hashed_config_is_not_waived(tmp_path: Path, target: str) -> None:
    package = _historical_package(tmp_path / "package", "neutral", count_present=True)
    root = package.root
    path = root / ("config.json" if target == "policy-count" else "hf_model/config.json")
    payload = _json(path)
    if target == "policy-count":
        payload["trained_steps"] += 1
    else:
        payload["genesis_artifact"]["trained_steps"] += 1
    _write_json(path, payload)
    before = _bytes(root)

    with pytest.raises(RuntimeError, match="Neutral-debug package digest mismatch"):
        _read_actual("neutral", root)

    assert _bytes(root) == before


@pytest.mark.parametrize("defect", ["kind", "source-config", "source-index", "materialization"])
def test_neutral_manifest_keeps_its_historical_unsigned_identity_contract(
    tmp_path: Path, defect: str
) -> None:
    package = _historical_package(tmp_path / "package", "neutral", count_present=True)
    root = package.root
    manifest = _json(root / NEUTRAL_MANIFEST)
    if defect == "kind":
        manifest["artifact_kind"] = "trained_policy"
        message = "manifest identity disagrees"
    elif defect == "source-config":
        manifest["source_config_sha256"] = "0" * 64
        message = "source config digest disagrees"
    elif defect == "source-index":
        manifest["source_index_sha256"] = "0" * 64
        message = "source index digest disagrees"
    else:
        manifest["model_materialization"].pop("tokenizer.json")
        message = "materialization inventory is malformed"
    _write_json(root / NEUTRAL_MANIFEST, manifest)
    before = _bytes(root)

    with pytest.raises(RuntimeError, match=message):
        _read_actual("neutral", root)

    assert _bytes(root) == before
