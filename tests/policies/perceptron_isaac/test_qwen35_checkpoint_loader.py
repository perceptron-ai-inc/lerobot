import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.perceptron_isaac import modeling_qwen35_vla
from lerobot.policies.perceptron_isaac.checkpoint_integrity import canonical_json_sha256, file_sha256
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
    load_qwen35_vla_from_hf,
    load_sharded_safetensors_into_model,
)
from lerobot.policies.perceptron_isaac.qwen35_checkpoint import (
    QWEN35_CONVERSION_ALGORITHM,
    QWEN35_CONVERSION_DTYPE,
    QWEN35_CONVERSION_PROVENANCE_FILE,
    QWEN35_CONVERSION_PROVENANCE_SCHEMA,
    QWEN35_IMPORT_PROVENANCE_FILE,
    QWEN35_IMPORT_PROVENANCE_SCHEMA,
    QWEN35_IMPORT_PROVENANCE_SCHEMA_V2,
    QWEN35_PACKAGED_SOURCE_FILENAMES,
    QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE,
    QWEN35_VOCAB_ALGORITHM,
    QWEN35_VOCAB_SCHEMA,
    verify_qwen35_imported_package,
)
from lerobot.policies.perceptron_isaac.trained_package import (
    QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME,
)


def _write_indexed_checkpoint(root: Path) -> tuple[torch.nn.Module, list[str]]:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    state = model.state_dict()
    keys = sorted(state)
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    save_file({key: state[key] for key in keys[:2]}, root / shard_names[0])
    save_file({key: state[key] for key in keys[2:]}, root / shard_names[1])
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {key: shard_names[0] if key in keys[:2] else shard_names[1] for key in keys},
            }
        )
    )
    return model, shard_names


def _write_imported_package(root: Path) -> Path:
    hf_model = root / "hf_model"
    hf_model.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"type": "perceptron_isaac", "hf_model_path": "hf_model"}) + "\n"
    )
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    shard_states = [
        {
            "model.language_model.norm.weight": torch.ones(2),
            "model.visual.weight": torch.ones(1),
        },
        {"model.action_expert.weight": torch.ones(2, 2)},
    ]
    for name, state in zip(shard_names, shard_states, strict=True):
        save_file(state, hf_model / name)
    (hf_model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    key: name for name, state in zip(shard_names, shard_states, strict=True) for key in state
                },
            }
        )
    )

    assets = {
        "config.json": {"model_type": "qwen3_5"},
        "tokenizer.json": {"model": {"vocab": {"token": 0}}},
        "processor_config.json": {"processor_class": "FixtureProcessor"},
        "vocab.json": {"token": 0},
    }
    for filename, payload in assets.items():
        (hf_model / filename).write_text(json.dumps(payload, sort_keys=True) + "\n")

    conversion = {
        "schema": QWEN35_CONVERSION_PROVENANCE_SCHEMA,
        "algorithm": QWEN35_CONVERSION_ALGORITHM,
        "conversion_dtype": QWEN35_CONVERSION_DTYPE,
        "converted_key_count": 1,
        "converted_keys_sha256": canonical_json_sha256(["model.language_model.norm.weight"]),
        "qwen35_vocab": {
            "schema": QWEN35_VOCAB_SCHEMA,
            "algorithm": QWEN35_VOCAB_ALGORITHM,
            "token_count": 1,
            "tokenizer_sha256": file_sha256(hf_model / "tokenizer.json"),
            "vocab_sha256": file_sha256(hf_model / "vocab.json"),
        },
        "shards": [
            {
                "filename": name,
                "source_sha256": "2" * 64,
                "output_sha256": file_sha256(hf_model / name),
                "tensor_count": len(state),
            }
            for name, state in zip(shard_names, shard_states, strict=True)
        ],
    }
    (hf_model / QWEN35_CONVERSION_PROVENANCE_FILE).write_text(
        json.dumps(conversion, indent=2, sort_keys=True) + "\n"
    )
    for filename in QWEN35_PACKAGED_SOURCE_FILENAMES:
        (root / filename).write_text(json.dumps({"fixture": filename}, sort_keys=True) + "\n")
    adapter = root / "isaac_deployment_adapter.json"
    adapter.write_text('{"fixture": true}\n')
    source_hashes = {
        filename: file_sha256(hf_model / filename)
        for filename in (
            "config.json",
            "tokenizer.json",
            "processor_config.json",
            "model.safetensors.index.json",
        )
    }
    source_hashes.update(dict.fromkeys(shard_names, "2" * 64))
    source_hashes.update(
        {filename: file_sha256(root / filename) for filename in QWEN35_PACKAGED_SOURCE_FILENAMES}
    )
    provenance = {
        "schema": QWEN35_IMPORT_PROVENANCE_SCHEMA,
        "contract_authentication": "hf_and_dcp_identity_authenticated",
        "dcp_identity_sha256": "3" * 64,
        "policy_state_identity_source": "hf_and_dcp_identity_authenticated",
        "source_artifact_manifest_sha256": "4" * 64,
        "authenticated_import_identity_sha256": "5" * 64,
        "policy_state_dataset": "fixture/qwen35",
        "normalization_scope": "fixture",
        "objective": "Flow",
        "conditioning_deployment": None,
        "source_hashes": source_hashes,
        "deployment_adapter": {
            "filename": adapter.name,
            "sha256": file_sha256(adapter),
            "validation_status": "fixture_validated",
        },
        "rmsnorm_conversion": conversion,
        "fast_processor": {
            "repository": "fixture/fast",
            "revision": "6" * 40,
            "tree_sha256": "7" * 64,
            "file_count": 1,
        },
    }
    (root / QWEN35_IMPORT_PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return hf_model


def _rewrite_conversion(root: Path, conversion: dict) -> None:
    (root / "hf_model" / QWEN35_CONVERSION_PROVENANCE_FILE).write_text(
        json.dumps(conversion, indent=2, sort_keys=True) + "\n"
    )
    provenance_path = root / QWEN35_IMPORT_PROVENANCE_FILE
    provenance = json.loads(provenance_path.read_text())
    provenance["rmsnorm_conversion"] = conversion
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def _write_test_trained_manifest(root: Path) -> None:
    manifest_path = root / QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "files_sha256": {
                    path.name: "0" * 64
                    for path in root.iterdir()
                    if path.name not in {"config.json", "hf_model", manifest_path.name}
                }
            }
        )
        + "\n"
    )


def test_single_safetensors_rejects_added_shard_override(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    save_file(model.state_dict(), tmp_path / "model-00001-of-00001.safetensors")

    with pytest.raises(ValueError, match="both model.safetensors and sharded model files"):
        load_sharded_safetensors_into_model(model, tmp_path)


def test_indexed_safetensors_rejects_unindexed_shard_file(tmp_path: Path) -> None:
    model, _ = _write_indexed_checkpoint(tmp_path)
    save_file({"unused": torch.zeros(1)}, tmp_path / "model-99999-of-99999.safetensors")

    with pytest.raises(ValueError, match="unindexed shard files"):
        load_sharded_safetensors_into_model(model, tmp_path)


def test_safetensors_index_rejects_duplicate_tensor_keys(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    shard_name = "model-00001-of-00001.safetensors"
    save_file(model.state_dict(), tmp_path / shard_name)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"weight": "model-00001-of-00001.safetensors", '
        '"weight": "model-00001-of-00001.safetensors", '
        '"bias": "model-00001-of-00001.safetensors"}}'
    )

    with pytest.raises(ValueError, match="duplicate key 'weight'"):
        load_sharded_safetensors_into_model(model, tmp_path)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "tokenizer.json",
        "processor_config.json",
        "model.safetensors.index.json",
    ],
)
def test_import_provenance_binds_every_consumed_runtime_asset(tmp_path: Path, filename: str) -> None:
    hf_model = _write_imported_package(tmp_path)
    verify_qwen35_imported_package(tmp_path)

    with (hf_model / filename).open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(RuntimeError, match=filename):
        verify_qwen35_imported_package(tmp_path)


def test_import_provenance_rejects_unlisted_runtime_asset(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    (hf_model / "special_tokens_map.json").write_text("{}\n")

    with pytest.raises(RuntimeError, match="unlisted runtime assets.*special_tokens_map.json"):
        verify_qwen35_imported_package(tmp_path)


def test_import_provenance_rejects_extra_model_shard(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    save_file({"unused": torch.zeros(1)}, hf_model / "model-99999-of-99999.safetensors")

    with pytest.raises(RuntimeError, match="model shard inventory.*unindexed shard"):
        verify_qwen35_imported_package(tmp_path)


def test_direct_loader_verifies_imported_package_before_model_allocation(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    with (hf_model / "config.json").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(RuntimeError, match="config.json"):
        load_qwen35_vla_from_hf(hf_model)


def test_direct_loader_rejects_deleted_import_provenance(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    (tmp_path / QWEN35_IMPORT_PROVENANCE_FILE).unlink()

    with pytest.raises(RuntimeError, match="import provenance"):
        load_qwen35_vla_from_hf(hf_model)


def test_direct_loader_rejects_deleted_conversion_marker(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    (hf_model / QWEN35_CONVERSION_PROVENANCE_FILE).unlink()

    with pytest.raises(RuntimeError, match="conversion"):
        load_qwen35_vla_from_hf(hf_model)


def test_direct_loader_rejects_active_and_source_conversion_markers(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    (hf_model / QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE).write_bytes(b"unbound lineage")

    with pytest.raises(RuntimeError, match="unsupported entries.*source_rmsnorm_conversion"):
        load_qwen35_vla_from_hf(hf_model)


def test_direct_loader_rejects_deleted_import_and_conversion_markers(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    (tmp_path / QWEN35_IMPORT_PROVENANCE_FILE).unlink()
    (hf_model / QWEN35_CONVERSION_PROVENANCE_FILE).unlink()

    with pytest.raises(RuntimeError, match="import provenance"):
        load_qwen35_vla_from_hf(hf_model)


def test_direct_loader_rejects_stripped_package_using_parent_policy_config(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    (tmp_path / QWEN35_IMPORT_PROVENANCE_FILE).unlink()
    (hf_model / QWEN35_CONVERSION_PROVENANCE_FILE).unlink()
    (tmp_path / "isaac_deployment_adapter.json").unlink()
    for filename in QWEN35_PACKAGED_SOURCE_FILENAMES:
        (tmp_path / filename).unlink()

    with pytest.raises(RuntimeError, match="import provenance"):
        load_qwen35_vla_from_hf(hf_model)


def test_direct_loader_consumes_verified_snapshot_when_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_model = _write_imported_package(tmp_path)
    original_config = (hf_model / "config.json").read_bytes()
    loaded_from: list[Path] = []

    def fake_load(stable_hf_model: str | Path, **_kwargs):
        stable_path = Path(stable_hf_model)
        loaded_from.append(stable_path)
        (hf_model / "config.json").write_bytes(b"source changed after verification")
        assert stable_path != hf_model
        assert (stable_path / "config.json").read_bytes() == original_config
        return object(), object()

    monkeypatch.setattr(modeling_qwen35_vla, "_load_qwen35_vla_from_hf", fake_load)

    load_qwen35_vla_from_hf(hf_model, apply_offset_norm=False)

    assert loaded_from and not loaded_from[0].exists()


def test_direct_loader_routes_trained_package_through_outer_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lerobot.policies.perceptron_isaac import qwen35_checkpoint
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    hf_model = _write_imported_package(tmp_path)
    (tmp_path / QWEN35_IMPORT_PROVENANCE_FILE).unlink()
    (hf_model / QWEN35_CONVERSION_PROVENANCE_FILE).rename(hf_model / QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE)
    policy_config = json.loads((tmp_path / "config.json").read_text())
    policy_config["apply_offset_norm"] = False
    (tmp_path / "config.json").write_text(json.dumps(policy_config) + "\n")
    _write_test_trained_manifest(tmp_path)
    verified_roots: list[Path] = []
    loaded_from: list[Path] = []
    snapshot_calls: list[Path] = []
    snapshot_trained_package = qwen35_checkpoint._snapshot_verified_qwen35_trained_package

    def track_snapshot(package_root: str | Path, **kwargs):
        snapshot_calls.append(Path(package_root))
        return snapshot_trained_package(package_root, **kwargs)

    def fake_verify(_config, snapshot_root: str | Path):
        root = Path(snapshot_root)
        verified_roots.append(root)
        assert root != tmp_path

    def fake_load(stable_hf_model: str | Path, **_kwargs):
        loaded_from.append(Path(stable_hf_model))
        return object(), object()

    monkeypatch.setattr(
        PerceptronIsaacPolicy,
        "_verify_qwen35_trained_package_manifest",
        staticmethod(fake_verify),
    )
    monkeypatch.setattr(
        modeling_qwen35_vla,
        "_snapshot_verified_qwen35_trained_package",
        track_snapshot,
    )
    monkeypatch.setattr(
        qwen35_checkpoint,
        "_snapshot_verified_qwen35_trained_package",
        track_snapshot,
    )
    monkeypatch.setattr(modeling_qwen35_vla, "_load_qwen35_vla_from_hf", fake_load)

    load_qwen35_vla_from_hf(hf_model, apply_offset_norm=False)

    assert verified_roots and [path.resolve() for path in loaded_from] == [
        (verified_roots[0] / "hf_model").resolve()
    ]
    assert snapshot_calls == [tmp_path]
    assert not verified_roots[0].exists()


def test_direct_trained_loader_rejects_unbound_root_entry_before_snapshot(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    (tmp_path / QWEN35_IMPORT_PROVENANCE_FILE).unlink()
    (hf_model / QWEN35_CONVERSION_PROVENANCE_FILE).rename(hf_model / QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE)
    _write_test_trained_manifest(tmp_path)
    (tmp_path / "unbound.bin").write_bytes(b"must not be copied")

    with pytest.raises(RuntimeError, match="unbound root entries.*unbound.bin"):
        load_qwen35_vla_from_hf(hf_model, apply_offset_norm=False)


def test_direct_loader_rejects_detached_legacy_canonical_marker_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_model = tmp_path / "standalone-canonical-hf-model"
    hf_model.mkdir()
    (hf_model / QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE).write_text("{}\n")

    def fail_load(*_args, **_kwargs):
        raise AssertionError("detached canonical weights reached model allocation")

    monkeypatch.setattr(modeling_qwen35_vla, "_load_qwen35_vla_from_hf", fail_load)
    with pytest.raises(RuntimeError, match="Detached canonical Qwen3.5 hf_model"):
        load_qwen35_vla_from_hf(hf_model)


def test_direct_loader_rejects_non_hf_symlink_before_snapshot_copy(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    external = tmp_path / "external.bin"
    external.write_bytes(b"untrusted")
    (hf_model / "chat_template.jinja").symlink_to(external)

    with pytest.raises(ValueError, match="rejects non-HF symlink"):
        load_qwen35_vla_from_hf(hf_model)


def test_direct_loader_rejects_unbound_model_entry_before_snapshot_copy(tmp_path: Path) -> None:
    hf_model = _write_imported_package(tmp_path)
    (hf_model / "unbound.bin").write_bytes(b"unbound")

    with pytest.raises(RuntimeError, match="unsupported entries.*unbound.bin"):
        load_qwen35_vla_from_hf(hf_model)


def test_import_provenance_requires_every_key_of_its_schema(tmp_path: Path) -> None:
    _write_imported_package(tmp_path)
    provenance_path = tmp_path / QWEN35_IMPORT_PROVENANCE_FILE
    provenance = json.loads(provenance_path.read_text())
    provenance.pop("dcp_identity_sha256")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="keys mismatch.*dcp_identity_sha256"):
        verify_qwen35_imported_package(tmp_path)


def _rewrite_provenance(root: Path, provenance: dict) -> None:
    (root / QWEN35_IMPORT_PROVENANCE_FILE).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


def _load_provenance(root: Path) -> dict:
    return json.loads((root / QWEN35_IMPORT_PROVENANCE_FILE).read_text())


def test_import_provenance_accepts_a_legacy_v2_record_without_a_conditioning_field(
    tmp_path: Path,
) -> None:
    _write_imported_package(tmp_path)
    provenance = _load_provenance(tmp_path)
    provenance["schema"] = QWEN35_IMPORT_PROVENANCE_SCHEMA_V2
    del provenance["conditioning_deployment"]
    _rewrite_provenance(tmp_path, provenance)

    verify_qwen35_imported_package(tmp_path)


def test_import_provenance_rejects_a_conditioning_field_in_a_v2_record(tmp_path: Path) -> None:
    _write_imported_package(tmp_path)
    provenance = _load_provenance(tmp_path)
    provenance["schema"] = QWEN35_IMPORT_PROVENANCE_SCHEMA_V2
    _rewrite_provenance(tmp_path, provenance)

    with pytest.raises(RuntimeError, match="keys mismatch.*conditioning_deployment"):
        verify_qwen35_imported_package(tmp_path)


def test_import_provenance_accepts_a_declared_conditioning_free_deployment(tmp_path: Path) -> None:
    _write_imported_package(tmp_path)
    provenance = _load_provenance(tmp_path)
    provenance["conditioning_deployment"] = {
        "renders_action_conditioning": False,
        "renders_mistake_conditioning": False,
    }
    _rewrite_provenance(tmp_path, provenance)

    verify_qwen35_imported_package(tmp_path)


def test_import_provenance_rejects_an_unknown_conditioning_key(tmp_path: Path) -> None:
    _write_imported_package(tmp_path)
    provenance = _load_provenance(tmp_path)
    provenance["conditioning_deployment"] = {
        "renders_action_conditioning": False,
        "renders_mistake_conditioning": False,
        "renders_scene_conditioning": False,
    }
    _rewrite_provenance(tmp_path, provenance)

    with pytest.raises(RuntimeError, match="keys mismatch.*renders_scene_conditioning"):
        verify_qwen35_imported_package(tmp_path)


def test_import_provenance_rejects_a_non_boolean_conditioning_flag(tmp_path: Path) -> None:
    _write_imported_package(tmp_path)
    provenance = _load_provenance(tmp_path)
    provenance["conditioning_deployment"] = {
        "renders_action_conditioning": "false",
        "renders_mistake_conditioning": False,
    }
    _rewrite_provenance(tmp_path, provenance)

    with pytest.raises(RuntimeError, match="renders_action_conditioning must be a boolean"):
        verify_qwen35_imported_package(tmp_path)


def test_import_provenance_rejects_an_unknown_schema_version(tmp_path: Path) -> None:
    _write_imported_package(tmp_path)
    provenance = _load_provenance(tmp_path)
    provenance["schema"] = "perceptron_isaac_import_provenance_v4"
    _rewrite_provenance(tmp_path, provenance)

    with pytest.raises(RuntimeError, match="Unsupported Qwen3.5 import provenance"):
        verify_qwen35_imported_package(tmp_path)


def test_conversion_provenance_binds_source_shard_digest(tmp_path: Path) -> None:
    _write_imported_package(tmp_path)
    provenance_path = tmp_path / QWEN35_IMPORT_PROVENANCE_FILE
    provenance = json.loads(provenance_path.read_text())
    shard_name = provenance["rmsnorm_conversion"]["shards"][0]["filename"]
    provenance["source_hashes"][shard_name] = "8" * 64
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="disagree on source shard"):
        verify_qwen35_imported_package(tmp_path)


@pytest.mark.parametrize(
    "audit_field",
    ["converted_key_count", "converted_keys_sha256", "tensor_count", "token_count"],
)
def test_conversion_audit_metadata_matches_checkpoint_contents(tmp_path: Path, audit_field: str) -> None:
    _write_imported_package(tmp_path)
    conversion = json.loads((tmp_path / "hf_model" / QWEN35_CONVERSION_PROVENANCE_FILE).read_text())
    records = {
        "converted_key_count": conversion,
        "converted_keys_sha256": conversion,
        "tensor_count": conversion["shards"][0],
        "token_count": conversion["qwen35_vocab"],
    }
    if audit_field.endswith("sha256"):
        records[audit_field][audit_field] = "9" * 64
    else:
        records[audit_field][audit_field] += 1
    _rewrite_conversion(tmp_path, conversion)

    with pytest.raises(RuntimeError, match="audit|tensor_count"):
        verify_qwen35_imported_package(tmp_path)


def test_direct_loader_rejects_noncanonical_model_directory(tmp_path: Path) -> None:
    _write_imported_package(tmp_path)
    rogue_model = tmp_path / "rogue_model"
    rogue_model.mkdir()

    with pytest.raises(RuntimeError, match="canonical package-local hf_model"):
        load_qwen35_vla_from_hf(rogue_model)
