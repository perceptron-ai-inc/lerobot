"""Raw exports remain loadable, but saving their advanced layout is deferred.

Keep containment, snapshot ownership and native F32 storage contracts intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import (
    PerceptronIsaacConfig,
    is_portable_isaac05_repository,
)
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContractError,
    finalize_mk1_runtime_checkpoint_layout,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import (
    PerceptronIsaacPolicy,
    _package_storage_dtypes,
)

PACKAGE_DIRECTORY_NAME = "lerobot_policy"


def _write_raw_export(root: Path) -> Path:
    """Build inert raw-export assets and return their nested policy package."""
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type": "isaac_0_5"}\n')
    (root / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
    (root / "model-00001-of-00002.safetensors").write_bytes(b"stale-shard-one")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"stale-shard-two")
    fast_processor = root / "fast_processor_pinned"
    fast_processor.mkdir()
    (fast_processor / "tokenizer.json").write_text("{}\n")
    package = root / PACKAGE_DIRECTORY_NAME
    package.mkdir()
    (package / "config.json").write_text(
        json.dumps(
            {
                "type": "perceptron_isaac",
                "hf_model_path": "..",
                "fast_processor_path": "../fast_processor_pinned",
            }
        )
        + "\n"
    )
    return package


def _config(**overrides: object) -> PerceptronIsaacConfig:
    if "fast_processor_path" in overrides:
        overrides.setdefault("fast_processor_tree_sha256", "0" * 64)
    return PerceptronIsaacConfig(device="cpu", **overrides)


def _unstarted_policy(config: PerceptronIsaacConfig, save_root: Path) -> PerceptronIsaacPolicy:
    """Exercise save preflight without allocating a backbone or connecting CUDA."""
    policy = PerceptronIsaacPolicy.__new__(PerceptronIsaacPolicy)
    policy.config = config
    policy._canonical_inner_save_root = str(save_root.resolve())
    return policy


def _portable_checkpoint(root: Path) -> Path:
    model_root = root / "hf_model"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model_type": "isaac_0_5"}\n')
    adapter = root / "isaac_deployment_adapter.json"
    adapter.write_text('{"schema": "isaac_deployment_adapter_v1"}\n')
    return adapter


@pytest.mark.parametrize("entry_point", ["save", "finalize"])
@pytest.mark.parametrize("layout", ["raw", "portable"])
def test_raw_export_saving_is_rejected_before_collectives_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_point: str, layout: str
) -> None:
    import lerobot.distributed.checkpoint as checkpoint

    package = _write_raw_export(tmp_path / "export")
    config = _config(hf_model_path="..")
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)
    destination = tmp_path / "saved"
    if entry_point == "finalize" or layout == "portable":
        adapter = _portable_checkpoint(destination)
        config.deployment_adapter_sha256 = file_sha256(adapter)
    if layout == "portable":
        config.hf_model_path = str(destination / "hf_model")
        config._checkpoint_source_root = str(destination)
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    policy = _unstarted_policy(config, destination)

    def forbidden_gather(*args: object, **kwargs: object) -> None:
        pytest.fail("Unsupported raw export save reached collective state gathering")

    monkeypatch.setattr(checkpoint, "full_model_state_dict", forbidden_gather)
    with pytest.raises(RuntimeError, match="Raw/portable-export saving is unsupported"):
        if entry_point == "save":
            policy._save_pretrained(destination)
        else:
            policy.finalize_pretrained_package(destination)
    after = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    assert after == before


def test_resolve_refuses_dotdot_when_the_parent_is_not_an_export(tmp_path: Path) -> None:
    package = tmp_path / "bare" / PACKAGE_DIRECTORY_NAME
    package.mkdir(parents=True)
    config = _config(hf_model_path="..")
    with pytest.raises(RuntimeError, match="escapes the package root"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)


def test_resolve_refuses_a_sideways_escape_out_of_an_export(tmp_path: Path) -> None:
    package = _write_raw_export(tmp_path / "export")
    (tmp_path / "export" / "elsewhere").mkdir()
    config = _config(hf_model_path="../elsewhere")
    with pytest.raises(RuntimeError, match="escapes the package root"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)


def test_snapshot_save_source_keeps_the_export_root(tmp_path: Path) -> None:
    package = _write_raw_export(tmp_path / "export")
    config = _config(hf_model_path="..", fast_processor_path="../fast_processor_pinned")
    with PerceptronIsaacPolicy._snapshot_save_source(package, prefix="test-export-") as package_root:
        assert package_root.name == PACKAGE_DIRECTORY_NAME
        assert is_portable_isaac05_repository(package_root.parent)
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package_root)
        assert Path(config.hf_model_path) == package_root.parent
        assert Path(config.fast_processor_path) == package_root.parent / "fast_processor_pinned"
        assert not list(package_root.parent.glob("model-*.safetensors"))
        assert not (package_root.parent / "model.safetensors.index.json").exists()


def test_snapshot_save_source_leaves_a_self_contained_package_alone(tmp_path: Path) -> None:
    package = tmp_path / "imported"
    package.mkdir()
    (package / "config.json").write_text('{"type": "perceptron_isaac"}\n')
    (package / "hf_model").mkdir()
    with PerceptronIsaacPolicy._snapshot_save_source(package, prefix="test-imported-") as package_root:
        assert package_root.name == "snapshot"
        assert (package_root / "hf_model").is_dir()


def test_embedded_model_assets_keep_the_export_storage_contract(tmp_path: Path) -> None:
    export = tmp_path / "export"
    _write_raw_export(export)
    assert _package_storage_dtypes(export) == frozenset({"F32"})
    checkpoint = tmp_path / "checkpoint"
    _portable_checkpoint(checkpoint)
    assert _package_storage_dtypes(checkpoint / "hf_model") == frozenset({"F32"})
    assert _package_storage_dtypes(checkpoint) == frozenset({"BF16"})


def test_plain_mk1_model_directory_is_bf16_storage(tmp_path: Path) -> None:
    model_root = tmp_path / "hf_model"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"model_type": "qwen3_5_moe"}\n')
    assert _package_storage_dtypes(model_root) == frozenset({"BF16"})


@pytest.mark.parametrize("vla_namespace", ["genesis_vla", "isaac05_vla"])
def test_finalize_reads_the_vla_namespace_the_saved_config_uses(tmp_path: Path, vla_namespace: str) -> None:
    model_root = tmp_path / vla_namespace
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "isaac_0_5",
                vla_namespace: {"mtp": {"present": False, "physical_layers": 0, "rollout_steps": 0}},
                "text_config": {"mtp_num_hidden_layers": 0},
            }
        )
    )
    (model_root / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
    finalize_mk1_runtime_checkpoint_layout(model_root)


def test_finalize_still_refuses_a_config_with_no_vla_block(tmp_path: Path) -> None:
    model_root = tmp_path / "no_vla"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"model_type": "isaac_0_5", "text_config": {}}\n')
    with pytest.raises(Mk1CheckpointContractError, match="missing its MTP contract"):
        finalize_mk1_runtime_checkpoint_layout(model_root)


def test_finalize_still_refuses_an_mk1_package_without_source_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    model_root = checkpoint / "hf_model"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "genesis_vla": {"backbone_family": "mk1_qwen3_6_moe"}})
    )
    config = _config(hf_model_path=str(model_root))
    policy = _unstarted_policy(config, checkpoint)
    with pytest.raises(RuntimeError, match="without source model-import identity"):
        policy.finalize_pretrained_package(checkpoint)
