# pyright: reportMissingImports=false
# This test runs only under the remote host's interpreter
# (/home/phil/isaac-ft26-stage/venv with PYTHONPATH=/home/phil/isaac-ft26-stage/lerobot/src),
# where every ``lerobot`` import below resolves. The authoring machine has no lerobot
# install, so a local import check cannot see them.
"""Save contracts for a checkpoint written from a raw Isaac-0.5 export.

A raw export nests the LeRobot policy package in ``lerobot_policy/`` and points at the
model with ``hf_model_path=".."``. ``_verified_checkpoint_source`` snapshots its source
package before copying it, and a snapshot of the package ALONE leaves that parent behind,
so ``_resolve_checkpoint_local_paths`` calls the ".." an escape and the save dies before
writing anything. The guard is right; the snapshot is what has to change.

These tests pin both halves of the fix and the guard that must survive it:
  * a genuine escape is still refused (no portable parent, and a sideways "../elsewhere");
  * the save source keeps the export root its ".." resolves against;
  * the written checkpoint embeds the export assets as ``hf_model/`` and records the
    relative-and-downward names ``checkpoint_import`` already writes
    (``hf_model_path="hf_model"``), which is also where ``_save_canonical_inner_weights``
    puts its trained tensors;
  * ``is_portable_isaac05_repository`` means a raw EXPORT, not any Isaac-0.5 model
    directory, so a checkpoint's embedded ``hf_model/`` is not read as F32 export storage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256, snapshot_directory
from lerobot.policies.perceptron_isaac.mharmony_native import read_recipe_reserved_token_groups
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContractError,
    finalize_mk1_runtime_checkpoint_layout,
)
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import (
    NATIVE_RECIPE_EXPORT_FILENAME,
    PerceptronIsaacConfig,
    is_portable_isaac05_repository,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import (
    PerceptronIsaacPolicy,
    _package_storage_dtypes,
)

PACKAGE_DIRECTORY_NAME = "lerobot_policy"

# The shipped Isaac-0.5 export declares its FAST pool at an explicit offset and names the
# coordinate block "coord"; these are the values in the base export's
# policy_inference_recipe.json. DEFAULT_MHARMONY_RESERVED_TOKEN_GROUPS has neither, which is
# precisely the fallback a written checkpoint must never reach.
EXPORT_GROUPS = (
    {"name": None, "offset": 249321, "size": 2048, "tokenizer": "physical-intelligence/fast"},
    {"name": "coord", "offset": 248320, "size": 1001, "tokenizer": None},
)
COORD_OFFSET = 248320
COORD_SIZE = 1001


def _write_raw_export(root: Path) -> Path:
    """Build the on-disk shape of a raw Isaac-0.5 export and return its policy package."""
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "isaac_0_5", "architectures": ["Isaac05ForConditionalGeneration"]})
        + "\n",
        encoding="utf-8",
    )
    (root / "modeling_isaac05.py").write_text("# model code\n", encoding="utf-8")
    (root / NATIVE_RECIPE_EXPORT_FILENAME).write_text(
        json.dumps(
            {
                "recipes": [
                    {"rendering": {"reserved_token_groups": [dict(group) for group in EXPORT_GROUPS]}}
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}) + "\n")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"stale-shard-one")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"stale-shard-two")
    fast_processor = root / "fast_processor_pinned"
    fast_processor.mkdir()
    (fast_processor / "tokenizer.json").write_text("{}\n", encoding="utf-8")
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
        + "\n",
        encoding="utf-8",
    )
    return package


# The config rejects a fast processor path without its pinned tree digest. These tests are
# about path resolution, not artifact pinning, so the digest is a well-formed placeholder.
PLACEHOLDER_TREE_SHA256 = "0" * 64


def _config(**overrides: object) -> PerceptronIsaacConfig:
    if "fast_processor_path" in overrides:
        overrides.setdefault("fast_processor_tree_sha256", PLACEHOLDER_TREE_SHA256)
    return PerceptronIsaacConfig(device="cpu", **overrides)


def test_resolve_refuses_dotdot_when_the_parent_is_not_an_export(tmp_path: Path) -> None:
    """The escape hatch is the export, not the "..": a bare parent is still refused."""
    package = tmp_path / "bare" / PACKAGE_DIRECTORY_NAME
    package.mkdir(parents=True)
    config = _config(hf_model_path="..")

    with pytest.raises(RuntimeError, match="escapes the package root"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)


def test_resolve_refuses_a_sideways_escape_out_of_an_export(tmp_path: Path) -> None:
    """Only the two documented export relatives are accepted; a neighbour is not."""
    package = _write_raw_export(tmp_path / "export")
    (tmp_path / "export" / "elsewhere").mkdir()
    config = _config(hf_model_path="../elsewhere")

    with pytest.raises(RuntimeError, match="escapes the package root"):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)


def test_package_only_snapshot_is_what_breaks_the_export_relative(tmp_path: Path) -> None:
    """Characterise the defect: copying the package alone strands its "..".

    This is the RED half. It documents why ``_snapshot_save_source`` exists, so a later
    reader does not "simplify" it back to a plain ``snapshot_directory(source_root)``.
    """
    package = _write_raw_export(tmp_path / "export")
    config = _config(hf_model_path="..")

    with (
        snapshot_directory(package, prefix="test-package-only-") as snapshot,
        pytest.raises(RuntimeError, match="escapes the package root"),
    ):
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, snapshot.root)


def test_snapshot_save_source_keeps_the_export_root(tmp_path: Path) -> None:
    """The yielded package still has its export above it, so ".." resolves inside the snapshot."""
    package = _write_raw_export(tmp_path / "export")
    config = _config(hf_model_path="..", fast_processor_path="../fast_processor_pinned")

    with PerceptronIsaacPolicy._snapshot_save_source(package, prefix="test-export-") as package_root:
        assert package_root.name == PACKAGE_DIRECTORY_NAME
        assert is_portable_isaac05_repository(package_root.parent)
        PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package_root)
        assert Path(config.hf_model_path) == package_root.parent
        assert Path(config.fast_processor_path) == package_root.parent / "fast_processor_pinned"
        # The save replaces both weight inventories, so the snapshot never carries them.
        assert not list(package_root.parent.glob("model-*.safetensors"))
        assert not (package_root.parent / "model.safetensors.index.json").exists()


def test_snapshot_save_source_leaves_a_self_contained_package_alone(tmp_path: Path) -> None:
    """An importer-written package has no export above it and takes the unchanged path."""
    package = tmp_path / "imported"
    package.mkdir()
    (package / "config.json").write_text(json.dumps({"type": "perceptron_isaac"}) + "\n")
    (package / "hf_model").mkdir()

    with PerceptronIsaacPolicy._snapshot_save_source(package, prefix="test-imported-") as package_root:
        assert package_root.name == "snapshot"
        assert (package_root / "hf_model").is_dir()


def test_materialized_checkpoint_records_relative_downward_names(tmp_path: Path) -> None:
    """The written checkpoint embeds the export assets and records them checkpoint-local."""
    package = _write_raw_export(tmp_path / "export")
    destination = tmp_path / "checkpoint"
    destination.mkdir()

    overrides = PerceptronIsaacPolicy._materialize_portable_export_assets(package, destination)

    # ``fast_processor`` is the canonical written-package name in PROCESSOR_PACKAGE_PATHS,
    # and the one the serialized processor rewrite looks for at the package root.
    assert overrides == {"hf_model_path": "hf_model", "fast_processor_path": "fast_processor"}
    assert (destination / "fast_processor" / "tokenizer.json").is_file()
    model_root = destination / "hf_model"
    assert (model_root / "config.json").is_file()
    assert (model_root / "modeling_isaac05.py").is_file()
    assert (model_root / "fast_processor_pinned" / "tokenizer.json").is_file()
    # The base weights are the stale inventory the save is about to replace.
    assert not list(model_root.glob("model-*.safetensors"))
    assert not (model_root / "model.safetensors.index.json").exists()
    # The policy package is not nested inside its own embedded model directory.
    assert not (model_root / PACKAGE_DIRECTORY_NAME).exists()
    # Nothing the checkpoint records leaves the checkpoint.
    for relative in overrides.values():
        assert ".." not in Path(relative).parts
        assert (destination / relative).is_dir()


def test_materialized_names_resolve_without_entering_the_escape_branch(tmp_path: Path) -> None:
    """A checkpoint written this way reloads through the guard unchanged."""
    package = _write_raw_export(tmp_path / "export")
    destination = tmp_path / "checkpoint"
    destination.mkdir()
    overrides = PerceptronIsaacPolicy._materialize_portable_export_assets(package, destination)
    config = _config(**overrides)

    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, destination)

    assert Path(config.hf_model_path) == destination / "hf_model"
    assert Path(config.fast_processor_path) == destination / "fast_processor"


def test_materialize_is_a_no_op_for_a_self_contained_source(tmp_path: Path) -> None:
    package = tmp_path / "imported"
    package.mkdir()
    (package / "config.json").write_text(json.dumps({"type": "perceptron_isaac"}) + "\n")
    destination = tmp_path / "checkpoint"
    destination.mkdir()

    assert PerceptronIsaacPolicy._materialize_portable_export_assets(package, destination) == {}
    assert not (destination / "hf_model").exists()


def test_embedded_model_assets_keep_the_export_storage_contract(tmp_path: Path) -> None:
    """The embedded ``hf_model/`` inherits the export's F32 storage contract.

    Measured on the real save: the written inventory is 1616 F32 tensors, because FSDP's
    full-state-dict gather upcasts. So the embedded assets must answer the storage question
    the same way the export does, and the save must validate them that way too.
    """
    export = tmp_path / "export"
    package = _write_raw_export(export)
    assert _package_storage_dtypes(export) == frozenset({"F32"})

    destination = tmp_path / "checkpoint"
    destination.mkdir()
    PerceptronIsaacPolicy._materialize_portable_export_assets(package, destination)

    assert _package_storage_dtypes(destination / "hf_model") == frozenset({"F32"})
    # The checkpoint root is a policy package, not model assets, so it is not F32 storage.
    assert _package_storage_dtypes(destination) == frozenset({"BF16"})


def test_plain_mk1_model_directory_is_bf16_storage(tmp_path: Path) -> None:
    """Only Isaac-0.5 model assets get the F32 contract; everything else stays BF16."""
    model_root = tmp_path / "hf_model"
    model_root.mkdir()
    (model_root / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}) + "\n")

    assert _package_storage_dtypes(model_root) == frozenset({"BF16"})


def _write_runtime_model_dir(root: Path, vla_namespace: str) -> Path:
    """Write a saved model directory whose VLA block uses the given namespace."""
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "isaac_0_5",
                vla_namespace: {"mtp": {"present": False, "physical_layers": 0, "rollout_steps": 0}},
                "text_config": {"mtp_num_hidden_layers": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}) + "\n")
    return root


@pytest.mark.parametrize("vla_namespace", ["genesis_vla", "isaac05_vla"])
def test_finalize_reads_the_vla_namespace_the_saved_config_uses(
    tmp_path: Path, vla_namespace: str
) -> None:
    """A raw export keeps ``isaac05_vla`` on disk; the renamed form is in-memory only.

    ``_normalize_isaac05`` returns ``genesis_vla``, but that dict does not re-parse as an
    on-disk config, so an embedded export config legitimately still says ``isaac05_vla``.
    """
    model_root = _write_runtime_model_dir(tmp_path / vla_namespace, vla_namespace)

    finalize_mk1_runtime_checkpoint_layout(model_root)


def test_finalize_still_refuses_a_config_with_no_vla_block(tmp_path: Path) -> None:
    """Reading two namespaces must not become reading none."""
    model_root = tmp_path / "no_vla"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps({"model_type": "isaac_0_5", "text_config": {}}) + "\n", encoding="utf-8"
    )

    with pytest.raises(Mk1CheckpointContractError, match="missing its MTP contract"):
        finalize_mk1_runtime_checkpoint_layout(model_root)


def test_written_checkpoint_resolves_the_export_reserved_token_groups(tmp_path: Path) -> None:
    """DEFECT 6: the checkpoint resolves the export's coord block, not the coord-less default.

    ``checkpoint_asset_roots`` looks at the package root and a portable parent only, and a
    written checkpoint has neither above it. Embedding the export under ``hf_model/`` left
    ``policy_inference_recipe.json`` somewhere no load would look, so
    ``load_native_render_metadata`` fell back to ``DEFAULT_MHARMONY_RESERVED_TOKEN_GROUPS`` and
    ``Mk1CheckpointContract.validate_coord_reserved_token_groups`` refused the package the save
    had just written: "reserved-token groups must contain exactly one name='coord' entry".
    """
    package = _write_raw_export(tmp_path / "export")
    destination = tmp_path / "checkpoint"
    destination.mkdir()
    overrides = PerceptronIsaacPolicy._materialize_portable_export_assets(package, destination)
    config = _config(**overrides, pretrained_path=destination)

    recipe_path = config.resolve_native_recipe_path()

    assert recipe_path == str(destination / NATIVE_RECIPE_EXPORT_FILENAME)
    groups = read_recipe_reserved_token_groups(recipe_path)
    coord_groups = [group for group in groups if group.get("name") == "coord"]
    assert len(coord_groups) == 1
    assert (coord_groups[0]["offset"], coord_groups[0]["size"]) == (COORD_OFFSET, COORD_SIZE)


def test_written_checkpoint_recipe_is_resolved_from_the_package_it_moves_with(
    tmp_path: Path,
) -> None:
    """The recipe is found relative to wherever the checkpoint currently lives."""
    package = _write_raw_export(tmp_path / "export")
    destination = tmp_path / "checkpoint"
    destination.mkdir()
    overrides = PerceptronIsaacPolicy._materialize_portable_export_assets(package, destination)
    moved = tmp_path / "relocated" / "ckpt"
    moved.parent.mkdir()
    destination.rename(moved)
    config = _config(**overrides, pretrained_path=moved)

    assert config.resolve_native_recipe_path() == str(moved / NATIVE_RECIPE_EXPORT_FILENAME)


def _portable_checkpoint(root: Path) -> Path:
    """Write the non-weight shape of a checkpoint saved from a raw Isaac-0.5 export."""
    model_root = root / "hf_model"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text(
        json.dumps({"model_type": "isaac_0_5"}) + "\n", encoding="utf-8"
    )
    adapter = root / "isaac_deployment_adapter.json"
    adapter.write_text(json.dumps({"schema": "isaac_deployment_adapter_v1"}) + "\n", encoding="utf-8")
    return adapter


def _unstarted_policy(config: PerceptronIsaacConfig, save_root: Path) -> PerceptronIsaacPolicy:
    """A policy object for the finalize path alone: no weights, no backbone, no CUDA.

    ``finalize_pretrained_package`` reads only the config and the recorded save root, so
    building the 35.7 B-parameter module to exercise it would test the loader, not the save.
    """
    policy = PerceptronIsaacPolicy.__new__(PerceptronIsaacPolicy)
    policy.config = config
    policy._canonical_inner_save_root = str(save_root.resolve())
    return policy


def test_finalize_closes_a_portable_export_package_on_its_adapter_identity(tmp_path: Path) -> None:
    """DEFECT 5: a save from a raw export finalizes instead of dying on a missing identity.

    The export carries no ``mk1_model_import.json``, so ``mk1_source_model_import_sha256`` is
    None and ``finalize_canonical_trained_package(family="mk1")`` raised "Cannot finalize an MK1
    package without source model-import identity" -- after the 133 GiB package was already on
    disk, leaving it incomplete. ``_verify_packaged_contract_digests`` authenticates this package
    shape by its deployment adapter, which is the record finalize verifies here.
    """
    checkpoint = tmp_path / "checkpoint"
    adapter = _portable_checkpoint(checkpoint)
    config = _config(
        hf_model_path=str(checkpoint / "hf_model"),
        deployment_adapter_sha256=file_sha256(adapter),
    )
    policy = _unstarted_policy(config, checkpoint)

    policy.finalize_pretrained_package(checkpoint)

    # No canonical manifest is written, because the package has no identity to bind one to.
    assert not (checkpoint / "mk1_trained_package_manifest.json").exists()


def test_finalize_refuses_a_portable_package_whose_adapter_digest_disagrees(tmp_path: Path) -> None:
    """The adapter record is verified, not merely assumed present."""
    checkpoint = tmp_path / "checkpoint"
    adapter = _portable_checkpoint(checkpoint)
    config = _config(hf_model_path=str(checkpoint / "hf_model"), deployment_adapter_sha256="a" * 64)
    policy = _unstarted_policy(config, checkpoint)
    assert file_sha256(adapter) != "a" * 64

    with pytest.raises(RuntimeError, match="deployment_adapter_sha256 mismatch"):
        policy.finalize_pretrained_package(checkpoint)


def test_finalize_refuses_a_portable_package_with_no_deployment_adapter(tmp_path: Path) -> None:
    """A missing adapter is a missing identity, not a package to publish."""
    checkpoint = tmp_path / "checkpoint"
    adapter = _portable_checkpoint(checkpoint)
    digest = file_sha256(adapter)
    adapter.unlink()
    config = _config(hf_model_path=str(checkpoint / "hf_model"), deployment_adapter_sha256=digest)
    policy = _unstarted_policy(config, checkpoint)

    with pytest.raises(RuntimeError, match="missing its deployment adapter"):
        policy.finalize_pretrained_package(checkpoint)


def test_finalize_still_refuses_an_mk1_package_that_is_not_a_portable_export(tmp_path: Path) -> None:
    """The identity requirement survives the fix for every non-portable MK1 package."""
    checkpoint = tmp_path / "checkpoint"
    model_root = checkpoint / "hf_model"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "genesis_vla": {"backbone_family": "mk1_qwen3_6_moe"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = _config(hf_model_path=str(model_root))
    assert PerceptronIsaacPolicy._config_declares_mk1(config)
    policy = _unstarted_policy(config, checkpoint)

    with pytest.raises(RuntimeError, match="without source model-import identity"):
        policy.finalize_pretrained_package(checkpoint)
