#!/usr/bin/env python
"""Merge a Perceptron ISAAC training adapter into a portable BF16 deployment package."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from lerobot.policies.peft import _resolve_peft_base_snapshot
from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256, snapshot_directory
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import (
    NATIVE_STATS_EXPORT_FILENAME,
    SUITE_BY_TASK_INDEX_EXPORT_FILENAME,
    SUITE_STATS_EXPORT_FILENAME,
    PerceptronIsaacConfig,
)
from lerobot.utils.hub import hub_snapshot_revision

_RUNTIME_ASSETS = (
    "fast_processor",
    "isaac_deployment_adapter.json",
    NATIVE_STATS_EXPORT_FILENAME,
    "native_render_metadata.json",
    "policy_inference_recipe.json",
    "policy_normalization.json",
    "policy_state_contracts.json",
    "policy_state_identity.safetensors",
)
_PROCESSOR_PREFIXES = ("policy_preprocessor", "policy_postprocessor")
_PROCESSOR_PATH_FIELDS = {
    "config_toml_path",
    "fast_processor_path",
    "native_render_metadata_path",
    "native_stats_path",
    "stats_path",
    "suite_by_task_index_path",
    "suite_stats_path",
}
_SUITE_EXPORT_PATHS = {
    "suite_stats_path": SUITE_STATS_EXPORT_FILENAME,
    "suite_by_task_index_path": SUITE_BY_TASK_INDEX_EXPORT_FILENAME,
}


def _policy_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    for filename in ("config.json", "adapter_config.json", "adapter_model.safetensors"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"ISAAC training checkpoint is missing {filename}: {path}")
    return path


def _verify_bf16_model_shards(shards: list[Path]) -> None:
    """Reject an export containing any floating tensor that is not BF16."""
    nonfloating_dtypes = {"BOOL", "I8", "I16", "I32", "I64", "U8"}
    tensor_count = 0
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():  # noqa: SIM118 - safe_open is not iterable
                tensor_count += 1
                dtype = reader.get_slice(name).get_dtype()
                if dtype != "BF16" and dtype not in nonfloating_dtypes:
                    raise RuntimeError(
                        f"ISAAC deployment export tensor {name!r} in {shard.name} "
                        f"has dtype {dtype}, expected BF16."
                    )
    if tensor_count == 0:
        raise RuntimeError("ISAAC deployment export contains no model tensors.")


def _copy_item(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_runtime_assets(
    base_root: Path,
    policy_dir: Path,
    output: Path,
    *,
    native_stats_source: Path,
) -> None:
    for name in _RUNTIME_ASSETS:
        # Fine-tuning owns normalization, while immutable runtime contracts stay pinned to
        # the exact base revision used to load the weights.
        source = native_stats_source if name == NATIVE_STATS_EXPORT_FILENAME else base_root / name
        if source.exists():
            _copy_item(source, output / name)

    for source in policy_dir.iterdir():
        if any(source.name.startswith(prefix) for prefix in _PROCESSOR_PREFIXES):
            _copy_item(source, output / source.name)


def _copy_hf_assets(base_hf_dir: Path, output_hf_dir: Path) -> None:
    # The base MK1 import record authenticates the base shards and is stale after merging an
    # adapter. The deployment export has its own shard provenance below.
    excluded = {"config.json", "mk1_model_import.json", "rmsnorm_conversion.json"}
    for source in base_hf_dir.iterdir():
        if source.name in excluded or source.name.startswith("model"):
            continue
        destination = output_hf_dir / source.name
        if destination.exists():
            continue
        _copy_item(source, destination)

    # Keep the importer's known-good native Qwen config. The loader supplies the VLA expert config
    # from the LeRobot policy package, exactly as it does for the authenticated base checkpoint.
    shutil.copy2(base_hf_dir / "config.json", output_hf_dir / "config.json")


def _checkpoint_source_file(root: Path, raw_path: str | Path) -> Path:
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    path = Path(str(raw_path))
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"ISAAC checkpoint-local path must be package-relative: {path}")
    candidate = root / path
    if PerceptronIsaacPolicy._is_package_file(root.resolve(), candidate):
        return candidate
    try:
        candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return candidate
    raise RuntimeError(f"ISAAC checkpoint-local path is not a contained regular file: {candidate}")


def _native_stats_source(
    policy_config: PerceptronIsaacConfig,
    policy_dir: Path,
    base_root: Path,
) -> Path:
    adapter_stats = _checkpoint_source_file(policy_dir, NATIVE_STATS_EXPORT_FILENAME)
    if adapter_stats.is_file():
        return adapter_stats

    declared = getattr(policy_config, "native_stats_path", None)
    if declared and str(declared).strip():
        candidate = _checkpoint_source_file(policy_dir, declared)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"ISAAC fine-tune normalization stats declared by native_stats_path are missing: {candidate}"
        )

    base_stats = base_root / NATIVE_STATS_EXPORT_FILENAME
    if base_stats.is_file():
        return base_stats
    raise FileNotFoundError(
        "ISAAC normalization stats are missing from both the fine-tune checkpoint and its base."
    )


def _prepare_suite_tables(policy_config: PerceptronIsaacConfig, policy_dir: Path) -> bool:
    declared = {name: getattr(policy_config, name, None) for name in _SUITE_EXPORT_PATHS}
    if not any(str(value or "").strip() for value in declared.values()):
        for name in declared:
            setattr(policy_config, name, None)
        return False

    sources = {
        name: _checkpoint_source_file(policy_dir, value)
        for name, value in declared.items()
        if str(value or "").strip()
    }
    if len(sources) != len(declared) or not all(source.is_file() for source in sources.values()):
        logging.getLogger(__name__).warning(
            "ISAAC deployment export is clearing incomplete or unreadable per-suite routing tables: %s",
            {name: str(source) for name, source in sources.items()},
        )
        for name in declared:
            setattr(policy_config, name, None)
        return False

    for name, source in sources.items():
        setattr(policy_config, name, str(source.resolve()))
    return True


def _validate_processor_paths(package: Path) -> None:
    directory_fields = {"fast_processor_path"}

    def validate(value: Any, *, filename: str) -> None:
        if isinstance(value, list):
            for item in value:
                validate(item, filename=filename)
            return
        if not isinstance(value, dict):
            return
        if bool(value.get("suite_stats_path")) != bool(value.get("suite_by_task_index_path")):
            raise RuntimeError(f"ISAAC deployment export has an incomplete suite path pair in {filename}.")
        for key, item in value.items():
            if key in _PROCESSOR_PATH_FIELDS and item:
                path = Path(str(item))
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError(
                        f"ISAAC deployment export path {key} in {filename} must be package-relative: {path}"
                    )
                candidate = package / path
                expected_type_matches = candidate.is_dir() if key in directory_fields else candidate.is_file()
                if not expected_type_matches:
                    raise RuntimeError(
                        f"ISAAC deployment export path {key} in {filename} does not exist: {path}"
                    )
            validate(item, filename=filename)

    for prefix in _PROCESSOR_PREFIXES:
        config_path = package / f"{prefix}.json"
        validate(json.loads(config_path.read_text()), filename=config_path.name)


def _validate_export_package(package: Path) -> None:
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    config = PreTrainedConfig.from_pretrained(package)
    if not isinstance(config, PerceptronIsaacConfig):
        raise RuntimeError("ISAAC deployment export did not reload as a PerceptronIsaacConfig.")
    if bool(config.suite_stats_path) != bool(config.suite_by_task_index_path):
        raise RuntimeError("ISAAC deployment export has an incomplete config suite path pair.")
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, package)
    _validate_processor_paths(package)
    if PerceptronIsaacPolicy._config_declares_mk1(config):
        PerceptronIsaacPolicy._verify_packaged_contract_digests(config, package)
    else:
        # The staging directory is private and is the directory atomically published
        # below. Verify it in place instead of copying the multi-GB model into a
        # throwaway load snapshot whose bytes would never be published.
        PerceptronIsaacPolicy._verify_qwen35_trained_package_manifest(config, package)


def export_isaac_bf16_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    device: str = "cuda",
    max_shard_size: str = "5GB",
) -> Path:
    """Export from one stable adapter snapshot, keeping the source checkpoint untouched."""
    policy_dir = _policy_dir(checkpoint)
    adapter_entries = {path.name for path in policy_dir.iterdir() if path.name != "base_model"}
    try:
        adapter_snapshot = snapshot_directory(
            policy_dir,
            prefix="lerobot-isaac-adapter-export-",
            root_entries=adapter_entries,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"ISAAC adapter package contains an unsafe or not a contained regular file: {exc}"
        ) from exc
    with adapter_snapshot:
        return _export_isaac_bf16_checkpoint(
            adapter_snapshot.root,
            base_anchor=policy_dir,
            output=output,
            device=device,
            max_shard_size=max_shard_size,
        )


def _export_isaac_bf16_checkpoint(
    policy_dir: Path,
    *,
    base_anchor: Path,
    output: Path,
    device: str,
    max_shard_size: str,
) -> Path:
    """Merge a stable adapter with one resolved base into an atomic BF16 package."""
    from peft import PeftConfig

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite deployment export: {output}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA export requested but no CUDA device is available; run this command via srun."
        )

    policy_config = PreTrainedConfig.from_pretrained(policy_dir)
    if not isinstance(policy_config, PerceptronIsaacConfig):
        raise TypeError(f"Expected a perceptron_isaac checkpoint, got {type(policy_config).__name__}.")
    is_mk1 = PerceptronIsaacPolicy._config_declares_mk1(policy_config)
    source_mk1_model_import_sha256 = (
        policy_config.mk1_model_import_sha256 or policy_config.mk1_source_model_import_sha256
    )
    source_mk1_trained_package_manifest_sha256 = policy_config.mk1_trained_package_manifest_sha256
    if is_mk1 and source_mk1_model_import_sha256 is None:
        raise RuntimeError("Cannot export an MK1 package without source model-import identity.")
    adapter_config = PeftConfig.from_pretrained(policy_dir)
    base_reference = adapter_config.base_model_name_or_path
    if not base_reference:
        raise ValueError("ISAAC adapter_config.json does not identify its base checkpoint.")
    base_revision = getattr(adapter_config, "revision", None)
    base_root, base_is_local = _resolve_peft_base_snapshot(
        base_anchor,
        base_reference,
        revision=base_revision,
        adapter_is_local=True,
    )
    source_base_revision = base_revision if base_is_local else hub_snapshot_revision(base_root)
    try:
        base_snapshot = snapshot_directory(base_root, prefix="lerobot-isaac-base-export-")
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"ISAAC base package cannot be snapshotted safely: {exc}") from exc
    with base_snapshot:
        return _export_isaac_bf16_from_snapshots(
            policy_dir,
            base_root=base_snapshot.root,
            base_reference=str(base_reference),
            source_base_revision=source_base_revision,
            adapter_config=adapter_config,
            policy_config=policy_config,
            is_mk1=is_mk1,
            source_mk1_model_import_sha256=source_mk1_model_import_sha256,
            source_mk1_trained_package_manifest_sha256=source_mk1_trained_package_manifest_sha256,
            output=output,
            device=device,
            max_shard_size=max_shard_size,
        )


def _export_isaac_bf16_from_snapshots(
    policy_dir: Path,
    *,
    base_root: Path,
    base_reference: str,
    source_base_revision: str | None,
    adapter_config: Any,
    policy_config: PerceptronIsaacConfig,
    is_mk1: bool,
    source_mk1_model_import_sha256: str | None,
    source_mk1_trained_package_manifest_sha256: str | None,
    output: Path,
    device: str,
    max_shard_size: str,
) -> Path:
    """Merge and publish using only private adapter and base snapshots."""
    from peft import PeftModel

    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
    from lerobot.policies.perceptron_isaac.trained_package import (
        finalize_canonical_trained_package,
        rewrite_trained_processor_paths,
    )

    policy_config.device = device
    policy_config.pretrained_path = str(policy_dir)
    policy_config.train_storage_fp32 = True
    native_stats_source = _native_stats_source(policy_config, policy_dir, base_root)
    policy_config.native_stats_path = str(native_stats_source.resolve())
    policy_config.stats_path = None
    policy_config._allow_trusted_absolute_checkpoint_paths = True
    _prepare_suite_tables(policy_config, policy_dir)
    base_policy = PerceptronIsaacPolicy.from_pretrained(
        str(base_root),
        config=policy_config,
        revision=None,
    )
    adapted = PeftModel.from_pretrained(
        base_policy,
        str(policy_dir),
        config=adapter_config,
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    merged = adapted.merge_and_unload(safe_merge=True)
    if merged._isaac_model is None:
        raise RuntimeError("ISAAC PEFT merge did not materialize the native backbone.")
    merged._isaac_model.to(dtype=torch.bfloat16)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        output_hf_dir = temporary / "hf_model"
        merged._isaac_model.save_pretrained(
            output_hf_dir,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        _copy_hf_assets(base_root / "hf_model", output_hf_dir)
        if is_mk1:
            from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
                MK1_TEST_GEOMETRY_MARKER,
                Mk1CheckpointContractError,
                finalize_mk1_runtime_checkpoint_layout,
                validate_mk1_checkpoint,
            )

            finalize_mk1_runtime_checkpoint_layout(output_hf_dir)
            try:
                inner_config = json.loads((output_hf_dir / "config.json").read_text())
                validate_mk1_checkpoint(
                    output_hf_dir,
                    allow_test_only_reduced_geometry=bool(inner_config.get(MK1_TEST_GEOMETRY_MARKER, False)),
                )
            except (json.JSONDecodeError, Mk1CheckpointContractError) as exc:
                raise RuntimeError(f"ISAAC MK1 deployment export is invalid: {exc}") from exc
        else:
            from lerobot.policies.perceptron_isaac.qwen35_checkpoint import (
                write_qwen35_weight_convention,
            )

            write_qwen35_weight_convention(output_hf_dir)
        _copy_runtime_assets(
            base_root,
            policy_dir,
            temporary,
            native_stats_source=native_stats_source,
        )

        deployment_config = copy.deepcopy(policy_config)
        deployment_config.device = "cpu"
        deployment_config.pretrained_path = None
        deployment_config.pretrained_revision = None
        deployment_config.use_peft = False
        deployment_config.push_to_hub = False
        deployment_config.hf_model_path = "hf_model"
        deployment_config.train_storage_fp32 = False
        deployment_config.apply_offset_norm = False
        deployment_config.native_render_metadata_path = "native_render_metadata.json"
        deployment_config.native_stats_path = NATIVE_STATS_EXPORT_FILENAME
        deployment_config.fast_processor_path = "fast_processor"
        deployment_config.config_toml_path = None
        deployment_config.stats_path = None
        deployment_config.per_suite_stats_dir = None
        # Base manifests authenticate the unmerged base and cannot describe this new model tree.
        # The canonical outer manifest written after every output asset binds the merged package.
        deployment_config.mk1_model_import_sha256 = None
        deployment_config.mk1_source_model_import_sha256 = source_mk1_model_import_sha256 if is_mk1 else None
        deployment_config.mk1_trained_package_manifest_sha256 = None
        deployment_config.qwen35_trained_package_manifest_sha256 = None
        deployment_config._save_pretrained(temporary)
        rewrite_trained_processor_paths(temporary)

        shard_paths = sorted(output_hf_dir.glob("model*.safetensors"))
        if not shard_paths:
            raise RuntimeError("ISAAC deployment export wrote no safetensors model shards.")
        _verify_bf16_model_shards(shard_paths)
        shards = []
        shard_digests: dict[str, str] = {}
        for shard in shard_paths:
            digest = file_sha256(shard)
            shards.append({"filename": shard.name, "sha256": digest})
            shard_digests[f"hf_model/{shard.name}"] = digest
        provenance = {
            "schema": "perceptron_isaac_bf16_deployment_export_v1",
            "storage_dtype": "bfloat16",
            "source_adapter_sha256": file_sha256(policy_dir / "adapter_model.safetensors"),
            "source_base": str(base_reference),
            "source_base_revision": source_base_revision,
            "source_mk1_model_import_sha256": source_mk1_model_import_sha256,
            "source_mk1_trained_package_manifest_sha256": (source_mk1_trained_package_manifest_sha256),
            "model_shards": shards,
        }
        (temporary / "isaac_export_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        finalize_canonical_trained_package(
            deployment_config,
            temporary,
            family="mk1" if is_mk1 else "qwen35",
            mk1_source_model_import_sha256=(source_mk1_model_import_sha256 if is_mk1 else None),
            precomputed_files_sha256=shard_digests,
        )
        _validate_export_package(temporary)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a Perceptron ISAAC LoRA training checkpoint into a separate, portable "
            "BF16 inference package. The source FP32 training checkpoint is never modified."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint root or its pretrained_model directory.",
    )
    parser.add_argument("output", type=Path, help="New output directory; it must not already exist.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Merge device (default: cuda). Run GPU export through srun on the cluster.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum output safetensors shard size.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = export_isaac_bf16_checkpoint(
        args.checkpoint,
        args.output,
        device=args.device,
        max_shard_size=args.max_shard_size,
    )
    print(f"Wrote portable BF16 ISAAC deployment package: {output}")


if __name__ == "__main__":
    main()
