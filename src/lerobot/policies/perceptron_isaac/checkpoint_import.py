"""Authenticated import primitives for Genesis ISAAC checkpoints.

This module deliberately has no Genesis imports.  It validates the canonical
JSON digests used by Genesis and authenticates them against the fixed-width
``policy_state_identity`` tensor exported as a safetensors sidecar or stored
inside the source DCP checkpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from .checkpoint_integrity import (
    DirectorySnapshot,
    canonical_json_sha256 as canonical_sha256,
    file_sha256,
    load_json_object as load_integrity_json_object,
    require_sha256 as require_integrity_sha256,
    snapshot_directory,
)
from .fast_processor import (
    DEFAULT_FAST_PROCESSOR_REPOSITORY,
    DEFAULT_FAST_PROCESSOR_REVISION,
    DEFAULT_FAST_PROCESSOR_TREE_SHA256,
    materialize_fast_processor_artifact,
    materialize_pinned_fast_processor_snapshot,
    resolve_pinned_fast_processor_snapshot,
)
from .isaac_stats import NATIVE_NORMALIZATION_EPS
from .mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    Mk1CheckpointContractError,
    read_mk1_config,
    validate_mk1_checkpoint,
    validate_mk1_tokenizer,
)
from .qwen35_checkpoint import (
    HF_RUNTIME_ASSET_FILENAMES,
    QWEN35_BASE_VOCAB_SIZE,
    QWEN35_CONVERSION_ALGORITHM,
    QWEN35_CONVERSION_DTYPE,
    QWEN35_CONVERSION_PROVENANCE_SCHEMA,
    QWEN35_IMPORT_PROVENANCE_SCHEMA,
    QWEN35_VOCAB_ALGORITHM,
    QWEN35_VOCAB_SCHEMA,
    is_qwen35_offset_norm_key,
    safetensors_checkpoint_layout,
)
from .trained_package import (
    AUTHENTICATED_IMPORT_MODEL_STORAGE,
    MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
    MK1_TRAINED_PACKAGE_MANIFEST_FILENAME,
    MK1_TRAINED_PACKAGE_MANIFEST_SCHEMA,
    QWEN35_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
    QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME,
    QWEN35_TRAINED_PACKAGE_MANIFEST_SCHEMA,
    finalize_trained_package_config,
    write_trained_package_manifest,
)

POLICY_STATE_CONTRACT_FILENAME = "policy_state_contracts.json"
POLICY_NORMALIZATION_FILENAME = "policy_normalization.json"
POLICY_INFERENCE_RECIPE_FILENAME = "policy_inference_recipe.json"
POLICY_STATE_IDENTITY_FILENAME = "policy_state_identity.safetensors"
ISAAC_DEPLOYMENT_ADAPTER_FILENAME = "isaac_deployment_adapter.json"
POLICY_STATE_DCP_IDENTITY_KEY = "policy_state_identity"
POLICY_STATE_DCP_IDENTITY_SCHEMA_VERSION = 2
POLICY_STATE_DCP_IDENTITY_BYTES = 4 + 3 * 32
AUTHENTICATED_SOURCE_MANIFEST_SCHEMA = "perceptron_isaac_authenticated_source_manifest_v1"
AUTHENTICATED_IMPORT_IDENTITY_SCHEMA = "perceptron_isaac_authenticated_import_identity_v1"
_MANIFEST_KEYS = {
    "schema_version",
    "policy_state_contract_version",
    "robotics_config_sha256",
    "training_dataset_expression_sha256",
    "datasets",
}
_DATASET_KEYS = {
    "contract_hash",
    "contract_version",
    "deployment_profile_hash",
    "mode",
    "normalization_identities",
    "target_dim",
}
_NORMALIZATION_KEYS = {
    "schema_version",
    "policy_state_manifest_sha256",
    "entries_sha256",
    "entries",
    "coverage",
}
_RECIPE_KEYS = {
    "schema_version",
    "policy_state_manifest_sha256",
    "recipes_sha256",
    "recipes",
}
_NORMALIZATION_ENTRY_REQUIRED_KEYS = {
    "action",
    "action_dim",
    "action_horizon",
    "action_normalized",
    "action_normalization_eps",
    "action_representation",
    "clip_normalized_actions",
    "clip_normalized_max",
    "normalization_scope",
    "objective",
    "policy_state_contract_hash",
    "policy_state_contract_version",
    "policy_state_dataset",
    "proprio",
    "proprio_dim",
    "proprio_normalization_eps",
    "proprio_normalized",
    "relative_exclude_joints",
    "schema",
    "stats_key",
    "target_fps",
    "stats_sha256",
}
_RECIPE_RECORD_KEYS = {
    "policy_state_dataset",
    "normalization_scope",
    "objective",
    "lowering_transform",
    "lowering",
    "transform_chain",
    "deployment_profile",
    "model_config_sha256",
    "rendering",
    "image_preprocess",
}
_LOWERING_KEYS = {"layout", "selection", "observation", "action_wire"}
_SELECTION_KEYS = {
    "observations_per_action",
    "observation_window_seconds",
    "action_window_seconds",
    "action_anchor",
    "action_anchor_index",
    "observation_delta_indices",
    "action_delta_indices",
}
_OBSERVATION_KEYS = {
    "inline_media_bytes",
    "materialize_media",
    "emit_vector_observations",
    "emit_observation_timestamps",
    "timestamp_precision",
    "last_state_only",
    "include_depth_images",
    "include_reasoning_trace",
    "include_scene_description",
    "emit_scene_prediction_target",
    "scene_description_probability",
    "reasoning_max_points",
    "depth_key_substrings",
    "clip_window_seconds",
}
# Genesis recipe schema v7 (bumped 2 -> 7 by genesis #3404, 2026-08-19) added four
# conditioning fields to the canonical observation block. Conditioning itself is NOT
# implemented by this runtime, so the values are validated below and non-zero
# probabilities are rejected outright -- genesis forces conditioning ON at serving
# whenever the checkpoint was trained with nonzero support, so accepting such a recipe
# silently would create a train/serve prompt skew.
_CONDITIONING_OBSERVATION_KEYS = {
    "action_conditioning_probability",
    "action_conditioning_role",
    "mistake_conditioning_probability",
    "mistake_conditioning_override",
}
_OBSERVATION_KEYS_V7 = _OBSERVATION_KEYS | _CONDITIONING_OBSERVATION_KEYS
# Schema versions this importer implements exactly. Genesis bumped 2 -> 7 in a single
# commit, so no intermediate version was ever emitted.
_SUPPORTED_RECIPE_SCHEMA_VERSIONS = {2, 7}
_COMMON_ACTION_WIRE_KEYS = {
    "kind",
    "stats_key",
    "require_stats",
    "normalize",
    "eps",
    "clip_normalized_actions",
    "clip_normalized_max",
    "action_representation",
    "relative_exclude_joints",
    "drop_action_payload",
    "token_group_name",
}
_FLOW_ACTION_WIRE_KEYS = _COMMON_ACTION_WIRE_KEYS | {"marker_token_id"}
_FAST_ACTION_WIRE_KEYS = _COMMON_ACTION_WIRE_KEYS | {
    "tokenizer_name_or_path",
    "tokenizer_revision",
    "processor_artifact",
    "token_group_key",
    "emit_tokens_item",
}
# Keys genesis v7 attaches to a Flow action wire whenever
# action_conditioning_probability > 0 (its _FLOW_FAST_CONDITIONING_KEYS). Their
# presence means the recipe requires FAST-tokenized action conditioning; reject them
# with a targeted message instead of a generic unexpected-keys mismatch.
_FLOW_FAST_CONDITIONING_WIRE_KEYS = {
    "conditioning_tokenizer_name_or_path",
    "conditioning_tokenizer_revision",
    "conditioning_processor_artifact",
    "conditioning_token_group_key",
    "conditioning_token_group_name",
}
_DEPLOYMENT_PROFILE_KEYS = {"robot_type", "camera", "state_action_schema", "normalization"}
_RENDERING_KEYS = {"mharmony_encoding", "reserved_token_groups", "system_default_hints"}
_IMAGE_PREPROCESS_KEYS = {
    "patch_size",
    "pixel_shuffle_scale_factor",
    "temporal_patch_size",
    "max_num_patches",
    "min_num_patches",
}
_DEPLOYMENT_ADAPTER_KEYS = {
    "schema",
    "contract_sha256",
    "policy_state_dataset",
    "normalization_scope",
    "objective",
    "render_dataset_name",
    "robot_type",
    "control_mode",
    "image_size",
    "camera_order",
    "camera_views",
    "n_action_steps",
    "num_inference_steps",
    "num_flow_samples",
    "clip_action_pose",
    "gripper_binary_to_signed",
    "num_settle_steps",
    "settle_gripper",
    "normalize_task_text",
    "joint_signs",
    "joint_offsets",
    "normalization_profile_id",
    "normalization_profile_scope",
    "normalization_validation_status",
    "adapter_validation_status",
    "provenance",
}
MK1_MODEL_IMPORT_FILENAME = "mk1_model_import.json"
MK1_MODEL_IMPORT_SCHEMA = "perceptron_isaac_mk1_model_import_v2"


class IsaacCheckpointImportError(RuntimeError):
    """The source checkpoint cannot be safely imported."""


@dataclass(frozen=True)
class AuthenticatedIsaacContracts:
    manifest: dict[str, Any]
    normalization: dict[str, Any]
    inference_recipe: dict[str, Any]
    dcp_identity_sha256: str | None
    source_hashes: dict[str, str]
    source_artifact_manifest_sha256: str | None = None
    authenticated_import_identity_sha256: str | None = None
    contract_authentication: str = "dcp_authenticated"
    artifact_paths: dict[str, Path] | None = None


@dataclass(frozen=True)
class ValidatedIsaacDeploymentAdapter:
    payload: dict[str, Any]
    source_path: Path
    sha256: str
    normalization_entry: dict[str, Any]
    recipe: dict[str, Any]


@dataclass(frozen=True)
class ImportedIsaacPackage:
    output_path: Path
    policy_state_dataset: str
    normalization_scope: str
    objective: str
    provenance: dict[str, Any]


def _authenticated_inference_source_paths(hf_export: Path) -> list[Path]:
    """Return every raw HF inference artifact consumed by package construction."""
    if not (hf_export / "config.json").is_file() or not (hf_export / "tokenizer.json").is_file():
        raise IsaacCheckpointImportError("HF export requires config.json and tokenizer.json assets.")

    shards, _advertised_by_shard = _source_safetensors_layout(hf_export)
    paths = [
        hf_export / filename for filename in HF_RUNTIME_ASSET_FILENAMES if (hf_export / filename).is_file()
    ]
    index_path = hf_export / "model.safetensors.index.json"
    if index_path.is_file():
        paths.append(index_path)
    paths.extend(shards)

    root = hf_export.resolve()
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved.parent != root or path.name != str(path.relative_to(hf_export)):
            raise IsaacCheckpointImportError(
                f"HF inference artifact must be a root-level file inside the export: {path}."
            )
        if not resolved.is_file():
            raise IsaacCheckpointImportError(f"HF inference artifact does not exist: {path}.")
        unique[path.name] = path
    return [unique[name] for name in sorted(unique)]


def _authenticated_source_binding(
    hf_export: Path,
    *,
    contract_identity_sha256: str,
) -> tuple[dict[str, str], str, str]:
    paths = _authenticated_inference_source_paths(hf_export)
    source_hashes = {path.name: file_sha256(path) for path in paths}
    manifest = {
        "schema": AUTHENTICATED_SOURCE_MANIFEST_SCHEMA,
        "files": [
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": source_hashes[path.name],
            }
            for path in paths
        ],
    }
    source_manifest_sha256 = canonical_sha256(manifest)
    import_identity_sha256 = canonical_sha256(
        {
            "schema": AUTHENTICATED_IMPORT_IDENTITY_SCHEMA,
            "contract_identity_sha256": contract_identity_sha256,
            "source_artifact_manifest_sha256": source_manifest_sha256,
        }
    )
    return source_hashes, source_manifest_sha256, import_identity_sha256


def _verify_authenticated_source_binding(
    hf_export: Path,
    contracts: AuthenticatedIsaacContracts,
) -> None:
    """Fail if authenticated raw inference inputs changed before atomic packaging."""
    if (
        contracts.authenticated_import_identity_sha256 is None
        or contracts.dcp_identity_sha256 is None
        or contracts.source_artifact_manifest_sha256 is None
    ):
        raise IsaacCheckpointImportError("Authenticated import source binding is incomplete.")
    hashes, manifest_sha256, import_identity_sha256 = _authenticated_source_binding(
        hf_export,
        contract_identity_sha256=contracts.dcp_identity_sha256,
    )
    if not hmac.compare_digest(manifest_sha256, contracts.source_artifact_manifest_sha256):
        raise IsaacCheckpointImportError(
            "Authenticated HF inference artifact manifest changed before packaging."
        )
    if not hmac.compare_digest(import_identity_sha256, contracts.authenticated_import_identity_sha256):
        raise IsaacCheckpointImportError("Authenticated import identity changed before packaging.")
    for filename, digest in hashes.items():
        expected = contracts.source_hashes.get(filename)
        if expected is None or not hmac.compare_digest(digest, expected):
            raise IsaacCheckpointImportError(
                f"Authenticated HF inference artifact changed before packaging: {filename}."
            )


def _snapshot_authenticated_source(
    hf_export: Path,
    contracts: AuthenticatedIsaacContracts,
) -> DirectorySnapshot:
    """Copy authenticated inputs once, verify the copy, and consume only that stable snapshot."""
    try:
        snapshot = snapshot_directory(
            hf_export,
            prefix="lerobot-isaac-import-",
            root_entries=contracts.source_hashes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise IsaacCheckpointImportError(f"Cannot snapshot authenticated HF export: {exc}") from exc
    try:
        _verify_authenticated_source_binding(snapshot.root, contracts)
        inference_names = {path.name for path in _authenticated_inference_source_paths(snapshot.root)}
        for filename in sorted(set(contracts.source_hashes) - inference_names):
            path = snapshot.root / filename
            if not path.is_file() or path.is_symlink():
                raise IsaacCheckpointImportError(
                    f"Authenticated source contract artifact is missing or non-regular: {filename}."
                )
            actual = file_sha256(path)
            expected = contracts.source_hashes[filename]
            if not hmac.compare_digest(actual, expected):
                raise IsaacCheckpointImportError(
                    f"Authenticated source contract artifact changed before packaging: {filename}."
                )
    except BaseException:
        snapshot.cleanup()
        raise
    return snapshot


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = load_integrity_json_object(path)
    except ValueError as exc:
        raise IsaacCheckpointImportError(f"Cannot read strict JSON artifact {path}: {exc}") from exc
    if raw.get("placeholder") is True:
        raise IsaacCheckpointImportError(
            f"Checkpoint artifact {path.name} is explicitly marked placeholder and cannot be authenticated."
        )
    return raw


def _mk1_trained_package_identity(
    package_root: str | Path,
    *,
    trained_steps: int,
    mk1_model_import_sha256: str,
) -> dict[str, Any]:
    """Validate and return the immutable identity fields for an imported MK1 package."""
    root = Path(package_root)
    if not root.is_dir():
        raise IsaacCheckpointImportError(f"MK1 trained package root does not exist: {root}.")
    if not isinstance(trained_steps, int) or isinstance(trained_steps, bool) or trained_steps <= 0:
        raise IsaacCheckpointImportError("MK1 trained package manifest requires positive trained_steps.")
    try:
        require_integrity_sha256(
            mk1_model_import_sha256,
            context="MK1 trained package manifest mk1_model_import_sha256",
        )
    except ValueError as exc:
        raise IsaacCheckpointImportError(
            "MK1 trained package manifest requires a lowercase mk1_model_import SHA-256 digest."
        ) from exc

    model_import_path = root / "hf_model" / MK1_MODEL_IMPORT_FILENAME
    if not model_import_path.is_file() or model_import_path.is_symlink():
        raise IsaacCheckpointImportError(
            f"MK1 trained package is missing a regular hf_model/{MK1_MODEL_IMPORT_FILENAME}."
        )
    actual_model_import_sha256 = file_sha256(model_import_path)
    if not hmac.compare_digest(actual_model_import_sha256, mk1_model_import_sha256):
        raise IsaacCheckpointImportError(
            "MK1 trained package model-import record changed before manifest generation."
        )
    return {
        "artifact_kind": "trained_policy",
        "trained_steps": trained_steps,
        "mk1_model_import_sha256": mk1_model_import_sha256,
    }


def _write_mk1_trained_package_manifest(
    package_root: str | Path,
    *,
    trained_steps: int,
    mk1_model_import_sha256: str,
) -> str:
    """Write the exhaustive outer-package record and return its raw file digest."""
    root = Path(package_root)
    identity = _mk1_trained_package_identity(
        root,
        trained_steps=trained_steps,
        mk1_model_import_sha256=mk1_model_import_sha256,
    )

    try:
        return write_trained_package_manifest(
            root,
            label="MK1 trained package",
            manifest_filename=MK1_TRAINED_PACKAGE_MANIFEST_FILENAME,
            manifest_schema=MK1_TRAINED_PACKAGE_MANIFEST_SCHEMA,
            digest_field=MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
            identity=identity,
            delegated_directory="hf_model",
        )
    except RuntimeError as exc:
        raise IsaacCheckpointImportError(str(exc)) from exc


def materialize_qwen35_mharmony_vocab(
    tokenizer_path: str | Path,
    output_path: str | Path,
    *,
    source_vocab_path: str | Path | None = None,
) -> dict[str, Any]:
    """Derive mharmony's root token-to-id ``vocab.json`` from tokenizer.json.

    Hugging Face's tokenizer JSON nests the base BPE vocabulary under
    ``model.vocab``. Mharmony's Qwen3.5 encoder intentionally consumes the
    standalone root mapping instead; pointing it at tokenizer.json fails and a
    missing vocab silently falls back to an ambient Hub cache. Imported
    packages therefore materialize and hash this deterministic checkpoint-local
    asset.
    """
    tokenizer = Path(tokenizer_path)
    output = Path(output_path)
    raw = _load_json_object(tokenizer)
    model = raw.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if not isinstance(vocab, dict) or not vocab:
        raise IsaacCheckpointImportError("Qwen3.5 tokenizer.json is missing model.vocab.")
    if len(vocab) != QWEN35_BASE_VOCAB_SIZE:
        raise IsaacCheckpointImportError(
            f"Qwen3.5 base vocabulary size mismatch: expected {QWEN35_BASE_VOCAB_SIZE}, got {len(vocab)}."
        )
    if any(
        not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int)
        for token, token_id in vocab.items()
    ):
        raise IsaacCheckpointImportError("Qwen3.5 model.vocab must map strings to integer token IDs.")
    token_ids = set(vocab.values())
    if token_ids != set(range(QWEN35_BASE_VOCAB_SIZE)):
        raise IsaacCheckpointImportError(
            "Qwen3.5 model.vocab token IDs must be unique and contiguous from zero."
        )

    source_vocab = Path(source_vocab_path) if source_vocab_path is not None else None
    if source_vocab is not None and source_vocab.is_file():
        packaged = _load_json_object(source_vocab)
        if packaged != vocab:
            raise IsaacCheckpointImportError("HF export vocab.json differs from tokenizer.json model.vocab.")

    output.write_text(json.dumps(vocab, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "schema": QWEN35_VOCAB_SCHEMA,
        "algorithm": QWEN35_VOCAB_ALGORITHM,
        "token_count": len(vocab),
        "tokenizer_sha256": file_sha256(tokenizer),
        "vocab_sha256": file_sha256(output),
    }


def _exact_keys(raw: dict[str, Any], expected: set[str], *, location: str) -> None:
    missing = sorted(expected - set(raw))
    unexpected = sorted(set(raw) - expected)
    if missing or unexpected:
        raise IsaacCheckpointImportError(
            f"{location} keys mismatch: missing={missing}, unexpected={unexpected}."
        )


def _require_sha256(value: Any, *, location: str) -> str:
    try:
        return require_integrity_sha256(value, context=location)
    except ValueError as exc:
        raise IsaacCheckpointImportError(f"{location} must be a lowercase SHA-256 digest.") from exc


def _require_string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise IsaacCheckpointImportError(
            f"{location} must be a non-empty string without surrounding whitespace."
        )
    return value


def _require_optional_string(value: Any, *, location: str) -> str | None:
    return None if value is None else _require_string(value, location=location)


def _require_bool(value: Any, *, location: str) -> bool:
    if not isinstance(value, bool):
        raise IsaacCheckpointImportError(f"{location} must be a boolean.")
    return value


def _require_int(value: Any, *, location: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IsaacCheckpointImportError(f"{location} must be an integer.")
    if minimum is not None and value < minimum:
        raise IsaacCheckpointImportError(f"{location} must be >= {minimum}.")
    return value


def _require_optional_int(
    value: Any,
    *,
    location: str,
    minimum: int | None = None,
) -> int | None:
    return None if value is None else _require_int(value, location=location, minimum=minimum)


def _require_float(
    value: Any,
    *,
    location: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IsaacCheckpointImportError(f"{location} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise IsaacCheckpointImportError(f"{location} must be a finite number.")
    if positive and result <= 0.0:
        raise IsaacCheckpointImportError(f"{location} must be positive.")
    if minimum is not None and result < minimum:
        raise IsaacCheckpointImportError(f"{location} must be >= {minimum}.")
    return 0.0 if result == 0.0 else result


def _require_optional_float(
    value: Any,
    *,
    location: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float | None:
    return (
        None
        if value is None
        else _require_float(value, location=location, minimum=minimum, positive=positive)
    )


def _require_string_list(
    value: Any,
    *,
    location: str,
    nonempty: bool = False,
    sorted_values: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise IsaacCheckpointImportError(f"{location} must be a list.")
    result = [_require_string(item, location=f"{location}[]") for item in value]
    if nonempty and not result:
        raise IsaacCheckpointImportError(f"{location} must be non-empty.")
    if len(result) != len(set(result)):
        raise IsaacCheckpointImportError(f"{location} must not contain duplicates.")
    if sorted_values and result != sorted(result):
        raise IsaacCheckpointImportError(f"{location} must be sorted.")
    return result


def _require_int_list(value: Any, *, location: str) -> list[int]:
    if not isinstance(value, list):
        raise IsaacCheckpointImportError(f"{location} must be a list.")
    return [_require_int(item, location=f"{location}[]") for item in value]


def _validate_effective_stats_block(
    raw: Any,
    *,
    expected_dim: int,
    location: str,
) -> None:
    if not isinstance(raw, dict):
        raise IsaacCheckpointImportError(f"{location} must be an object.")
    _exact_keys(raw, {"q01", "q99"}, location=location)
    vectors: dict[str, list[float]] = {}
    for key in ("q01", "q99"):
        values = raw[key]
        if not isinstance(values, list) or len(values) != expected_dim:
            raise IsaacCheckpointImportError(f"{location}.{key} must contain exactly {expected_dim} values.")
        vectors[key] = [
            _require_float(value, location=f"{location}.{key}[{index}]") for index, value in enumerate(values)
        ]
    if any(upper < lower for lower, upper in zip(vectors["q01"], vectors["q99"], strict=True)):
        raise IsaacCheckpointImportError(f"{location} must satisfy q99 >= q01 elementwise.")


def _identity_tuple(record: Any, *, location: str) -> tuple[str, str, str]:
    if not isinstance(record, dict):
        raise IsaacCheckpointImportError(f"{location} must be an object.")
    values = tuple(record.get(key) for key in ("policy_state_dataset", "normalization_scope", "objective"))
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise IsaacCheckpointImportError(f"{location} has an invalid policy normalization identity.")
    if values[2] not in {"FAST", "Flow"}:
        raise IsaacCheckpointImportError(f"{location}.objective must be FAST or Flow.")
    return values  # type: ignore[return-value]


def _validate_manifest(raw: dict[str, Any]) -> tuple[dict[str, Any], set[tuple[str, str, str]]]:
    _exact_keys(raw, _MANIFEST_KEYS, location="policy_state_contracts.json")
    if raw["schema_version"] != 5 or raw["policy_state_contract_version"] != 1:
        raise IsaacCheckpointImportError("Unsupported policy-state manifest schema or contract version.")
    _require_sha256(raw["robotics_config_sha256"], location="manifest.robotics_config_sha256")
    _require_sha256(
        raw["training_dataset_expression_sha256"],
        location="manifest.training_dataset_expression_sha256",
    )
    datasets = raw["datasets"]
    if not isinstance(datasets, dict) or not datasets:
        raise IsaacCheckpointImportError("policy-state manifest datasets must be a non-empty object.")
    authorized: set[tuple[str, str, str]] = set()
    for dataset_name, record in datasets.items():
        if not isinstance(dataset_name, str) or not dataset_name.strip() or not isinstance(record, dict):
            raise IsaacCheckpointImportError("policy-state manifest contains an invalid dataset record.")
        _exact_keys(record, _DATASET_KEYS, location=f"manifest.datasets[{dataset_name!r}]")
        _require_sha256(
            record["contract_hash"], location=f"manifest.datasets[{dataset_name!r}].contract_hash"
        )
        _require_sha256(
            record["deployment_profile_hash"],
            location=f"manifest.datasets[{dataset_name!r}].deployment_profile_hash",
        )
        if record["contract_version"] != 1 or record["mode"] not in {"components", "none"}:
            raise IsaacCheckpointImportError(
                f"manifest dataset {dataset_name!r} has invalid contract mode/version."
            )
        target_dim = record["target_dim"]
        if isinstance(target_dim, bool) or not isinstance(target_dim, int) or target_dim < 0:
            raise IsaacCheckpointImportError(f"manifest dataset {dataset_name!r} has invalid target_dim.")
        identities = record["normalization_identities"]
        if not isinstance(identities, list) or not identities:
            raise IsaacCheckpointImportError(
                f"manifest dataset {dataset_name!r} requires normalization identities."
            )
        ordered = []
        for identity in identities:
            if not isinstance(identity, dict) or set(identity) != {"normalization_scope", "objective"}:
                raise IsaacCheckpointImportError(
                    f"manifest dataset {dataset_name!r} has malformed normalization identity."
                )
            item = (dataset_name, identity["normalization_scope"], identity["objective"])
            _identity_tuple(
                {
                    "policy_state_dataset": item[0],
                    "normalization_scope": item[1],
                    "objective": item[2],
                },
                location=f"manifest.datasets[{dataset_name!r}]",
            )
            ordered.append(item)
        if ordered != sorted(set(ordered)):
            raise IsaacCheckpointImportError(
                f"manifest dataset {dataset_name!r} normalization identities must be sorted and unique."
            )
        authorized.update(ordered)
    return raw, authorized


def _validate_normalization(
    raw: dict[str, Any],
    *,
    manifest: dict[str, Any],
    manifest_digest: str,
    authorized: set[tuple[str, str, str]],
) -> dict[str, Any]:
    _exact_keys(raw, _NORMALIZATION_KEYS, location=POLICY_NORMALIZATION_FILENAME)
    if raw["schema_version"] != 2:
        raise IsaacCheckpointImportError("Unsupported policy normalization schema version.")
    if not hmac.compare_digest(str(raw["policy_state_manifest_sha256"]), manifest_digest):
        raise IsaacCheckpointImportError("Normalization bundle is bound to a different manifest.")
    entries = raw["entries"]
    if not isinstance(entries, list) or not entries:
        raise IsaacCheckpointImportError("Normalization bundle entries must be non-empty.")
    computed = canonical_sha256(entries)
    if not hmac.compare_digest(str(raw["entries_sha256"]), computed):
        raise IsaacCheckpointImportError("Normalization entries digest mismatch.")
    identities: list[tuple[str, str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise IsaacCheckpointImportError(f"Normalization entry {index} must be an object.")
        location = f"normalization.entries[{index}]"
        _exact_keys(entry, _NORMALIZATION_ENTRY_REQUIRED_KEYS, location=location)
        identity = _identity_tuple(entry, location=location)
        if identity not in authorized:
            raise IsaacCheckpointImportError(
                f"Normalization entry {identity} is not authorized by the manifest."
            )
        dataset_record = manifest["datasets"][identity[0]]
        contract_hash = _require_sha256(
            entry["policy_state_contract_hash"],
            location=f"{location}.policy_state_contract_hash",
        )
        contract_version = _require_int(
            entry["policy_state_contract_version"],
            location=f"{location}.policy_state_contract_version",
            minimum=1,
        )
        if (
            contract_version != dataset_record["contract_version"]
            or contract_hash != dataset_record["contract_hash"]
        ):
            raise IsaacCheckpointImportError(
                f"Normalization entry {identity} contract does not match the manifest."
            )
        action_dim = _require_int(entry["action_dim"], location=f"{location}.action_dim", minimum=1)
        proprio_dim = _require_int(entry["proprio_dim"], location=f"{location}.proprio_dim", minimum=0)
        if proprio_dim != dataset_record["target_dim"]:
            raise IsaacCheckpointImportError(
                f"Normalization entry {identity} proprio_dim does not match manifest target_dim."
            )
        _require_int(entry["action_horizon"], location=f"{location}.action_horizon", minimum=1)
        # Genesis records the state/action layout name here (for example,
        # ``joint_gripper_14``), not the normalization algorithm.  The exact
        # value is authenticated by stats_sha256 and cross-checked against the
        # recipe deployment profile after all recipe records are validated.
        _require_string(entry["schema"], location=f"{location}.schema")
        _require_string(entry["stats_key"], location=f"{location}.stats_key")
        _require_optional_float(
            entry["target_fps"],
            location=f"{location}.target_fps",
            positive=True,
        )
        action_normalized = _require_bool(
            entry["action_normalized"], location=f"{location}.action_normalized"
        )
        proprio_normalized = _require_bool(
            entry["proprio_normalized"], location=f"{location}.proprio_normalized"
        )
        if not action_normalized or not proprio_normalized:
            raise IsaacCheckpointImportError(
                f"Normalization entry {identity} must normalize both action and proprio for native ISAAC."
            )
        _require_float(
            entry["action_normalization_eps"],
            location=f"{location}.action_normalization_eps",
            positive=True,
        )
        _require_float(
            entry["proprio_normalization_eps"],
            location=f"{location}.proprio_normalization_eps",
            positive=True,
        )
        _validate_effective_stats_block(
            entry["action"], expected_dim=action_dim, location=f"{location}.action"
        )
        _validate_effective_stats_block(
            entry["proprio"], expected_dim=proprio_dim, location=f"{location}.proprio"
        )
        clip_actions = _require_bool(
            entry["clip_normalized_actions"],
            location=f"{location}.clip_normalized_actions",
        )
        if clip_actions:
            _require_float(
                entry["clip_normalized_max"],
                location=f"{location}.clip_normalized_max",
                positive=True,
            )
        elif entry["clip_normalized_max"] is not None:
            raise IsaacCheckpointImportError(
                f"{location}.clip_normalized_max must be null when clipping is disabled."
            )
        representation = _require_string(
            entry["action_representation"], location=f"{location}.action_representation"
        )
        if representation not in {"absolute", "relative"}:
            raise IsaacCheckpointImportError(
                f"{location}.action_representation must be absolute or relative."
            )
        relative_exclusions = _require_string_list(
            entry["relative_exclude_joints"],
            location=f"{location}.relative_exclude_joints",
            sorted_values=True,
        )
        if representation == "absolute" and relative_exclusions:
            raise IsaacCheckpointImportError(
                f"{location}.relative_exclude_joints must be empty for absolute actions."
            )
        declared_stats_digest = _require_sha256(entry["stats_sha256"], location=f"{location}.stats_sha256")
        digest_payload = {key: value for key, value in entry.items() if key != "stats_sha256"}
        if not hmac.compare_digest(declared_stats_digest, canonical_sha256(digest_payload)):
            raise IsaacCheckpointImportError(f"Normalization entry {identity} stats digest mismatch.")
        identities.append(identity)
    if identities != sorted(set(identities)):
        raise IsaacCheckpointImportError("Normalization entries must be sorted and unique.")
    coverage = raw["coverage"]
    if not isinstance(coverage, dict) or set(coverage) != {"complete", "observed", "missing"}:
        raise IsaacCheckpointImportError("Normalization coverage is malformed.")
    if not isinstance(coverage["observed"], list) or not isinstance(coverage["missing"], list):
        raise IsaacCheckpointImportError("Normalization coverage observed/missing values must be lists.")
    observed = {
        _identity_tuple(item, location="normalization.coverage.observed") for item in coverage["observed"]
    }
    missing = {
        _identity_tuple(item, location="normalization.coverage.missing") for item in coverage["missing"]
    }
    if (
        observed != set(identities)
        or missing != authorized - observed
        or coverage["complete"] != (not missing)
    ):
        raise IsaacCheckpointImportError("Normalization coverage does not match entries and manifest.")
    return raw


def _validate_fast_processor_identity(raw: Any, *, location: str) -> None:
    if not isinstance(raw, dict):
        raise IsaacCheckpointImportError(f"{location} must be an object.")
    expected = {
        "schema_version",
        "source",
        "repository",
        "revision",
        "files_sha256",
        "file_count",
        "identity_sha256",
    }
    _exact_keys(raw, expected, location=location)
    if raw["schema_version"] != 1 or raw["source"] not in {"local", "huggingface_hub"}:
        raise IsaacCheckpointImportError(f"{location} has an unsupported schema or source.")
    if raw["source"] == "huggingface_hub":
        _require_string(raw["repository"], location=f"{location}.repository")
        revision = _require_string(raw["revision"], location=f"{location}.revision")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise IsaacCheckpointImportError(
                f"{location}.revision must be an immutable lowercase 40-hex commit."
            )
    elif raw["repository"] is not None or raw["revision"] is not None:
        raise IsaacCheckpointImportError(
            f"{location} local artifacts must not declare repository or revision."
        )
    _require_sha256(raw["files_sha256"], location=f"{location}.files_sha256")
    _require_int(raw["file_count"], location=f"{location}.file_count", minimum=1)
    supplied = _require_sha256(raw["identity_sha256"], location=f"{location}.identity_sha256")
    payload = {key: raw[key] for key in expected if key != "identity_sha256"}
    if not hmac.compare_digest(supplied, canonical_sha256(payload)):
        raise IsaacCheckpointImportError(f"{location}.identity_sha256 does not match its payload.")


def _validate_transform_chain(raw: Any, *, location: str) -> None:
    if not isinstance(raw, list) or not raw:
        raise IsaacCheckpointImportError(f"{location} must be a non-empty list.")
    for index, entry in enumerate(raw):
        item_location = f"{location}[{index}]"
        if not isinstance(entry, dict):
            raise IsaacCheckpointImportError(f"{item_location} must be an object.")
        _exact_keys(entry, {"resolution_phase", "policy", "config"}, location=item_location)
        if entry["resolution_phase"] not in {"pre_resolve", "post_resolve"}:
            raise IsaacCheckpointImportError(f"{item_location}.resolution_phase is unsupported.")
        if entry["policy"] not in {
            "folded",
            "replay",
            "target_canonicalization",
            "training_only",
        }:
            raise IsaacCheckpointImportError(f"{item_location}.policy is unsupported.")
        config = entry["config"]
        if not isinstance(config, dict):
            raise IsaacCheckpointImportError(f"{item_location}.config must be an object.")
        _require_string(config.get("name"), location=f"{item_location}.config.name")
        probability = _require_float(config.get("p"), location=f"{item_location}.config.p", minimum=0.0)
        if probability > 1.0:
            raise IsaacCheckpointImportError(f"{item_location}.config.p must be <= 1.")
        if entry["policy"] != "training_only" and probability != 1.0:
            raise IsaacCheckpointImportError(f"{item_location} serving-semantic transforms must use p=1.0.")
        if entry["policy"] == "replay" and entry["resolution_phase"] == "pre_resolve":
            raise IsaacCheckpointImportError(
                f"{item_location} cannot replay a pre-resolve transform at inference."
            )


def _validate_deployment_profile(
    raw: Any,
    *,
    manifest_record: dict[str, Any],
    location: str,
) -> None:
    if not isinstance(raw, dict):
        raise IsaacCheckpointImportError(f"{location} must be an object.")
    _exact_keys(raw, _DEPLOYMENT_PROFILE_KEYS, location=location)
    _require_string(raw["robot_type"], location=f"{location}.robot_type")
    camera = raw["camera"]
    if camera is not None:
        if not isinstance(camera, dict):
            raise IsaacCheckpointImportError(f"{location}.camera must be null or an object.")
        _exact_keys(camera, {"observation_keys", "prompt_views"}, location=f"{location}.camera")
        observation_keys = _require_string_list(
            camera["observation_keys"],
            location=f"{location}.camera.observation_keys",
            nonempty=True,
        )
        prompt_views = _require_string_list(
            camera["prompt_views"],
            location=f"{location}.camera.prompt_views",
            nonempty=True,
        )
        if len(observation_keys) != len(prompt_views):
            raise IsaacCheckpointImportError(
                f"{location}.camera observation keys and prompt views must have equal length."
            )
    schema = raw["state_action_schema"]
    if not isinstance(schema, dict):
        raise IsaacCheckpointImportError(f"{location}.state_action_schema must be an object.")
    _exact_keys(schema, {"name", "control_mode", "components"}, location=f"{location}.state_action_schema")
    _require_string(schema["name"], location=f"{location}.state_action_schema.name")
    _require_optional_string(schema["control_mode"], location=f"{location}.state_action_schema.control_mode")
    _require_string_list(
        schema["components"],
        location=f"{location}.state_action_schema.components",
        nonempty=True,
    )
    normalization = raw["normalization"]
    if normalization is not None:
        if not isinstance(normalization, dict):
            raise IsaacCheckpointImportError(f"{location}.normalization must be null or an object.")
        _exact_keys(normalization, {"action", "proprio"}, location=f"{location}.normalization")
        _require_optional_string(normalization["action"], location=f"{location}.normalization.action")
        _require_optional_string(normalization["proprio"], location=f"{location}.normalization.proprio")
    if not hmac.compare_digest(
        canonical_sha256(raw),
        manifest_record["deployment_profile_hash"],
    ):
        raise IsaacCheckpointImportError(
            f"{location} does not match the deployment_profile_hash in the manifest."
        )


def _validate_recipe_record(
    record: dict[str, Any],
    *,
    index: int,
    schema_version: int,
    manifest: dict[str, Any],
    authorized: set[tuple[str, str, str]],
) -> tuple[str, str, str]:
    location = f"inference_recipe.recipes[{index}]"
    _exact_keys(record, _RECIPE_RECORD_KEYS, location=location)
    identity = _identity_tuple(record, location=location)
    if identity not in authorized:
        raise IsaacCheckpointImportError(f"Recipe identity {identity} is not authorized by the manifest.")
    objective = identity[2]
    expected_lowering = (
        "trajectory_fast_pipeline" if objective == "FAST" else "trajectory_flow_matching_pipeline"
    )
    if record["lowering_transform"] != expected_lowering:
        raise IsaacCheckpointImportError(f"{location}.lowering_transform must be {expected_lowering!r}.")

    lowering = record["lowering"]
    if not isinstance(lowering, dict):
        raise IsaacCheckpointImportError(f"{location}.lowering must be an object.")
    _exact_keys(lowering, _LOWERING_KEYS, location=f"{location}.lowering")
    _require_string(lowering["layout"], location=f"{location}.lowering.layout")
    selection = lowering["selection"]
    observation = lowering["observation"]
    wire = lowering["action_wire"]
    if not isinstance(selection, dict) or not isinstance(observation, dict) or not isinstance(wire, dict):
        raise IsaacCheckpointImportError(
            f"{location}.lowering selection, observation, and action_wire must be objects."
        )
    _exact_keys(selection, _SELECTION_KEYS, location=f"{location}.lowering.selection")
    _require_int(
        selection["observations_per_action"],
        location=f"{location}.lowering.selection.observations_per_action",
        minimum=1,
    )
    _require_optional_float(
        selection["observation_window_seconds"],
        location=f"{location}.lowering.selection.observation_window_seconds",
        minimum=0.0,
    )
    _require_optional_float(
        selection["action_window_seconds"],
        location=f"{location}.lowering.selection.action_window_seconds",
        minimum=0.0,
    )
    _require_string(selection["action_anchor"], location=f"{location}.lowering.selection.action_anchor")
    _require_optional_int(
        selection["action_anchor_index"], location=f"{location}.lowering.selection.action_anchor_index"
    )
    observation_deltas = _require_int_list(
        selection["observation_delta_indices"],
        location=f"{location}.lowering.selection.observation_delta_indices",
    )
    action_deltas = _require_int_list(
        selection["action_delta_indices"],
        location=f"{location}.lowering.selection.action_delta_indices",
    )
    if not observation_deltas or not action_deltas:
        raise IsaacCheckpointImportError(f"{location}.lowering selection deltas must be non-empty.")

    observation_keys = _OBSERVATION_KEYS_V7 if schema_version >= 7 else _OBSERVATION_KEYS
    _exact_keys(observation, observation_keys, location=f"{location}.lowering.observation")
    for key in (
        "inline_media_bytes",
        "materialize_media",
        "emit_vector_observations",
        "emit_observation_timestamps",
        "last_state_only",
        "include_depth_images",
        "include_reasoning_trace",
        "include_scene_description",
        "emit_scene_prediction_target",
    ):
        _require_bool(observation[key], location=f"{location}.lowering.observation.{key}")
    _require_optional_int(
        observation["timestamp_precision"],
        location=f"{location}.lowering.observation.timestamp_precision",
        minimum=0,
    )
    probability = _require_float(
        observation["scene_description_probability"],
        location=f"{location}.lowering.observation.scene_description_probability",
        minimum=0.0,
    )
    if probability > 1.0:
        raise IsaacCheckpointImportError(
            f"{location}.lowering.observation.scene_description_probability must be <= 1."
        )
    if schema_version == 7:
        action_conditioning = _require_float(
            observation["action_conditioning_probability"],
            location=f"{location}.lowering.observation.action_conditioning_probability",
            minimum=0.0,
        )
        mistake_conditioning = _require_float(
            observation["mistake_conditioning_probability"],
            location=f"{location}.lowering.observation.mistake_conditioning_probability",
            minimum=0.0,
        )
        if action_conditioning > 1.0 or mistake_conditioning > 1.0:
            raise IsaacCheckpointImportError(
                f"{location}.lowering.observation conditioning probabilities must be <= 1."
            )
        conditioning_role = _require_string(
            observation["action_conditioning_role"],
            location=f"{location}.lowering.observation.action_conditioning_role",
        )
        if conditioning_role not in {"user", "agent"}:
            raise IsaacCheckpointImportError(
                f"{location}.lowering.observation.action_conditioning_role must be 'user' or 'agent'."
            )
        if action_conditioning == 0.0 and conditioning_role != "user":
            raise IsaacCheckpointImportError(
                f"{location}.lowering.observation action_conditioning_role must be 'user' when "
                "action_conditioning_probability is 0; role='agent' requires "
                "action_conditioning_probability > 0."
            )
        mistake_override = observation["mistake_conditioning_override"]
        if mistake_override is not None:
            _require_bool(
                mistake_override,
                location=f"{location}.lowering.observation.mistake_conditioning_override",
            )
        if mistake_conditioning == 0.0 and mistake_override is not None:
            raise IsaacCheckpointImportError(
                f"{location}.lowering.observation mistake_conditioning_override must be null when "
                "mistake_conditioning_probability is 0; a non-null value requires "
                "mistake_conditioning_probability > 0."
            )
        for key, probability in (
            ("action_conditioning_probability", action_conditioning),
            ("mistake_conditioning_probability", mistake_conditioning),
        ):
            if probability != 0.0:
                raise IsaacCheckpointImportError(
                    f"{location}.lowering.observation.{key}={probability} declares "
                    "action/mistake conditioning support, which this runtime does not "
                    "implement (no causal action history or 'mistake:' preamble line is "
                    "rendered, and genesis forces conditioning ON at serving whenever the "
                    "checkpoint was trained with nonzero support). Importing it would "
                    "create a silent train/serve prompt skew; use a checkpoint trained "
                    "without conditioning support."
                )
    _require_int(
        observation["reasoning_max_points"],
        location=f"{location}.lowering.observation.reasoning_max_points",
        minimum=0,
    )
    _require_string_list(
        observation["depth_key_substrings"],
        location=f"{location}.lowering.observation.depth_key_substrings",
    )
    _require_optional_float(
        observation["clip_window_seconds"],
        location=f"{location}.lowering.observation.clip_window_seconds",
        minimum=0.0,
    )
    conditioning_wire_keys = sorted(set(wire) & _FLOW_FAST_CONDITIONING_WIRE_KEYS)
    if conditioning_wire_keys:
        raise IsaacCheckpointImportError(
            f"{location}.lowering.action_wire carries conditioning keys "
            f"{conditioning_wire_keys}: the recipe requires FAST-tokenized action "
            "conditioning, which this runtime does not implement."
        )
    wire_keys = _FAST_ACTION_WIRE_KEYS if objective == "FAST" else _FLOW_ACTION_WIRE_KEYS
    _exact_keys(wire, wire_keys, location=f"{location}.lowering.action_wire")
    expected_kind = "fast_tokens" if objective == "FAST" else "flow_action"
    if wire["kind"] != expected_kind:
        raise IsaacCheckpointImportError(f"{location}.lowering.action_wire.kind must be {expected_kind!r}.")
    _require_string(wire["stats_key"], location=f"{location}.lowering.action_wire.stats_key")
    require_stats = _require_bool(
        wire["require_stats"], location=f"{location}.lowering.action_wire.require_stats"
    )
    normalize = _require_bool(wire["normalize"], location=f"{location}.lowering.action_wire.normalize")
    if require_stats and not normalize:
        raise IsaacCheckpointImportError(
            f"{location}.lowering.action_wire require_stats=true requires normalize=true."
        )
    _require_float(wire["eps"], location=f"{location}.lowering.action_wire.eps", positive=True)
    clip_actions = _require_bool(
        wire["clip_normalized_actions"],
        location=f"{location}.lowering.action_wire.clip_normalized_actions",
    )
    clip_max = _require_float(
        wire["clip_normalized_max"],
        location=f"{location}.lowering.action_wire.clip_normalized_max",
        minimum=0.0,
    )
    if clip_actions and (not normalize or clip_max <= 0.0):
        raise IsaacCheckpointImportError(
            f"{location}.lowering.action_wire clipping requires normalized actions and a positive bound."
        )
    representation = _require_string(
        wire["action_representation"],
        location=f"{location}.lowering.action_wire.action_representation",
    )
    if representation not in {"absolute", "relative"}:
        raise IsaacCheckpointImportError(
            f"{location}.lowering.action_wire.action_representation is unsupported."
        )
    exclusions = _require_string_list(
        wire["relative_exclude_joints"],
        location=f"{location}.lowering.action_wire.relative_exclude_joints",
        sorted_values=True,
    )
    if representation == "absolute" and exclusions:
        raise IsaacCheckpointImportError(
            f"{location}.lowering.action_wire absolute actions require no relative exclusions."
        )
    _require_bool(
        wire["drop_action_payload"],
        location=f"{location}.lowering.action_wire.drop_action_payload",
    )
    _require_string(wire["token_group_name"], location=f"{location}.lowering.action_wire.token_group_name")
    if objective == "Flow":
        _require_int(
            wire["marker_token_id"],
            location=f"{location}.lowering.action_wire.marker_token_id",
        )
    else:
        tokenizer = _require_string(
            wire["tokenizer_name_or_path"],
            location=f"{location}.lowering.action_wire.tokenizer_name_or_path",
        )
        revision = _require_optional_string(
            wire["tokenizer_revision"],
            location=f"{location}.lowering.action_wire.tokenizer_revision",
        )
        _validate_fast_processor_identity(
            wire["processor_artifact"],
            location=f"{location}.lowering.action_wire.processor_artifact",
        )
        artifact = wire["processor_artifact"]
        if artifact["source"] == "huggingface_hub" and (
            artifact["repository"] != tokenizer or artifact["revision"] != revision
        ):
            raise IsaacCheckpointImportError(
                f"{location}.lowering.action_wire tokenizer and processor artifact pins disagree."
            )
        if artifact["source"] == "local" and revision is not None:
            raise IsaacCheckpointImportError(
                f"{location}.lowering.action_wire local processor artifacts require tokenizer_revision=null."
            )
        _require_string(
            wire["token_group_key"],
            location=f"{location}.lowering.action_wire.token_group_key",
        )
        _require_bool(
            wire["emit_tokens_item"],
            location=f"{location}.lowering.action_wire.emit_tokens_item",
        )

    _validate_transform_chain(record["transform_chain"], location=f"{location}.transform_chain")
    _validate_deployment_profile(
        record["deployment_profile"],
        manifest_record=manifest["datasets"][identity[0]],
        location=f"{location}.deployment_profile",
    )
    _require_sha256(record["model_config_sha256"], location=f"{location}.model_config_sha256")

    rendering = record["rendering"]
    if not isinstance(rendering, dict):
        raise IsaacCheckpointImportError(f"{location}.rendering must be an object.")
    _exact_keys(rendering, _RENDERING_KEYS, location=f"{location}.rendering")
    if rendering["mharmony_encoding"] != "QWEN35_HARMONY":
        raise IsaacCheckpointImportError(f"{location}.rendering.mharmony_encoding must be QWEN35_HARMONY.")
    groups = rendering["reserved_token_groups"]
    if not isinstance(groups, list) or not groups:
        raise IsaacCheckpointImportError(f"{location}.rendering.reserved_token_groups must be non-empty.")
    last_end = -1
    fast_group_count = 0
    for group_index, group in enumerate(groups):
        group_location = f"{location}.rendering.reserved_token_groups[{group_index}]"
        if not isinstance(group, dict):
            raise IsaacCheckpointImportError(f"{group_location} must be an object.")
        name = group.get("name")
        expected_group_keys = {"name", "offset", "size"}
        if name != "coord" or "tokenizer" in group:
            expected_group_keys.add("tokenizer")
        _exact_keys(group, expected_group_keys, location=group_location)
        _require_optional_string(name, location=f"{group_location}.name")
        offset = _require_int(group["offset"], location=f"{group_location}.offset", minimum=0)
        size = _require_int(group["size"], location=f"{group_location}.size", minimum=1)
        tokenizer = group.get("tokenizer")
        if tokenizer is not None:
            tokenizer = _require_string(tokenizer, location=f"{group_location}.tokenizer")
        if offset < last_end:
            raise IsaacCheckpointImportError(
                f"{location}.rendering.reserved_token_groups must be ordered and non-overlapping."
            )
        last_end = offset + size
        if tokenizer == DEFAULT_FAST_PROCESSOR_REPOSITORY and size == 2048:
            fast_group_count += 1
    if objective == "FAST" and fast_group_count != 1:
        raise IsaacCheckpointImportError(
            f"{location}.rendering for a FAST objective requires exactly one 2048-token "
            "physical-intelligence/fast group."
        )
    _require_string_list(
        rendering["system_default_hints"],
        location=f"{location}.rendering.system_default_hints",
    )

    image = record["image_preprocess"]
    if not isinstance(image, dict):
        raise IsaacCheckpointImportError(f"{location}.image_preprocess must be an object.")
    _exact_keys(image, _IMAGE_PREPROCESS_KEYS, location=f"{location}.image_preprocess")
    for key in ("patch_size", "pixel_shuffle_scale_factor", "temporal_patch_size"):
        _require_int(image[key], location=f"{location}.image_preprocess.{key}", minimum=1)
    _require_optional_int(
        image["max_num_patches"],
        location=f"{location}.image_preprocess.max_num_patches",
        minimum=1,
    )
    _require_optional_int(
        image["min_num_patches"],
        location=f"{location}.image_preprocess.min_num_patches",
        minimum=1,
    )
    return identity


def _validate_recipe(
    raw: dict[str, Any],
    *,
    manifest: dict[str, Any],
    normalization: dict[str, Any],
    manifest_digest: str,
    authorized: set[tuple[str, str, str]],
) -> dict[str, Any]:
    _exact_keys(raw, _RECIPE_KEYS, location=POLICY_INFERENCE_RECIPE_FILENAME)
    if raw["schema_version"] not in _SUPPORTED_RECIPE_SCHEMA_VERSIONS:
        raise IsaacCheckpointImportError(
            "Unsupported policy inference recipe schema version "
            f"{raw['schema_version']!r}; supported: {sorted(_SUPPORTED_RECIPE_SCHEMA_VERSIONS)}."
        )
    schema_version = int(raw["schema_version"])
    if not hmac.compare_digest(str(raw["policy_state_manifest_sha256"]), manifest_digest):
        raise IsaacCheckpointImportError("Inference recipe is bound to a different manifest.")
    recipes = raw["recipes"]
    if not isinstance(recipes, list) or not recipes:
        raise IsaacCheckpointImportError("Inference recipes must be non-empty.")
    if not hmac.compare_digest(str(raw["recipes_sha256"]), canonical_sha256(recipes)):
        raise IsaacCheckpointImportError("Inference recipes digest mismatch.")
    identities = []
    for index, record in enumerate(recipes):
        if not isinstance(record, dict):
            raise IsaacCheckpointImportError(f"Inference recipe {index} must be an object.")
        identities.append(
            _validate_recipe_record(
                record,
                index=index,
                schema_version=schema_version,
                manifest=manifest,
                authorized=authorized,
            )
        )
    if identities != sorted(authorized) or len(identities) != len(set(identities)):
        raise IsaacCheckpointImportError("Inference recipe coverage does not exactly match the manifest.")
    recipes_by_identity = dict(zip(identities, recipes, strict=True))
    for entry in normalization["entries"]:
        identity = _identity_tuple(entry, location="policy normalization bundle")
        wire = recipes_by_identity[identity]["lowering"]["action_wire"]
        selection = recipes_by_identity[identity]["lowering"]["selection"]
        expected_schema = recipes_by_identity[identity]["deployment_profile"]["state_action_schema"]["name"]
        if entry["schema"] != expected_schema:
            raise IsaacCheckpointImportError(
                f"Inference recipe and normalization state/action schemas disagree for {identity}."
            )
        expected = {
            "action_horizon": len(selection["action_delta_indices"]),
            "stats_key": wire["stats_key"],
            "action_normalized": wire["normalize"],
            "clip_normalized_actions": wire["clip_normalized_actions"],
            "action_representation": wire["action_representation"],
            "relative_exclude_joints": sorted(wire["relative_exclude_joints"]),
        }
        actual = {
            "action_horizon": entry["action_horizon"],
            "stats_key": entry["stats_key"],
            "action_normalized": entry["action_normalized"],
            "clip_normalized_actions": entry["clip_normalized_actions"],
            "action_representation": entry["action_representation"],
            "relative_exclude_joints": sorted(entry["relative_exclude_joints"]),
        }
        if expected != actual:
            raise IsaacCheckpointImportError(
                f"Inference recipe and normalization semantics disagree for {identity}."
            )
        if wire["normalize"] and wire["eps"] != entry["action_normalization_eps"]:
            raise IsaacCheckpointImportError(
                f"Inference recipe normalization epsilon disagrees for {identity}."
            )
        if wire["clip_normalized_actions"] and wire["clip_normalized_max"] != entry["clip_normalized_max"]:
            raise IsaacCheckpointImportError(
                f"Inference recipe normalized-action clip bound disagrees for {identity}."
            )
    return raw


def build_dcp_identity(
    manifest: dict[str, Any],
    normalization: dict[str, Any],
    inference_recipe: dict[str, Any],
) -> bytes:
    return b"".join(
        (
            POLICY_STATE_DCP_IDENTITY_SCHEMA_VERSION.to_bytes(4, byteorder="big", signed=False),
            bytes.fromhex(canonical_sha256(manifest)),
            bytes.fromhex(normalization["entries_sha256"]),
            bytes.fromhex(canonical_sha256(inference_recipe)),
        )
    )


def load_dcp_identity(checkpoint_path: str | Path) -> bytes:
    """Read the identity tensor out of a DCP checkpoint.

    SECURITY: ``read_metadata`` unpickles the checkpoint's ``.metadata`` file, and ``dcp.load``
    reads its storage plan, both necessarily *before* the identity can be compared -- the
    identity being authenticated lives inside the very checkpoint being opened. Torch's DCP
    format offers no pickle-free metadata path, so a ``--dcp-checkpoint`` argument is a trust
    boundary: point it only at a checkpoint your own training run produced. The HF sidecar
    path (``load_hf_identity``) reads safetensors and never unpickles.
    """
    checkpoint = Path(checkpoint_path)
    try:
        metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    except Exception as exc:  # noqa: BLE001
        raise IsaacCheckpointImportError(f"Cannot read DCP metadata from {checkpoint}: {exc}") from exc
    state_metadata = getattr(metadata, "state_dict_metadata", {})
    tensor_metadata = state_metadata.get(POLICY_STATE_DCP_IDENTITY_KEY)
    if tensor_metadata is None:
        raise IsaacCheckpointImportError(
            f"DCP does not contain required {POLICY_STATE_DCP_IDENTITY_KEY!r} state."
        )
    size = tuple(getattr(tensor_metadata, "size", ()) or ())
    properties = getattr(tensor_metadata, "properties", None)
    dtype = getattr(properties, "dtype", None)
    if size != (POLICY_STATE_DCP_IDENTITY_BYTES,) or dtype != torch.uint8:
        raise IsaacCheckpointImportError(
            f"DCP identity must be uint8[{POLICY_STATE_DCP_IDENTITY_BYTES}], got dtype={dtype}, size={size}."
        )
    state = {POLICY_STATE_DCP_IDENTITY_KEY: torch.empty(POLICY_STATE_DCP_IDENTITY_BYTES, dtype=torch.uint8)}
    try:
        dcp.load(state, checkpoint_id=checkpoint)  # nosec B614 - documented trust boundary: --dcp-checkpoint must be the operator's own training output
    except Exception as exc:  # noqa: BLE001
        raise IsaacCheckpointImportError(f"Cannot load DCP policy-state identity: {exc}") from exc
    return bytes(state[POLICY_STATE_DCP_IDENTITY_KEY].tolist())


def load_hf_identity(hf_export_path: str | Path) -> bytes:
    """Load the converter-exported policy-state identity sidecar."""
    from safetensors.torch import load_file

    source = Path(hf_export_path) / POLICY_STATE_IDENTITY_FILENAME
    if not source.is_file():
        raise IsaacCheckpointImportError(f"HF export is missing required {POLICY_STATE_IDENTITY_FILENAME}.")
    try:
        tensors = load_file(source, device="cpu")
    except Exception as exc:  # noqa: BLE001
        raise IsaacCheckpointImportError(
            f"Cannot load HF policy-state identity from {source}: {exc}"
        ) from exc
    if set(tensors) != {POLICY_STATE_DCP_IDENTITY_KEY}:
        raise IsaacCheckpointImportError(
            f"HF identity sidecar must contain exactly {POLICY_STATE_DCP_IDENTITY_KEY!r}."
        )
    identity = tensors[POLICY_STATE_DCP_IDENTITY_KEY]
    if identity.dtype != torch.uint8 or tuple(identity.shape) != (POLICY_STATE_DCP_IDENTITY_BYTES,):
        raise IsaacCheckpointImportError(
            f"HF identity must be uint8[{POLICY_STATE_DCP_IDENTITY_BYTES}], "
            f"got dtype={identity.dtype}, size={tuple(identity.shape)}."
        )
    return bytes(identity.tolist())


def authenticate_isaac_checkpoint(
    hf_export_path: str | Path,
    dcp_checkpoint_path: str | Path | None = None,
) -> AuthenticatedIsaacContracts:
    """Authenticate converter sidecars against an HF-exported and/or DCP-owned identity."""
    hf_export = Path(hf_export_path)
    dcp_checkpoint = Path(dcp_checkpoint_path) if dcp_checkpoint_path is not None else None
    filenames = (
        POLICY_STATE_CONTRACT_FILENAME,
        POLICY_NORMALIZATION_FILENAME,
        POLICY_INFERENCE_RECIPE_FILENAME,
    )
    for filename in filenames:
        if not (hf_export / filename).is_file():
            raise IsaacCheckpointImportError(f"HF export is missing required {filename}.")
        if dcp_checkpoint is not None:
            if not (dcp_checkpoint / filename).is_file():
                raise IsaacCheckpointImportError(f"DCP checkpoint is missing required {filename}.")
            if not hmac.compare_digest(
                file_sha256(hf_export / filename), file_sha256(dcp_checkpoint / filename)
            ):
                raise IsaacCheckpointImportError(
                    f"HF-exported {filename} differs from its DCP source artifact."
                )

    manifest, authorized = _validate_manifest(_load_json_object(hf_export / filenames[0]))
    manifest_digest = canonical_sha256(manifest)
    normalization = _validate_normalization(
        _load_json_object(hf_export / filenames[1]),
        manifest=manifest,
        manifest_digest=manifest_digest,
        authorized=authorized,
    )
    recipe = _validate_recipe(
        _load_json_object(hf_export / filenames[2]),
        manifest=manifest,
        normalization=normalization,
        manifest_digest=manifest_digest,
        authorized=authorized,
    )
    expected_identity = build_dcp_identity(manifest, normalization, recipe)
    identity_sources = []
    identity_path = hf_export / POLICY_STATE_IDENTITY_FILENAME
    if identity_path.is_file():
        hf_identity = load_hf_identity(hf_export)
        if not hmac.compare_digest(hf_identity, expected_identity):
            raise IsaacCheckpointImportError(
                "HF policy-state identity does not authenticate the exported manifest, normalization, and recipe."
            )
        identity_sources.append("hf_identity_safetensors")
    if dcp_checkpoint is not None:
        dcp_identity = load_dcp_identity(dcp_checkpoint)
        if not hmac.compare_digest(dcp_identity, expected_identity):
            raise IsaacCheckpointImportError(
                "DCP policy-state identity does not authenticate the exported manifest, normalization, and recipe."
            )
        identity_sources.append("dcp_identity_tensor")
    if not identity_sources:
        raise IsaacCheckpointImportError(
            f"Authenticated import requires {POLICY_STATE_IDENTITY_FILENAME} or a DCP checkpoint."
        )
    # MERGE NOTE (2026-08-13): the LIBERO-era line renamed the sidecar-only status to
    # "hf_identity_self_consistent" (correctly noting the sidecar proves internal
    # consistency, not provenance). The label is kept as "hf_identity_authenticated"
    # because the deployed step-60000 package's isaac_import_provenance.json carries it;
    # rename only together with a package migration.
    contract_authentication = {
        ("hf_identity_safetensors",): "hf_identity_authenticated",
        ("dcp_identity_tensor",): "dcp_authenticated",
        ("hf_identity_safetensors", "dcp_identity_tensor"): "hf_and_dcp_identity_authenticated",
    }[tuple(identity_sources)]
    dcp_identity_sha256 = hashlib.sha256(expected_identity).hexdigest()
    inference_hashes, source_manifest_sha256, import_identity_sha256 = _authenticated_source_binding(
        hf_export,
        contract_identity_sha256=dcp_identity_sha256,
    )
    source_hashes = {filename: file_sha256(hf_export / filename) for filename in filenames}
    source_hashes.update(inference_hashes)
    artifact_paths = {filename: hf_export / filename for filename in filenames}
    if identity_path.is_file():
        source_hashes[POLICY_STATE_IDENTITY_FILENAME] = file_sha256(identity_path)
        artifact_paths[POLICY_STATE_IDENTITY_FILENAME] = identity_path
    return AuthenticatedIsaacContracts(
        manifest=manifest,
        normalization=normalization,
        inference_recipe=recipe,
        dcp_identity_sha256=dcp_identity_sha256,
        source_hashes=source_hashes,
        source_artifact_manifest_sha256=source_manifest_sha256,
        authenticated_import_identity_sha256=import_identity_sha256,
        contract_authentication=contract_authentication,
        artifact_paths=artifact_paths,
    )


def _source_safetensors_layout(source: Path) -> tuple[list[Path], dict[str, set[str]] | None]:
    try:
        return safetensors_checkpoint_layout(source)
    except (FileNotFoundError, ValueError) as exc:
        raise IsaacCheckpointImportError(str(exc)) from exc


def _stage_hf_runtime_assets(source: Path, output: Path) -> dict[str, Any]:
    """Copy shared non-weight HF assets and materialize the canonical mharmony vocab."""
    output.mkdir(parents=True)
    for filename in HF_RUNTIME_ASSET_FILENAMES:
        source_file = source / filename
        if source_file.is_file():
            shutil.copy2(source_file, output / filename)
    if not (output / "config.json").is_file() or not (output / "tokenizer.json").is_file():
        raise IsaacCheckpointImportError("HF export requires config.json and tokenizer.json assets.")
    return materialize_qwen35_mharmony_vocab(
        output / "tokenizer.json",
        output / "vocab.json",
        source_vocab_path=output / "vocab.json",
    )


def _convert_qwen35_hf_weights_once(
    source_path: str | Path,
    output_path: str | Path,
    *,
    authenticated_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Copy a raw Genesis HF export and apply the serving RMSNorm correction once.

    At most one source shard and one converted shard dictionary are resident at
    a time. The output marker makes re-importing an already corrected package a
    hard error instead of silently subtracting one twice. The validated Genesis
    HF serving path loads parameters as bfloat16 and then subtracts one in
    bfloat16. Casting after a float32 subtraction is observably different after
    32 language-model layers, so the imported tensor must preserve that exact
    cast-then-subtract order.
    """
    from safetensors.torch import load_file, save_file

    source = Path(source_path)
    output = Path(output_path)
    if not source.is_dir():
        raise IsaacCheckpointImportError(f"HF export directory does not exist: {source}.")
    if (source / "rmsnorm_conversion.json").exists():
        raise IsaacCheckpointImportError(
            f"Source {source} is already marked RMSNorm-converted; refusing double conversion."
        )
    if output.exists():
        raise IsaacCheckpointImportError(f"RMSNorm conversion output already exists: {output}.")
    qwen35_vocab = _stage_hf_runtime_assets(source, output)

    shards, advertised_by_shard = _source_safetensors_layout(source)
    converted_keys: list[str] = []
    shard_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard_path in shards:
        if not shard_path.is_file():
            raise IsaacCheckpointImportError(f"Safetensors index references missing shard {shard_path}.")
        tensors = load_file(str(shard_path))
        keys = set(tensors)
        if advertised_by_shard is not None:
            advertised = advertised_by_shard[shard_path.name]
            if keys != advertised:
                raise IsaacCheckpointImportError(
                    f"Safetensors index mismatch for {shard_path.name}: "
                    f"missing={sorted(advertised - keys)}, unindexed={sorted(keys - advertised)}."
                )
        duplicates = seen.intersection(keys)
        if duplicates:
            raise IsaacCheckpointImportError(
                f"Duplicate tensor keys across source shards; first={sorted(duplicates)[:10]}."
            )
        for key in sorted(keys):
            if is_qwen35_offset_norm_key(key):
                tensors[key] = tensors[key].to(dtype=torch.bfloat16).sub(1.0)
                converted_keys.append(key)
        output_shard = output / shard_path.name
        save_file(tensors, output_shard, metadata={"format": "pt"})
        source_digest = (
            authenticated_source_hashes.get(shard_path.name)
            if authenticated_source_hashes is not None
            else file_sha256(shard_path)
        )
        if source_digest is None:
            raise IsaacCheckpointImportError(
                f"Authenticated source hashes do not bind Qwen3.5 shard {shard_path.name}."
            )
        shard_records.append(
            {
                "filename": shard_path.name,
                "source_sha256": source_digest,
                "output_sha256": file_sha256(output_shard),
                "tensor_count": len(keys),
            }
        )
        seen.update(keys)
        del tensors

    source_index = source / "model.safetensors.index.json"
    if source_index.is_file():
        shutil.copy2(source_index, output / source_index.name)
    if not converted_keys:
        raise IsaacCheckpointImportError(
            "No Qwen3.5 unit-offset RMSNorm weights were found; refusing an unverified conversion."
        )
    provenance = {
        "schema": QWEN35_CONVERSION_PROVENANCE_SCHEMA,
        "algorithm": QWEN35_CONVERSION_ALGORITHM,
        "conversion_dtype": QWEN35_CONVERSION_DTYPE,
        "converted_key_count": len(converted_keys),
        "converted_keys_sha256": canonical_sha256(converted_keys),
        "qwen35_vocab": qwen35_vocab,
        "shards": shard_records,
    }
    (output / "rmsnorm_conversion.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def _validate_mk1_artifact_metadata(
    raw_config: dict[str, Any],
    *,
    require_trained: bool,
) -> dict[str, Any]:
    """Validate the exporter-owned training-state marker without guessing defaults."""
    raw = raw_config.get("genesis_artifact")
    required = {"schema_version", "artifact_kind", "trained_steps"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise IsaacCheckpointImportError(
            "MK1 config.genesis_artifact must declare schema_version, artifact_kind, and trained_steps."
        )
    if raw["schema_version"] != 1:
        raise IsaacCheckpointImportError("Unsupported MK1 genesis_artifact schema_version.")
    kind = raw["artifact_kind"]
    steps = raw["trained_steps"]
    if kind not in {"trained_policy", "neutral_debug"}:
        raise IsaacCheckpointImportError(
            "MK1 genesis_artifact.artifact_kind must be trained_policy or neutral_debug."
        )
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise IsaacCheckpointImportError("MK1 genesis_artifact.trained_steps must be a nonnegative integer.")
    if kind == "neutral_debug" and steps != 0:
        raise IsaacCheckpointImportError("MK1 neutral_debug artifacts must declare trained_steps=0.")
    if kind == "trained_policy" and steps <= 0:
        raise IsaacCheckpointImportError(
            "MK1 trained_policy artifacts must declare a positive trained_steps value."
        )
    if require_trained:
        if set(raw) != required:
            raise IsaacCheckpointImportError(
                "Trained MK1 genesis_artifact keys do not match schema version 1."
            )
        if kind != "trained_policy":
            raise IsaacCheckpointImportError(
                "Production MK1 import rejects neutral_debug artifacts; use the debug package builder."
            )
    return {
        "schema_version": 1,
        "artifact_kind": str(kind),
        "trained_steps": int(steps),
    }


def _copy_mk1_hf_weights_once(
    source_path: str | Path,
    output_path: str | Path,
    *,
    authenticated_source_hashes: Mapping[str, str] | None = None,
    allow_test_only_reduced_geometry: bool = False,
) -> dict[str, Any]:
    """Copy an MK1 export without changing its already zero-centered RMSNorm tensors."""
    source = Path(source_path)
    output = Path(output_path)
    if not source.is_dir():
        raise IsaacCheckpointImportError(f"HF export directory does not exist: {source}.")
    if output.exists():
        raise IsaacCheckpointImportError(f"MK1 import output already exists: {output}.")

    try:
        contract, inventory = validate_mk1_checkpoint(
            source,
            allow_test_only_reduced_geometry=allow_test_only_reduced_geometry,
        )
    except Mk1CheckpointContractError as exc:
        raise IsaacCheckpointImportError(f"MK1 checkpoint contract validation failed: {exc}") from exc
    if not (source / "tokenizer.json").is_file():
        raise IsaacCheckpointImportError("MK1 HF export requires tokenizer.json.")
    artifact = _validate_mk1_artifact_metadata(
        _load_json_object(source / "config.json"),
        require_trained=not allow_test_only_reduced_geometry,
    )
    try:
        validate_mk1_tokenizer(source, contract)
    except Mk1CheckpointContractError as exc:
        raise IsaacCheckpointImportError(f"MK1 tokenizer contract validation failed: {exc}") from exc

    qwen35_vocab = _stage_hf_runtime_assets(source, output)

    def source_digest(path: Path) -> str:
        if authenticated_source_hashes is None:
            return file_sha256(path)
        digest = authenticated_source_hashes.get(path.name)
        if digest is None:
            raise IsaacCheckpointImportError(
                f"Authenticated source hashes do not bind MK1 artifact {path.name!r}."
            )
        return _require_sha256(digest, location=f"authenticated source hash for {path.name}")

    shard_records: list[dict[str, Any]] = []
    for shard_path in inventory.shards:
        output_shard = output / shard_path.name
        shutil.copy2(shard_path, output_shard)
        source_shard_digest = source_digest(shard_path)
        output_digest = file_sha256(output_shard)
        if not hmac.compare_digest(source_shard_digest, output_digest):
            raise IsaacCheckpointImportError(
                f"MK1 shard copy changed bytes for {shard_path.name}: "
                f"source={source_shard_digest}, output={output_digest}."
            )
        shard_records.append(
            {
                "filename": shard_path.name,
                "source_sha256": source_shard_digest,
                "output_sha256": output_digest,
                "tensor_count": sum(header.shard == shard_path.name for header in inventory.tensors.values()),
            }
        )

    shutil.copy2(inventory.index_path, output / inventory.index_path.name)
    try:
        copied_contract, _copied_inventory = validate_mk1_checkpoint(
            output,
            allow_test_only_reduced_geometry=allow_test_only_reduced_geometry,
        )
        validate_mk1_tokenizer(output, copied_contract)
    except Mk1CheckpointContractError as exc:
        raise IsaacCheckpointImportError(f"Copied MK1 checkpoint failed validation: {exc}") from exc

    source_assets = {
        *(source / filename for filename in HF_RUNTIME_ASSET_FILENAMES if (source / filename).is_file()),
        inventory.index_path,
    }
    source_files = {
        path.name: source_digest(path) for path in sorted(source_assets, key=lambda path: path.name)
    }
    source_files.update({record["filename"]: record["source_sha256"] for record in shard_records})
    copied_shard_digests = {record["filename"]: record["output_sha256"] for record in shard_records}
    output_files: dict[str, str] = {}
    for path in sorted(output.iterdir(), key=lambda candidate: candidate.name):
        if not path.is_file() or path.is_symlink():
            raise IsaacCheckpointImportError(
                f"MK1 import produced a non-regular model artifact: {path.name!r}."
            )
        output_files[path.name] = copied_shard_digests.get(path.name) or file_sha256(path)
    provenance = {
        "schema": MK1_MODEL_IMPORT_SCHEMA,
        "algorithm": "byte_for_byte_safetensors_copy",
        "backbone_family": "mk1_qwen3_6_moe",
        "rmsnorm_weight_convention": "zero_centered_1_plus_weight",
        "artifact": artifact,
        "source_files_sha256": source_files,
        "files_sha256": output_files,
        "qwen35_vocab": qwen35_vocab,
        "shards": shard_records,
    }
    (output / MK1_MODEL_IMPORT_FILENAME).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def _verify_mk1_import_copy_binding(
    hf_export: Path,
    provenance: dict[str, Any],
    contracts: AuthenticatedIsaacContracts,
) -> None:
    """Bind the bytes copied by the MK1 importer to the authenticated source snapshot.

    Re-hashing the source after conversion detects a persistent replacement, but not a source
    swapped only while it is copied and restored before the final check. The import record captures
    the bytes actually copied, so compare both its source and output digests directly with the
    authenticated source hashes before publishing the package.
    """
    if provenance.get("schema") != MK1_MODEL_IMPORT_SCHEMA:
        raise IsaacCheckpointImportError("MK1 conversion did not return a supported import record.")
    source_digests = provenance.get("source_files_sha256")
    output_digests = provenance.get("files_sha256")
    if not isinstance(source_digests, dict) or not isinstance(output_digests, dict):
        raise IsaacCheckpointImportError("MK1 import record is missing source/output digest maps.")

    expected_filenames = {path.name for path in _authenticated_inference_source_paths(hf_export)}
    if set(source_digests) != expected_filenames:
        raise IsaacCheckpointImportError(
            "MK1 import source inventory disagrees with the authenticated inference artifact: "
            f"missing={sorted(expected_filenames - set(source_digests))}, "
            f"unexpected={sorted(set(source_digests) - expected_filenames)}."
        )
    for filename in sorted(expected_filenames):
        authenticated = contracts.source_hashes.get(filename)
        copied_source = source_digests.get(filename)
        copied_output = output_digests.get(filename)
        if authenticated is None:
            raise IsaacCheckpointImportError(
                f"Authenticated contracts do not bind MK1 model source file {filename!r}."
            )
        if not isinstance(copied_source, str) or not hmac.compare_digest(copied_source, authenticated):
            raise IsaacCheckpointImportError(
                f"MK1 importer copied unauthenticated source bytes for {filename}."
            )
        # vocab.json is canonicalized from the authenticated tokenizer and may
        # differ only in JSON formatting from an optional source vocab. Every
        # other model artifact is copied byte for byte.
        expected_output = (
            provenance.get("qwen35_vocab", {}).get("vocab_sha256")
            if filename == "vocab.json"
            else authenticated
        )
        if (
            not isinstance(expected_output, str)
            or not isinstance(copied_output, str)
            or not hmac.compare_digest(copied_output, expected_output)
        ):
            raise IsaacCheckpointImportError(
                f"MK1 imported output bytes disagree with the authenticated source for {filename}."
            )


def convert_hf_weights_once(
    source_path: str | Path,
    output_path: str | Path,
    *,
    authenticated_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Import one HF model according to its explicit checkpoint norm contract."""
    source = Path(source_path)
    if not source.is_dir():
        raise IsaacCheckpointImportError(f"HF export directory does not exist: {source}.")
    config = _load_json_object(source / "config.json")
    genesis_vla = config.get("genesis_vla")
    if genesis_vla is None:
        return _convert_qwen35_hf_weights_once(
            source,
            output_path,
            authenticated_source_hashes=authenticated_source_hashes,
        )
    if not isinstance(genesis_vla, dict):
        raise IsaacCheckpointImportError("HF export config.json genesis_vla must be an object.")
    convention = genesis_vla.get("rmsnorm_weight_convention")
    if convention == "zero_centered_1_plus_weight":
        return _copy_mk1_hf_weights_once(
            source,
            output_path,
            authenticated_source_hashes=authenticated_source_hashes,
        )
    raise IsaacCheckpointImportError(
        "Unsupported genesis_vla.rmsnorm_weight_convention "
        f"{convention!r}; refusing to infer a weight transform."
    )


def _validate_mk1_recipe_coord_contract(hf_export: Path, recipe: dict[str, Any]) -> None:
    """Require MK1 neural and authenticated rendering coordinate metadata to agree."""
    config = _load_json_object(hf_export / "config.json")
    if config.get("genesis_vla") is None:
        return
    try:
        contract = read_mk1_config(hf_export)
        contract.validate_coord_reserved_token_groups(recipe["rendering"]["reserved_token_groups"])
    except (KeyError, TypeError, Mk1CheckpointContractError) as exc:
        raise IsaacCheckpointImportError(f"MK1 coordinate/rendering contract mismatch: {exc}") from exc


def _validate_trained_mk1_import_source(
    hf_export: Path,
) -> tuple[Mk1CheckpointContract, dict[str, Any]] | None:
    """Preflight a production MK1 source and return checkpoint-owned package fields."""
    raw_config = _load_json_object(hf_export / "config.json")
    if raw_config.get("genesis_vla") is None:
        return None
    try:
        contract, _inventory = validate_mk1_checkpoint(hf_export)
    except Mk1CheckpointContractError as exc:
        raise IsaacCheckpointImportError(f"MK1 checkpoint contract validation failed: {exc}") from exc
    artifact = _validate_mk1_artifact_metadata(raw_config, require_trained=True)
    return contract, artifact


def _derive_mk1_policy_fields(
    contract: Mk1CheckpointContract,
    artifact: dict[str, Any],
    *,
    served_action_dim: int,
    served_proprio_dim: int,
    served_action_horizon: int,
) -> dict[str, Any]:
    """Derive outer assertions from the inner graph while preserving served widths."""
    if served_action_dim > contract.action_expert.action_dim:
        raise IsaacCheckpointImportError(
            "Authenticated served action width exceeds the MK1 checkpoint maximum: "
            f"served={served_action_dim}, checkpoint={contract.action_expert.action_dim}."
        )
    if served_proprio_dim > contract.vector_encoder.max_states:
        raise IsaacCheckpointImportError(
            "Authenticated served proprio width exceeds the MK1 vector maximum: "
            f"served={served_proprio_dim}, checkpoint={contract.vector_encoder.max_states}."
        )
    if served_action_horizon > contract.action_expert.action_horizon:
        raise IsaacCheckpointImportError(
            "Authenticated served action horizon exceeds the MK1 checkpoint maximum: "
            f"served={served_action_horizon}, checkpoint={contract.action_expert.action_horizon}."
        )
    return {
        "artifact_kind": artifact["artifact_kind"],
        "trained_steps": artifact["trained_steps"],
        "vector_max_states": contract.vector_encoder.max_states,
        "max_state_dim": contract.vector_encoder.max_states,
        "max_action_dim": contract.action_expert.action_dim,
        "max_action_horizon": contract.action_expert.action_horizon,
    }


def _require_runtime_supported_normalization(entry: dict[str, Any]) -> None:
    """Reject contracts whose normalization the native runtime cannot reproduce.

    The contract schema models relative (delta) actions with an exclude-joint list and a
    configurable epsilon, but the native stats runtime only implements absolute actions at the
    hardcoded ``NATIVE_NORMALIZATION_EPS``. Importing such a checkpoint anyway would produce a
    package that unnormalizes deltas and commands them as absolute joint targets, so fail here
    rather than ship a silently wrong package.
    """
    representation = str(entry.get("action_representation"))
    if representation != "absolute":
        raise IsaacCheckpointImportError(
            f"Normalization entry declares action_representation={representation!r}, which the "
            "native ISAAC runtime does not implement; only 'absolute' can be served faithfully."
        )
    excluded = entry.get("relative_exclude_joints") or []
    if excluded:
        raise IsaacCheckpointImportError(
            "Normalization entry declares relative_exclude_joints, which the native ISAAC "
            "runtime does not implement."
        )
    for field_name in ("action_normalization_eps", "proprio_normalization_eps"):
        declared = float(entry[field_name])
        if declared != NATIVE_NORMALIZATION_EPS:
            raise IsaacCheckpointImportError(
                f"Normalization entry declares {field_name}={declared!r}, but the native ISAAC "
                f"quantile math is fixed at {NATIVE_NORMALIZATION_EPS!r}."
            )


def _select_identity_record(
    records: list[dict[str, Any]],
    *,
    policy_state_dataset: str,
    normalization_scope: str,
    objective: str,
    artifact: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if _identity_tuple(record, location=artifact)
        == (policy_state_dataset, normalization_scope, objective)
    ]
    if len(matches) != 1:
        raise IsaacCheckpointImportError(
            f"{artifact} requires exactly one record for "
            f"{policy_state_dataset}/{normalization_scope}/{objective}; found {len(matches)}."
        )
    return matches[0]


def validate_isaac_deployment_adapter(
    adapter_path: str | Path,
    *,
    contracts: AuthenticatedIsaacContracts,
    policy_state_dataset: str,
    normalization_scope: str,
    objective: str,
) -> ValidatedIsaacDeploymentAdapter:
    """Validate deployment-only behavior against one authenticated contract identity.

    Genesis owns training/lowering semantics. Resolution, camera choice for
    camera-less profiles, rollout cadence, safety clipping, and robot joint
    calibration are deployment concerns and live in this separately reviewed
    artifact. Its contract-file digest map prevents reusing it with another
    checkpoint.
    """
    normalization_entry = _select_identity_record(
        contracts.normalization["entries"],
        policy_state_dataset=policy_state_dataset,
        normalization_scope=normalization_scope,
        objective=objective,
        artifact="policy normalization bundle",
    )
    _require_runtime_supported_normalization(normalization_entry)
    recipe = _select_identity_record(
        contracts.inference_recipe["recipes"],
        policy_state_dataset=policy_state_dataset,
        normalization_scope=normalization_scope,
        objective=objective,
        artifact="policy inference recipe",
    )
    source = Path(adapter_path)
    payload = _load_json_object(source)
    _exact_keys(payload, _DEPLOYMENT_ADAPTER_KEYS, location=ISAAC_DEPLOYMENT_ADAPTER_FILENAME)
    if payload["schema"] != "perceptron_isaac_deployment_adapter_v1":
        raise IsaacCheckpointImportError("Unsupported ISAAC deployment adapter schema.")

    identity = _identity_tuple(payload, location=ISAAC_DEPLOYMENT_ADAPTER_FILENAME)
    requested_identity = (policy_state_dataset, normalization_scope, objective)
    if identity != requested_identity:
        raise IsaacCheckpointImportError(
            f"Deployment adapter identity {identity} does not match requested {requested_identity}."
        )
    if contracts.artifact_paths is None:
        raise IsaacCheckpointImportError("Validated contracts did not retain artifact source paths.")
    declared_hashes = payload["contract_sha256"]
    contract_filenames = {
        POLICY_STATE_CONTRACT_FILENAME,
        POLICY_NORMALIZATION_FILENAME,
        POLICY_INFERENCE_RECIPE_FILENAME,
    }
    if not isinstance(declared_hashes, dict) or set(declared_hashes) != contract_filenames:
        raise IsaacCheckpointImportError(
            "Deployment adapter contract_sha256 must bind all three Genesis contract files."
        )
    for filename in sorted(contract_filenames):
        declared = _require_sha256(
            declared_hashes[filename],
            location=f"deployment_adapter.contract_sha256[{filename!r}]",
        )
        artifact_path = contracts.artifact_paths.get(filename)
        if artifact_path is None or not hmac.compare_digest(declared, file_sha256(artifact_path)):
            raise IsaacCheckpointImportError(
                f"Deployment adapter is bound to a different {filename} artifact."
            )

    profile = recipe["deployment_profile"]
    schema = profile["state_action_schema"]
    robot_type = _require_string(payload["robot_type"], location="deployment_adapter.robot_type")
    if robot_type not in {"bi_yam", "so100_so101", "libero"}:
        raise IsaacCheckpointImportError(
            "Deployment adapter robot_type must be bi_yam, so100_so101, or libero."
        )
    legacy_generic_libero_profile = (
        robot_type == "libero"
        and profile["robot_type"] == "generic"
        and policy_state_dataset == "libero"
        and schema
        == {
            "name": "gripper_7",
            "control_mode": None,
            "components": ["world_vector", "rotation_delta", "gripper_closedness_action"],
        }
        and profile["camera"]
        == {
            "observation_keys": ["image", "wrist_image"],
            "prompt_views": ["primary", "wrist"],
        }
    )
    if robot_type != profile["robot_type"] and not legacy_generic_libero_profile:
        raise IsaacCheckpointImportError(
            "Deployment adapter robot_type disagrees with the Genesis deployment profile."
        )
    control_mode = _require_string(payload["control_mode"], location="deployment_adapter.control_mode")
    if schema["control_mode"] is not None and control_mode != schema["control_mode"]:
        raise IsaacCheckpointImportError(
            "Deployment adapter control_mode disagrees with the Genesis deployment profile."
        )

    _require_string(payload["render_dataset_name"], location="deployment_adapter.render_dataset_name")
    image_size = payload["image_size"]
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise IsaacCheckpointImportError("deployment_adapter.image_size must be [height, width].")
    for index, size in enumerate(image_size):
        _require_int(size, location=f"deployment_adapter.image_size[{index}]", minimum=1)
    camera_order = _require_string_list(
        payload["camera_order"], location="deployment_adapter.camera_order", nonempty=True
    )
    camera_views = _require_string_list(
        payload["camera_views"], location="deployment_adapter.camera_views", nonempty=True
    )
    if len(camera_order) != len(camera_views):
        raise IsaacCheckpointImportError(
            "Deployment adapter camera_order and camera_views must have equal length."
        )
    profile_camera = profile["camera"]
    if profile_camera is not None:
        profile_order = [key.rsplit(".", 1)[-1] for key in profile_camera["observation_keys"]]
        if camera_order != profile_order or camera_views != profile_camera["prompt_views"]:
            raise IsaacCheckpointImportError(
                "Deployment adapter camera routing disagrees with the Genesis deployment profile."
            )

    action_horizon = _require_int(
        normalization_entry["action_horizon"],
        location="normalization.action_horizon",
        minimum=1,
    )
    n_action_steps = _require_int(
        payload["n_action_steps"], location="deployment_adapter.n_action_steps", minimum=1
    )
    if n_action_steps > action_horizon:
        raise IsaacCheckpointImportError(
            "Deployment adapter n_action_steps cannot exceed the authenticated action horizon."
        )
    _require_int(
        payload["num_inference_steps"],
        location="deployment_adapter.num_inference_steps",
        minimum=1,
    )
    _require_int(
        payload["num_flow_samples"],
        location="deployment_adapter.num_flow_samples",
        minimum=1,
    )
    for key in ("clip_action_pose", "gripper_binary_to_signed", "normalize_task_text"):
        _require_bool(payload[key], location=f"deployment_adapter.{key}")
    _require_int(
        payload["num_settle_steps"],
        location="deployment_adapter.num_settle_steps",
        minimum=0,
    )
    _require_float(payload["settle_gripper"], location="deployment_adapter.settle_gripper")

    joint_signs = payload["joint_signs"]
    joint_offsets = payload["joint_offsets"]
    if (joint_signs is None) != (joint_offsets is None):
        raise IsaacCheckpointImportError(
            "Deployment adapter joint_signs and joint_offsets must both be null or both be lists."
        )
    if joint_signs is not None:
        if not isinstance(joint_signs, list) or not isinstance(joint_offsets, list):
            raise IsaacCheckpointImportError(
                "Deployment adapter joint_signs and joint_offsets must be lists."
            )
        if not joint_signs or len(joint_signs) != len(joint_offsets):
            raise IsaacCheckpointImportError(
                "Deployment adapter joint calibration lists must be non-empty and equal length."
            )
        max_width = min(normalization_entry["action_dim"], normalization_entry["proprio_dim"])
        if len(joint_signs) > max_width:
            raise IsaacCheckpointImportError(
                "Deployment adapter joint calibration exceeds action/state dimensions."
            )
        for index, sign in enumerate(joint_signs):
            if _require_float(sign, location=f"deployment_adapter.joint_signs[{index}]") not in {
                -1.0,
                1.0,
            }:
                raise IsaacCheckpointImportError("Deployment adapter joint_signs must contain only -1 or +1.")
        for index, offset in enumerate(joint_offsets):
            _require_float(offset, location=f"deployment_adapter.joint_offsets[{index}]")

    profile_fields = (
        "normalization_profile_id",
        "normalization_profile_scope",
        "normalization_validation_status",
    )
    profile_values = [
        _require_optional_string(payload[key], location=f"deployment_adapter.{key}") for key in profile_fields
    ]
    if any(value is None for value in profile_values) and not all(value is None for value in profile_values):
        raise IsaacCheckpointImportError(
            "Deployment adapter normalization profile id, scope, and status must be set together."
        )
    _require_string(
        payload["adapter_validation_status"],
        location="deployment_adapter.adapter_validation_status",
    )
    if not isinstance(payload["provenance"], dict) or not payload["provenance"]:
        raise IsaacCheckpointImportError("deployment_adapter.provenance must be a non-empty object.")
    return ValidatedIsaacDeploymentAdapter(
        payload=payload,
        source_path=source,
        sha256=file_sha256(source),
        normalization_entry=normalization_entry,
        recipe=recipe,
    )


def _stats_payload(entry: dict[str, Any], *, state_action_schema: str) -> dict[str, Any]:
    action = entry["action"]
    proprio = entry["proprio"]
    action_dim = int(entry.get("action_dim", len(action["q01"])))
    proprio_dim = int(entry.get("proprio_dim", len(proprio["q01"])))
    horizon = int(entry.get("action_horizon", 0))
    if action_dim != len(action["q01"]) or action_dim != len(action["q99"]):
        raise IsaacCheckpointImportError("Authenticated action stats dimensions are inconsistent.")
    if proprio_dim != len(proprio["q01"]) or proprio_dim != len(proprio["q99"]):
        raise IsaacCheckpointImportError("Authenticated proprio stats dimensions are inconsistent.")
    if horizon < 1:
        raise IsaacCheckpointImportError("Authenticated action_horizon must be positive.")
    if not entry["clip_normalized_actions"] or entry["clip_normalized_max"] is None:
        raise IsaacCheckpointImportError(
            "Native ISAAC packaging currently requires authenticated normalized-action clipping."
        )
    return {
        "schema": "flow_matching_stats_v1",
        "state_action_schema": state_action_schema,
        "action": action,
        "proprio": proprio,
        "target_fps": entry.get("target_fps"),
        "action_horizon": horizon,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "action_normalization_eps": entry["action_normalization_eps"],
        "proprio_normalization_eps": entry["proprio_normalization_eps"],
        "clip_normalized_actions": entry["clip_normalized_actions"],
        "clip_normalized_max": entry["clip_normalized_max"],
        "action_representation": entry["action_representation"],
        "relative_exclude_joints": entry["relative_exclude_joints"],
        "stats_sha256": entry["stats_sha256"],
    }


def _package_geometry(
    stats: dict[str, Any],
    adapter: ValidatedIsaacDeploymentAdapter,
) -> tuple[int, int, list[str], list[str], str, str, dict[str, Any]]:
    """Derive package geometry from the adapter's already-authenticated contract records."""
    recipe = adapter.recipe
    selection = recipe["lowering"]["selection"]
    n_obs_steps = int(selection["observations_per_action"])
    observation_deltas = selection["observation_delta_indices"]
    if n_obs_steps not in {1, 3} or observation_deltas != list(range(-(n_obs_steps - 1), 1)):
        raise IsaacCheckpointImportError(
            "Authenticated observation window must be nobs=1 [0] or nobs=3 [-2,-1,0]."
        )
    action_deltas = selection["action_delta_indices"]
    chunk_size = int(stats["action_horizon"])
    if action_deltas != list(range(chunk_size)):
        raise IsaacCheckpointImportError(
            "Authenticated action delta indices do not match the normalization horizon."
        )

    camera_order = list(adapter.payload["camera_order"])
    prompt_views = list(adapter.payload["camera_views"])
    action_layout = recipe["deployment_profile"]["state_action_schema"]["components"]
    robot_type = str(adapter.payload["robot_type"])
    control_mode = str(adapter.payload["control_mode"])
    if robot_type in {"bi_yam", "so100_so101"} and (
        len(action_layout) != int(stats["action_dim"]) or len(action_layout) != int(stats["proprio_dim"])
    ):
        raise IsaacCheckpointImportError(
            "Authenticated hardware state/action components must match action and proprio dimensions."
        )
    return (
        n_obs_steps,
        chunk_size,
        camera_order,
        [str(view) for view in prompt_views],
        robot_type,
        control_mode,
        recipe["image_preprocess"],
    )


_SO100_SEMANTIC_TO_HARDWARE = {
    "main_shoulder_pan": "shoulder_pan.pos",
    "main_shoulder_lift": "shoulder_lift.pos",
    "main_elbow_flex": "elbow_flex.pos",
    "main_wrist_flex": "wrist_flex.pos",
    "main_wrist_roll": "wrist_roll.pos",
    "main_gripper": "gripper.pos",
}


def _hardware_feature_names(robot_type: str, semantic_layout: list[str]) -> list[str] | None:
    if robot_type == "bi_yam":
        return list(semantic_layout)
    if robot_type == "so100_so101":
        expected_semantics = list(_SO100_SEMANTIC_TO_HARDWARE)
        if semantic_layout != expected_semantics:
            raise IsaacCheckpointImportError(
                "SO100/SO101 semantic state/action components do not match joint_gripper_6."
            )
        return [_SO100_SEMANTIC_TO_HARDWARE[name] for name in semantic_layout]
    return None


def _build_validated_isaac_package(
    hf_export_path: str | Path,
    output_path: str | Path,
    *,
    contracts: AuthenticatedIsaacContracts,
    policy_state_dataset: str,
    normalization_scope: str,
    deployment_adapter_path: str | Path,
    objective: str = "Flow",
    fast_processor_source: str | Path | None = None,
    allow_fast_remote_code: bool = False,
) -> ImportedIsaacPackage:
    """Build from a stable copy of the exact source bytes authenticated by the contracts."""
    with _snapshot_authenticated_source(Path(hf_export_path), contracts) as snapshot:
        return _build_validated_isaac_package_from_snapshot(
            snapshot.root,
            output_path,
            contracts=contracts,
            policy_state_dataset=policy_state_dataset,
            normalization_scope=normalization_scope,
            deployment_adapter_path=deployment_adapter_path,
            objective=objective,
            fast_processor_source=fast_processor_source,
            allow_fast_remote_code=allow_fast_remote_code,
        )


def _build_validated_isaac_package_from_snapshot(
    hf_export_path: str | Path,
    output_path: str | Path,
    *,
    contracts: AuthenticatedIsaacContracts,
    policy_state_dataset: str,
    normalization_scope: str,
    deployment_adapter_path: str | Path,
    objective: str = "Flow",
    fast_processor_source: str | Path | None = None,
    allow_fast_remote_code: bool = False,
) -> ImportedIsaacPackage:
    """Build one self-contained LeRobot ISAAC package atomically from a verified snapshot."""
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    from .configuration_perceptron_isaac import PerceptronIsaacConfig
    from .mharmony_native import (
        IsaacMharmonyRenderMetadata,
        build_isaac_training_proprio_contract,
    )
    from .processor_perceptron_isaac import make_perceptron_isaac_pre_post_processors

    if objective != "Flow":
        raise IsaacCheckpointImportError("Deployable ISAAC imports currently require objective='Flow'.")
    if not allow_fast_remote_code:
        raise IsaacCheckpointImportError(
            "ISAAC import packages executable FAST processor code; pass --allow-fast-remote-code "
            "after reviewing the pinned artifact revision and tree digest."
        )
    output = Path(output_path)
    if output.exists():
        raise IsaacCheckpointImportError(f"Import output already exists: {output}.")
    hf_export = Path(hf_export_path)
    mk1_source = _validate_trained_mk1_import_source(hf_export)
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter = validate_isaac_deployment_adapter(
        deployment_adapter_path,
        contracts=contracts,
        policy_state_dataset=policy_state_dataset,
        normalization_scope=normalization_scope,
        objective=objective,
    )
    normalization_entry = adapter.normalization_entry
    recipe = adapter.recipe
    _validate_mk1_recipe_coord_contract(hf_export, recipe)
    adapter_payload = adapter.payload
    stats = _stats_payload(
        normalization_entry,
        state_action_schema=recipe["deployment_profile"]["state_action_schema"]["name"],
    )
    stats.update(
        {
            "profile_id": adapter_payload["normalization_profile_id"],
            "profile_scope": adapter_payload["normalization_profile_scope"],
            "validation_status": adapter_payload["normalization_validation_status"],
        }
    )
    (
        n_obs_steps,
        chunk_size,
        camera_order,
        camera_views,
        robot_type,
        control_mode,
        image_preprocess,
    ) = _package_geometry(stats, adapter)
    mk1_policy_fields: dict[str, Any] = {}
    if mk1_source is not None:
        mk1_contract, mk1_artifact = mk1_source
        mk1_policy_fields = _derive_mk1_policy_fields(
            mk1_contract,
            mk1_artifact,
            served_action_dim=int(stats["action_dim"]),
            served_proprio_dim=int(stats["proprio_dim"]),
            served_action_horizon=chunk_size,
        )
    image_size = tuple(int(size) for size in adapter_payload["image_size"])
    rendering = recipe.get("rendering") or {}
    reserved_groups = rendering.get("reserved_token_groups") or [
        {"tokenizer": "physical-intelligence/fast", "size": 2048}
    ]
    action_layout = recipe["deployment_profile"]["state_action_schema"]["components"]
    is_hardware = robot_type in {"bi_yam", "so100_so101"}
    hardware_feature_names = _hardware_feature_names(
        robot_type,
        [str(component) for component in action_layout],
    )
    render_metadata = IsaacMharmonyRenderMetadata(
        camera_order=camera_order,
        camera_views=camera_views,
        image_size=list(image_size),
        n_obs_steps=n_obs_steps,
        action_horizon=chunk_size,
        action_dim=int(stats["action_dim"]),
        proprio_dim=int(stats["proprio_dim"]),
        target_fps=stats["target_fps"],
        dataset_name=str(adapter_payload["render_dataset_name"]),
        robot_type=robot_type,
        control_mode=control_mode,
        action_layout=[str(component) for component in action_layout],
        mharmony_encoding_name=str(rendering.get("mharmony_encoding") or "QWEN35_HARMONY"),
        mharmony_reserved_token_groups=reserved_groups,
        patch_size=int(image_preprocess.get("patch_size", 16)),
        pixel_shuffle_scale=int(image_preprocess.get("pixel_shuffle_scale_factor", 2)),
        temporal_patch_size=int(image_preprocess.get("temporal_patch_size", 2)),
        max_num_patches=image_preprocess.get("max_num_patches"),
        min_num_patches=image_preprocess.get("min_num_patches"),
        training_proprio_contract=build_isaac_training_proprio_contract(
            robot_type=robot_type,
            dataset_name=str(adapter_payload["render_dataset_name"]),
            proprio_dim=int(stats["proprio_dim"]),
            contract_version=int(normalization_entry["policy_state_contract_version"]),
            contract_hash=str(normalization_entry["policy_state_contract_hash"]),
        ),
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.import-", dir=output.parent))
    try:
        rmsnorm = convert_hf_weights_once(
            hf_export,
            temporary / "hf_model",
            authenticated_source_hashes=contracts.source_hashes,
        )
        if mk1_source is not None:
            _verify_mk1_import_copy_binding(hf_export, rmsnorm, contracts)
        if fast_processor_source is None:
            fast_processor_source = resolve_pinned_fast_processor_snapshot()
            fast_artifact = materialize_pinned_fast_processor_snapshot(
                fast_processor_source,
                temporary / "fast_processor",
            )
        else:
            fast_artifact = materialize_fast_processor_artifact(
                fast_processor_source,
                temporary / "fast_processor",
            )
        if not hmac.compare_digest(
            fast_artifact.tree_sha256,
            DEFAULT_FAST_PROCESSOR_TREE_SHA256,
        ):
            raise IsaacCheckpointImportError(
                "FAST processor artifact does not match the reviewed pinned revision: "
                f"expected {DEFAULT_FAST_PROCESSOR_TREE_SHA256}, got {fast_artifact.tree_sha256}."
            )
        stats_path = temporary / "isaac_stats.json"
        metadata_path = temporary / "native_render_metadata.json"
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        metadata_path.write_text(json.dumps(render_metadata.to_json_dict(), indent=2, sort_keys=True) + "\n")
        features = {
            f"observation.images.{camera}": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *image_size),
            )
            for camera in camera_order
        }
        features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(int(stats["proprio_dim"]),),
        )
        config = PerceptronIsaacConfig(
            device="cpu",
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=int(adapter_payload["n_action_steps"]),
            action_dim=int(stats["action_dim"]),
            proprio_dim=int(stats["proprio_dim"]),
            image_size=image_size,
            camera_order=tuple(camera_order),
            allow_image_key_fallback=False,
            hf_model_path="hf_model",
            native_stats_path="isaac_stats.json",
            native_render_metadata_path="native_render_metadata.json",
            fast_processor_path="fast_processor",
            fast_processor_tree_sha256=fast_artifact.tree_sha256,
            dataset_name=str(adapter_payload["render_dataset_name"]),
            policy_state_dataset=policy_state_dataset,
            robot_type=robot_type,
            control_mode=control_mode,
            target_fps=float(stats["target_fps"]) if stats["target_fps"] is not None else None,
            action_feature_names=hardware_feature_names,
            state_feature_names=hardware_feature_names,
            strict_hardware_feature_contract=is_hardware,
            strict_environment_feature_contract=robot_type == "libero",
            normalization_profile_id=adapter_payload["normalization_profile_id"],
            normalization_profile_scope=adapter_payload["normalization_profile_scope"],
            normalization_validation_status=adapter_payload["normalization_validation_status"],
            deployment_adapter_sha256=adapter.sha256,
            mk1_model_import_sha256=(
                file_sha256(temporary / "hf_model" / MK1_MODEL_IMPORT_FILENAME)
                if mk1_source is not None
                else None
            ),
            render_patch_size=render_metadata.patch_size,
            render_pixel_shuffle_scale=render_metadata.pixel_shuffle_scale,
            render_temporal_patch_size=render_metadata.temporal_patch_size,
            render_max_num_patches=render_metadata.max_num_patches,
            render_min_num_patches=render_metadata.min_num_patches,
            apply_offset_norm=False,
            num_inference_steps=int(adapter_payload["num_inference_steps"]),
            num_flow_samples=int(adapter_payload["num_flow_samples"]),
            clip_normalized_max=float(stats["clip_normalized_max"]),
            clip_action_pose=bool(adapter_payload["clip_action_pose"]),
            gripper_binary_to_signed=bool(adapter_payload["gripper_binary_to_signed"]),
            num_settle_steps=int(adapter_payload["num_settle_steps"]),
            settle_gripper=float(adapter_payload["settle_gripper"]),
            normalize_task_text=bool(adapter_payload["normalize_task_text"]),
            joint_signs=adapter_payload["joint_signs"],
            joint_offsets=adapter_payload["joint_offsets"],
            input_features=features,
            output_features={
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(int(stats["action_dim"]),))
            },
            **mk1_policy_fields,
        )
        config._save_pretrained(temporary)
        processor_config = PerceptronIsaacConfig(
            **{
                **config.__dict__,
                "hf_model_path": str(temporary / "hf_model"),
                "native_stats_path": str(stats_path),
                "native_render_metadata_path": str(metadata_path),
                # Keep this package-relative in the serialized pipeline. The
                # v0.6 pretrained loader reconnects it to the package root.
                "fast_processor_path": "fast_processor",
            }
        )
        preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(processor_config)
        preprocessor.save_pretrained(temporary)
        postprocessor.save_pretrained(temporary)
        for filename in (
            POLICY_STATE_CONTRACT_FILENAME,
            POLICY_NORMALIZATION_FILENAME,
            POLICY_INFERENCE_RECIPE_FILENAME,
        ):
            source_artifact = hf_export / filename
            if contracts.artifact_paths is None or filename not in contracts.artifact_paths:
                raise IsaacCheckpointImportError(f"Validated contracts did not retain {filename} source.")
            shutil.copy2(source_artifact, temporary / filename)
        if (
            contracts.artifact_paths is not None
            and POLICY_STATE_IDENTITY_FILENAME in contracts.artifact_paths
        ):
            shutil.copy2(
                hf_export / POLICY_STATE_IDENTITY_FILENAME,
                temporary / POLICY_STATE_IDENTITY_FILENAME,
            )
        packaged_adapter = temporary / ISAAC_DEPLOYMENT_ADAPTER_FILENAME
        shutil.copy2(adapter.source_path, packaged_adapter)
        packaged_adapter_digest = file_sha256(packaged_adapter)
        if not hmac.compare_digest(packaged_adapter_digest, adapter.sha256):
            raise IsaacCheckpointImportError(
                "ISAAC deployment adapter changed while the package was being built."
            )
        provenance = {
            "schema": QWEN35_IMPORT_PROVENANCE_SCHEMA,
            "contract_authentication": contracts.contract_authentication,
            "dcp_identity_sha256": contracts.dcp_identity_sha256,
            "policy_state_identity_source": contracts.contract_authentication,
            "source_artifact_manifest_sha256": contracts.source_artifact_manifest_sha256,
            "authenticated_import_identity_sha256": contracts.authenticated_import_identity_sha256,
            "policy_state_dataset": policy_state_dataset,
            "normalization_scope": normalization_scope,
            "objective": objective,
            "source_hashes": contracts.source_hashes,
            "deployment_adapter": {
                "filename": ISAAC_DEPLOYMENT_ADAPTER_FILENAME,
                "sha256": adapter.sha256,
                "validation_status": adapter_payload["adapter_validation_status"],
            },
            "rmsnorm_conversion": rmsnorm,
            "fast_processor": {
                "repository": DEFAULT_FAST_PROCESSOR_REPOSITORY,
                "revision": DEFAULT_FAST_PROCESSOR_REVISION,
                "tree_sha256": fast_artifact.tree_sha256,
                "file_count": fast_artifact.file_count,
            },
        }
        (temporary / "isaac_import_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        if mk1_source is not None:
            if config.mk1_model_import_sha256 is None:
                raise IsaacCheckpointImportError(
                    "MK1 trained package config is missing its model-import record digest."
                )
            identity = _mk1_trained_package_identity(
                temporary,
                trained_steps=config.trained_steps,
                mk1_model_import_sha256=config.mk1_model_import_sha256,
            )
            try:
                finalize_trained_package_config(
                    config,
                    temporary,
                    label="MK1 trained package",
                    manifest_filename=MK1_TRAINED_PACKAGE_MANIFEST_FILENAME,
                    manifest_schema=MK1_TRAINED_PACKAGE_MANIFEST_SCHEMA,
                    digest_field=MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
                    identity=identity,
                    delegated_directory="hf_model",
                )
            except RuntimeError as exc:
                raise IsaacCheckpointImportError(str(exc)) from exc
        else:
            try:
                finalize_trained_package_config(
                    config,
                    temporary,
                    label="Qwen3.5 imported package",
                    manifest_filename=QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME,
                    manifest_schema=QWEN35_TRAINED_PACKAGE_MANIFEST_SCHEMA,
                    digest_field=QWEN35_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
                    identity={
                        "artifact_kind": config.artifact_kind,
                        "trained_steps": config.trained_steps,
                    },
                    metadata={"model_storage": AUTHENTICATED_IMPORT_MODEL_STORAGE},
                    delegated_directory="hf_model",
                )
            except RuntimeError as exc:
                raise IsaacCheckpointImportError(str(exc)) from exc
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ImportedIsaacPackage(
        output_path=output,
        policy_state_dataset=policy_state_dataset,
        normalization_scope=normalization_scope,
        objective=objective,
        provenance=provenance,
    )


def import_authenticated_isaac_checkpoint(
    hf_export_path: str | Path,
    dcp_checkpoint_path: str | Path | None,
    output_path: str | Path,
    *,
    policy_state_dataset: str,
    normalization_scope: str,
    deployment_adapter_path: str | Path,
    objective: str = "Flow",
    fast_processor_source: str | Path | None = None,
    allow_fast_remote_code: bool = False,
) -> ImportedIsaacPackage:
    contracts = authenticate_isaac_checkpoint(hf_export_path, dcp_checkpoint_path)
    return _build_validated_isaac_package(
        hf_export_path,
        output_path,
        contracts=contracts,
        policy_state_dataset=policy_state_dataset,
        normalization_scope=normalization_scope,
        deployment_adapter_path=deployment_adapter_path,
        objective=objective,
        fast_processor_source=fast_processor_source,
        allow_fast_remote_code=allow_fast_remote_code,
    )
