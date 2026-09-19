"""Integrity and layout checks for packaged Qwen3.5 checkpoints."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .checkpoint_integrity import (
    DirectorySnapshot,
    canonical_json_sha256,
    load_json_object,
    require_sha256,
    snapshot_directory,
    verify_file_sha256,
)

SAFETENSORS_SINGLE_FILE = "model.safetensors"
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
QWEN35_IMPORT_PROVENANCE_FILE = "isaac_import_provenance.json"
QWEN35_CONVERSION_PROVENANCE_FILE = "rmsnorm_conversion.json"
QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE = "source_rmsnorm_conversion.json"
QWEN35_WEIGHT_CONVENTION_FILE = "qwen35_weight_convention.json"
# v3 adds the importing caller's conditioning declaration. v2 packages predate that field
# and stay loadable; every accepted version keeps its own closed, exhaustively checked key set.
QWEN35_IMPORT_PROVENANCE_SCHEMA_V2 = "perceptron_isaac_import_provenance_v2"
QWEN35_IMPORT_PROVENANCE_SCHEMA = "perceptron_isaac_import_provenance_v3"
QWEN35_CONVERSION_PROVENANCE_SCHEMA = "perceptron_isaac_rmsnorm_conversion_v2"
QWEN35_WEIGHT_CONVENTION_SCHEMA = "perceptron_isaac_qwen35_weight_convention_v1"
QWEN35_NATIVE_RMSNORM_CONVENTION = "native_unit_offset"
QWEN35_CONVERSION_ALGORITHM = "qwen3_5_rmsnorm_cast_bfloat16_then_minus_one"
QWEN35_CONVERSION_DTYPE = "bfloat16"
QWEN35_VOCAB_SCHEMA = "perceptron_isaac_qwen35_vocab_v1"
QWEN35_VOCAB_ALGORITHM = "extract_tokenizer_json_model_vocab"
QWEN35_BASE_VOCAB_SIZE = 248_044
QWEN35_PACKAGED_SOURCE_FILENAMES = frozenset(
    {
        "policy_state_contracts.json",
        "policy_normalization.json",
        "policy_inference_recipe.json",
    }
)
QWEN35_OPTIONAL_PACKAGED_SOURCE_FILENAMES = frozenset({"policy_state_identity.safetensors"})
QWEN35_SNAPSHOT_ROOT_ENTRIES = frozenset(
    {
        "hf_model",
        QWEN35_IMPORT_PROVENANCE_FILE,
        "isaac_deployment_adapter.json",
        *QWEN35_PACKAGED_SOURCE_FILENAMES,
        *QWEN35_OPTIONAL_PACKAGED_SOURCE_FILENAMES,
    }
)
QWEN35_IMPORTED_RUNTIME_ROOT_ENTRIES = frozenset(
    {
        *QWEN35_SNAPSHOT_ROOT_ENTRIES,
        ".gitattributes",
        "README.md",
        "config.json",
        "fast_processor",
        "isaac_stats.json",
        "native_render_metadata.json",
        "policy_postprocessor.json",
        "policy_preprocessor.json",
        "suite_by_task_index.json",
        "suite_stats.json",
    }
)

_IMPORT_PROVENANCE_KEYS_V2 = {
    "schema",
    "contract_authentication",
    "dcp_identity_sha256",
    "policy_state_identity_source",
    "source_artifact_manifest_sha256",
    "authenticated_import_identity_sha256",
    "policy_state_dataset",
    "normalization_scope",
    "objective",
    "source_hashes",
    "deployment_adapter",
    "rmsnorm_conversion",
    "fast_processor",
}
_IMPORT_PROVENANCE_KEYS_BY_SCHEMA: Mapping[str, set[str]] = {
    QWEN35_IMPORT_PROVENANCE_SCHEMA_V2: _IMPORT_PROVENANCE_KEYS_V2,
    QWEN35_IMPORT_PROVENANCE_SCHEMA: _IMPORT_PROVENANCE_KEYS_V2 | {"conditioning_deployment"},
}
_CONDITIONING_DEPLOYMENT_KEYS = {"renders_action_conditioning", "renders_mistake_conditioning"}
_CONVERSION_PROVENANCE_KEYS = {
    "schema",
    "algorithm",
    "conversion_dtype",
    "converted_key_count",
    "converted_keys_sha256",
    "qwen35_vocab",
    "shards",
}
_SHARD_RECORD_KEYS = {"filename", "source_sha256", "output_sha256", "tensor_count"}
_VOCAB_RECORD_KEYS = {
    "schema",
    "algorithm",
    "token_count",
    "tokenizer_sha256",
    "vocab_sha256",
}
_DEPLOYMENT_ADAPTER_KEYS = {"filename", "sha256", "validation_status"}
_FAST_PROCESSOR_KEYS = {"repository", "revision", "tree_sha256", "file_count"}
_CONTRACT_AUTHENTICATION_VALUES = {
    "hf_identity_authenticated",
    "dcp_authenticated",
    "hf_and_dcp_identity_authenticated",
}
_QWEN35_OFFSET_NORM_KEY = re.compile(
    r"^model\.language_model\.(?:norm|layers\.\d+\."
    r"(?:input_layernorm|post_attention_layernorm|self_attn\.(?:q_norm|k_norm)))\.weight$"
)
_VERIFIED_QWEN35_HANDLE_TOKEN = object()

HF_RUNTIME_ASSET_FILENAMES = frozenset(
    {
        "config.json",
        "config.toml",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "mk1_legacy_export_migration.json",
    }
)


class _VerifiedQwen35Package:
    """One-use exact-root snapshot accepted by the private model-load boundary."""

    __slots__ = ("_closed", "_snapshot", "apply_offset_norm", "hf_model_path", "root")

    def __init__(
        self,
        token: object,
        snapshot: DirectorySnapshot,
        *,
        hf_model_path: Path,
        apply_offset_norm: bool,
    ) -> None:
        if token is not _VERIFIED_QWEN35_HANDLE_TOKEN:
            raise TypeError("Verified Qwen3.5 package handles are created only by integrity factories.")
        self._snapshot = snapshot
        self.root = snapshot.root
        self.hf_model_path = hf_model_path
        self.apply_offset_norm = apply_offset_norm
        self._closed = False

    def _require_open(self, token: object) -> None:
        if token is not _VERIFIED_QWEN35_HANDLE_TOKEN or self._closed:
            raise RuntimeError("Verified Qwen3.5 package handle is invalid or already consumed.")

    def cleanup(self) -> None:
        if not self._closed:
            self._closed = True
            self._snapshot.cleanup()


def _verified_qwen35_model_path(handle: _VerifiedQwen35Package) -> Path:
    if not isinstance(handle, _VerifiedQwen35Package):
        raise TypeError("Qwen3.5 private load requires a verified package handle.")
    handle._require_open(_VERIFIED_QWEN35_HANDLE_TOKEN)
    return handle.hf_model_path


def is_qwen35_offset_norm_key(key: str) -> bool:
    """Return whether a raw Genesis weight needs Qwen3.5 unit-offset conversion."""
    return _QWEN35_OFFSET_NORM_KEY.fullmatch(key) is not None


def write_qwen35_weight_convention(hf_model: str | Path) -> None:
    """Stamp a canonical model tree whose RMSNorm tensors already use native Qwen semantics."""
    marker = Path(hf_model) / QWEN35_WEIGHT_CONVENTION_FILE
    temporary = marker.with_suffix(f"{marker.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": QWEN35_WEIGHT_CONVENTION_SCHEMA,
                "rmsnorm": QWEN35_NATIVE_RMSNORM_CONVENTION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)


def verify_qwen35_weight_convention(hf_model: str | Path) -> None:
    """Verify the checkpoint-owned native RMSNorm convention marker."""
    marker = Path(hf_model) / QWEN35_WEIGHT_CONVENTION_FILE
    if not marker.is_file() or marker.is_symlink():
        raise RuntimeError(f"Canonical Qwen3.5 weights require a regular {marker.name} marker.")
    try:
        payload = load_json_object(marker)
    except ValueError as exc:
        raise RuntimeError(f"Cannot read Qwen3.5 weight convention marker: {exc}") from exc
    _exact_keys(payload, {"schema", "rmsnorm"}, context="Qwen3.5 weight convention")
    if (
        payload["schema"] != QWEN35_WEIGHT_CONVENTION_SCHEMA
        or payload["rmsnorm"] != QWEN35_NATIVE_RMSNORM_CONVENTION
    ):
        raise RuntimeError("Qwen3.5 weight convention marker is unsupported.")


def safetensors_checkpoint_layout(
    checkpoint_dir: str | Path,
) -> tuple[list[Path], dict[str, set[str]] | None]:
    """Resolve one unambiguous safetensors layout and validate its authoritative index."""
    root = Path(checkpoint_dir)
    index_path = root / SAFETENSORS_INDEX_FILE
    single_path = root / SAFETENSORS_SINGLE_FILE
    discovered_shards = sorted(root.glob("model-*.safetensors"))

    if index_path.is_file():
        if single_path.is_file():
            raise ValueError(
                f"Checkpoint {root} contains both {SAFETENSORS_SINGLE_FILE} and an indexed sharded model."
            )
        index = load_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty safetensors weight_map in {index_path}.")
        advertised_by_shard: dict[str, set[str]] = {}
        for key, shard_name in weight_map.items():
            shard_path = Path(shard_name) if isinstance(shard_name, str) else Path()
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(shard_name, str)
                or not shard_name
                or shard_path.name != shard_name
                or not shard_name.startswith("model-")
                or not shard_name.endswith(".safetensors")
            ):
                raise ValueError(f"Invalid safetensors weight_map entry in {index_path}.")
            advertised_by_shard.setdefault(shard_name, set()).add(key)

        advertised_names = set(advertised_by_shard)
        discovered_names = {path.name for path in discovered_shards}
        missing = sorted(advertised_names - discovered_names)
        unindexed = sorted(discovered_names - advertised_names)
        if missing or unindexed:
            raise ValueError(
                f"Safetensors index shard inventory mismatch in {root}: "
                f"missing={missing}, unindexed shard files={unindexed}."
            )
        return [root / name for name in sorted(advertised_names)], advertised_by_shard

    if single_path.is_file():
        if discovered_shards:
            raise ValueError(
                f"Checkpoint {root} contains both model.safetensors and sharded model files: "
                f"{[path.name for path in discovered_shards]}."
            )
        return [single_path], None
    if discovered_shards:
        return discovered_shards, None
    raise FileNotFoundError(f"No safetensors shards found in {root}.")


def _exact_keys(value: dict[str, Any], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise RuntimeError(f"{context} keys mismatch: missing={missing}, unexpected={unexpected}.")


def _nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string.")
    return value


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{context} must be a positive integer.")
    return value


def _digest(value: Any, *, context: str) -> str:
    try:
        return require_sha256(value, context=context)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _verify_digest(path: Path, expected: Any, *, context: str) -> None:
    try:
        verify_file_sha256(path, expected, context=context)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _canonical_hf_model(package_root: str | Path, hf_model_dir: str | Path | None) -> tuple[Path, Path]:
    try:
        root = Path(package_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Qwen3.5 package root does not exist: {package_root}.") from exc
    hf_model = root / "hf_model"
    configured = Path(hf_model_dir) if hf_model_dir is not None else hf_model
    if not configured.is_absolute():
        configured = root / configured
    try:
        configured_hf_model = configured.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Configured Qwen3.5 hf_model directory does not exist: {configured}.") from exc
    if (
        not hf_model.is_dir()
        or hf_model.is_symlink()
        or configured.is_symlink()
        or configured_hf_model != hf_model.resolve(strict=True)
    ):
        raise RuntimeError("Qwen3.5 imported packages must load their canonical package-local hf_model.")
    return root, hf_model


def _validate_conditioning_deployment(value: Any, *, context: str) -> None:
    """Validate a v3 conditioning declaration: null, or a closed pair of boolean flags.

    Null records that the imported recipe required no declaration. That is a different
    fact from an explicitly conditioning-free deployment, so the two are never collapsed.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be null or an object.")
    _exact_keys(value, _CONDITIONING_DEPLOYMENT_KEYS, context=context)
    for key in sorted(_CONDITIONING_DEPLOYMENT_KEYS):
        if not isinstance(value[key], bool):
            raise RuntimeError(f"{context} {key} must be a boolean.")


def verify_qwen35_imported_package(
    package_root: str | Path,
    *,
    hf_model_dir: str | Path | None = None,
) -> None:
    """Verify every Qwen3.5 inference artifact bound by import provenance."""
    root, hf_model = _canonical_hf_model(package_root, hf_model_dir)
    if (hf_model / QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE).exists():
        raise RuntimeError(
            "Qwen3.5 imported packages cannot contain source_rmsnorm_conversion.json; "
            "that lineage marker is reserved for canonical trained packages."
        )
    provenance_path = root / QWEN35_IMPORT_PROVENANCE_FILE
    conversion_path = hf_model / QWEN35_CONVERSION_PROVENANCE_FILE
    try:
        provenance = load_json_object(provenance_path)
        conversion = load_json_object(conversion_path)
    except ValueError as exc:
        raise RuntimeError(f"Malformed Qwen3.5 import provenance: {exc}") from exc

    schema = provenance.get("schema")
    if not isinstance(schema, str) or schema not in _IMPORT_PROVENANCE_KEYS_BY_SCHEMA:
        raise RuntimeError(f"Unsupported Qwen3.5 import provenance at {provenance_path}.")
    _exact_keys(
        provenance,
        _IMPORT_PROVENANCE_KEYS_BY_SCHEMA[schema],
        context="Qwen3.5 import provenance",
    )
    if schema == QWEN35_IMPORT_PROVENANCE_SCHEMA:
        _validate_conditioning_deployment(
            provenance["conditioning_deployment"],
            context="Qwen3.5 import provenance conditioning_deployment",
        )
    authentication = provenance["contract_authentication"]
    if authentication not in _CONTRACT_AUTHENTICATION_VALUES:
        raise RuntimeError("Qwen3.5 import provenance has invalid contract_authentication.")
    if provenance["policy_state_identity_source"] != authentication:
        raise RuntimeError("Qwen3.5 import provenance identity-source fields disagree.")
    for field in (
        "dcp_identity_sha256",
        "source_artifact_manifest_sha256",
        "authenticated_import_identity_sha256",
    ):
        _digest(provenance[field], context=f"Qwen3.5 import provenance {field}")
    for field in ("policy_state_dataset", "normalization_scope"):
        _nonempty_string(provenance[field], context=f"Qwen3.5 import provenance {field}")
    if provenance["objective"] != "Flow":
        raise RuntimeError("Qwen3.5 import provenance objective must be 'Flow'.")
    if provenance["rmsnorm_conversion"] != conversion:
        raise RuntimeError(
            "Qwen3.5 package import provenance does not match hf_model/rmsnorm_conversion.json."
        )

    _exact_keys(conversion, _CONVERSION_PROVENANCE_KEYS, context="Qwen3.5 conversion provenance")
    if (
        conversion["schema"] != QWEN35_CONVERSION_PROVENANCE_SCHEMA
        or conversion["algorithm"] != QWEN35_CONVERSION_ALGORITHM
        or conversion["conversion_dtype"] != QWEN35_CONVERSION_DTYPE
    ):
        raise RuntimeError(f"Unsupported Qwen3.5 conversion provenance at {conversion_path}.")
    _positive_int(conversion["converted_key_count"], context="Qwen3.5 converted_key_count")
    _digest(conversion["converted_keys_sha256"], context="Qwen3.5 converted_keys_sha256")

    shard_records = conversion["shards"]
    if not isinstance(shard_records, list) or not shard_records:
        raise RuntimeError(f"Qwen3.5 conversion provenance has no model shards: {conversion_path}.")
    expected_shards: dict[str, tuple[str, str, int]] = {}
    for index, record in enumerate(shard_records):
        if not isinstance(record, dict):
            raise RuntimeError(f"Qwen3.5 conversion provenance shard record {index} is invalid.")
        _exact_keys(record, _SHARD_RECORD_KEYS, context=f"Qwen3.5 shard record {index}")
        filename = record["filename"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in expected_shards
            or not filename.endswith(".safetensors")
        ):
            raise RuntimeError(f"Qwen3.5 conversion provenance shard record {index} is invalid.")
        source_digest = _digest(record["source_sha256"], context=f"Qwen3.5 source digest for {filename}")
        output_digest = _digest(record["output_sha256"], context=f"Qwen3.5 output digest for {filename}")
        tensor_count = _positive_int(record["tensor_count"], context=f"Qwen3.5 tensor_count for {filename}")
        expected_shards[filename] = source_digest, output_digest, tensor_count

    try:
        shard_paths, advertised = safetensors_checkpoint_layout(hf_model)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Qwen3.5 model shard inventory is invalid: {exc}") from exc
    actual_shards = {path.name for path in shard_paths}
    missing_shards = sorted(set(expected_shards) - actual_shards)
    unlisted_shards = sorted(actual_shards - set(expected_shards))
    if missing_shards or unlisted_shards:
        raise RuntimeError(
            "Qwen3.5 model shard inventory disagrees with conversion provenance: "
            f"missing={missing_shards}, unlisted={unlisted_shards}."
        )
    from safetensors import safe_open

    converted_keys: list[str] = []
    seen_keys: set[str] = set()
    for shard_path in shard_paths:
        _source_digest, output_digest, declared_count = expected_shards[shard_path.name]
        _verify_digest(shard_path, output_digest, context="Qwen3.5 model shard")
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                keys = set(shard.keys())
        except Exception as exc:
            raise RuntimeError(f"Cannot inspect Qwen3.5 model shard {shard_path}: {exc}") from exc
        if advertised is not None and keys != advertised[shard_path.name]:
            raise RuntimeError(f"Qwen3.5 model shard {shard_path.name} disagrees with its index.")
        duplicates = seen_keys & keys
        if duplicates:
            raise RuntimeError(f"Qwen3.5 model shards contain duplicate tensors: {sorted(duplicates)[:10]}.")
        if len(keys) != declared_count:
            raise RuntimeError(
                f"Qwen3.5 tensor_count for {shard_path.name} is {declared_count}, found {len(keys)}."
            )
        converted_keys.extend(key for key in sorted(keys) if is_qwen35_offset_norm_key(key))
        seen_keys.update(keys)
    if (
        len(converted_keys) != conversion["converted_key_count"]
        or canonical_json_sha256(converted_keys) != conversion["converted_keys_sha256"]
    ):
        raise RuntimeError("Qwen3.5 converted-key audit metadata disagrees with model shard keys.")

    vocab = conversion["qwen35_vocab"]
    if (
        not isinstance(vocab, dict)
        or vocab.get("schema") != QWEN35_VOCAB_SCHEMA
        or vocab.get("algorithm") != QWEN35_VOCAB_ALGORITHM
    ):
        raise RuntimeError("Qwen3.5 conversion provenance has an invalid vocabulary record.")
    _exact_keys(vocab, _VOCAB_RECORD_KEYS, context="Qwen3.5 vocabulary record")
    _positive_int(vocab["token_count"], context="Qwen3.5 vocabulary token_count")
    tokenizer_digest = _digest(vocab["tokenizer_sha256"], context="Qwen3.5 tokenizer digest")
    vocab_digest = _digest(vocab["vocab_sha256"], context="Qwen3.5 vocabulary digest")
    _verify_digest(
        hf_model / "vocab.json",
        vocab_digest,
        context="Qwen3.5 conversion provenance",
    )
    try:
        vocab_payload = load_json_object(hf_model / "vocab.json")
    except ValueError as exc:
        raise RuntimeError(f"Malformed Qwen3.5 vocabulary: {exc}") from exc
    token_ids = list(vocab_payload.values())
    if (
        len(vocab_payload) != vocab["token_count"]
        or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids)
        or set(token_ids) != set(range(len(token_ids)))
    ):
        raise RuntimeError("Qwen3.5 vocabulary audit metadata or token IDs are invalid.")

    raw_source_hashes = provenance["source_hashes"]
    if not isinstance(raw_source_hashes, dict):
        raise RuntimeError("Qwen3.5 import provenance source_hashes must be an object.")
    source_hashes: dict[str, str] = {}
    for filename, value in raw_source_hashes.items():
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise RuntimeError(f"Qwen3.5 import provenance has invalid source filename {filename!r}.")
        source_hashes[filename] = _digest(value, context=f"Qwen3.5 source digest for {filename}")

    shard_names = set(expected_shards)
    index_name = {SAFETENSORS_INDEX_FILE} if (hf_model / SAFETENSORS_INDEX_FILE).is_file() else set()
    copied_assets = HF_RUNTIME_ASSET_FILENAMES - {"vocab.json"}
    expected_assets = {name for name in copied_assets if name in source_hashes}
    actual_assets = {name for name in copied_assets if (hf_model / name).is_file()}
    missing_assets = sorted(expected_assets - actual_assets)
    unlisted_assets = sorted(actual_assets - expected_assets)
    if missing_assets or unlisted_assets:
        raise RuntimeError(
            "Qwen3.5 copied runtime asset inventory disagrees with import provenance: "
            f"missing={missing_assets}, unlisted runtime assets={unlisted_assets}."
        )
    if not {"config.json", "tokenizer.json"}.issubset(expected_assets):
        raise RuntimeError("Qwen3.5 import provenance must bind config.json and tokenizer.json.")
    if source_hashes["tokenizer.json"] != tokenizer_digest:
        raise RuntimeError("Qwen3.5 import and vocabulary provenance disagree on tokenizer.json.")
    for filename in sorted(expected_assets):
        _verify_digest(
            hf_model / filename,
            source_hashes[filename],
            context="Qwen3.5 import provenance",
        )

    index_path = hf_model / SAFETENSORS_INDEX_FILE
    expected_index = SAFETENSORS_INDEX_FILE in source_hashes
    if bool(index_name) != expected_index:
        raise RuntimeError(
            "Qwen3.5 safetensors index inventory disagrees with import provenance: "
            f"expected={expected_index}, found={index_path.is_file()}."
        )
    if expected_index:
        _verify_digest(
            index_path,
            source_hashes[SAFETENSORS_INDEX_FILE],
            context="Qwen3.5 import provenance",
        )

    for filename, (source_digest, _output_digest, _tensor_count) in expected_shards.items():
        if source_hashes.get(filename) != source_digest:
            raise RuntimeError(
                f"Qwen3.5 import and conversion provenance disagree on source shard {filename}."
            )

    required_source_names = QWEN35_PACKAGED_SOURCE_FILENAMES | expected_assets | shard_names | index_name
    allowed_source_names = (
        required_source_names
        | QWEN35_OPTIONAL_PACKAGED_SOURCE_FILENAMES
        | ({"vocab.json"} if "vocab.json" in source_hashes else set())
    )
    missing_source_names = sorted(required_source_names - set(source_hashes))
    unexpected_source_names = sorted(set(source_hashes) - allowed_source_names)
    if missing_source_names or unexpected_source_names:
        raise RuntimeError(
            "Qwen3.5 source hash inventory is incomplete or unsupported: "
            f"missing={missing_source_names}, unexpected={unexpected_source_names}."
        )
    for filename in sorted(
        QWEN35_PACKAGED_SOURCE_FILENAMES | (QWEN35_OPTIONAL_PACKAGED_SOURCE_FILENAMES & set(source_hashes))
    ):
        _verify_digest(root / filename, source_hashes[filename], context="Qwen3.5 source contract")

    adapter = provenance["deployment_adapter"]
    if not isinstance(adapter, dict):
        raise RuntimeError("Qwen3.5 deployment_adapter provenance must be an object.")
    _exact_keys(adapter, _DEPLOYMENT_ADAPTER_KEYS, context="Qwen3.5 deployment adapter")
    if adapter["filename"] != "isaac_deployment_adapter.json":
        raise RuntimeError("Qwen3.5 deployment adapter filename is invalid.")
    _nonempty_string(adapter["validation_status"], context="Qwen3.5 adapter validation_status")
    _verify_digest(
        root / adapter["filename"],
        adapter["sha256"],
        context="Qwen3.5 deployment adapter",
    )

    fast_processor = provenance["fast_processor"]
    if not isinstance(fast_processor, dict):
        raise RuntimeError("Qwen3.5 fast_processor provenance must be an object.")
    _exact_keys(fast_processor, _FAST_PROCESSOR_KEYS, context="Qwen3.5 fast processor")
    _nonempty_string(fast_processor["repository"], context="Qwen3.5 fast processor repository")
    _nonempty_string(fast_processor["revision"], context="Qwen3.5 fast processor revision")
    _digest(fast_processor["tree_sha256"], context="Qwen3.5 fast processor tree_sha256")
    _positive_int(fast_processor["file_count"], context="Qwen3.5 fast processor file_count")


def validate_qwen35_hf_model_entries(
    hf_model: Path,
    *,
    allow_source_lineage: bool,
) -> set[str]:
    """Reject model-tree entries that the Qwen3.5 loader would never consume."""
    try:
        shard_paths, _advertised = safetensors_checkpoint_layout(hf_model)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Qwen3.5 model shard inventory is invalid: {exc}") from exc
    allowed_model_entries = {
        *HF_RUNTIME_ASSET_FILENAMES,
        QWEN35_CONVERSION_PROVENANCE_FILE,
        SAFETENSORS_INDEX_FILE,
        *(path.name for path in shard_paths),
    }
    if allow_source_lineage:
        allowed_model_entries.update(
            {
                QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE,
                QWEN35_WEIGHT_CONVENTION_FILE,
            }
        )
    actual_model_entries = {path.name for path in hf_model.iterdir()}
    unexpected = sorted(actual_model_entries - allowed_model_entries)
    if unexpected:
        raise RuntimeError(f"Qwen3.5 hf_model contains unsupported entries: {unexpected}.")
    non_files = sorted(
        path.name for path in hf_model.iterdir() if not path.is_file() and not path.is_symlink()
    )
    if non_files:
        raise RuntimeError(f"Qwen3.5 hf_model contains non-file entries: {non_files}.")
    return allowed_model_entries


def _snapshot_qwen35_package(
    package_root: str | Path,
    *,
    hf_model_dir: str | Path | None,
    full_package: bool,
    root_entries: set[str] | None = None,
    allow_source_lineage: bool = False,
) -> DirectorySnapshot:
    root, hf_model = _canonical_hf_model(package_root, hf_model_dir)
    allowed_model_entries = validate_qwen35_hf_model_entries(
        hf_model,
        allow_source_lineage=allow_source_lineage,
    )
    snapshot = snapshot_directory(
        root,
        prefix="lerobot-qwen35-",
        root_entries=root_entries if full_package else QWEN35_SNAPSHOT_ROOT_ENTRIES,
        directory_entries={"hf_model": allowed_model_entries},
    )
    return snapshot


def snapshot_qwen35_imported_package(
    package_root: str | Path,
    *,
    hf_model_dir: str | Path | None = None,
) -> DirectorySnapshot:
    """Copy then verify an imported package so later loads cannot race its source paths."""
    snapshot = _snapshot_qwen35_package(
        package_root,
        hf_model_dir=hf_model_dir,
        full_package=False,
    )
    try:
        verify_qwen35_imported_package(
            snapshot.root,
            hf_model_dir=snapshot.root / "hf_model",
        )
    except BaseException:
        snapshot.cleanup()
        raise
    return snapshot


def _snapshot_verified_qwen35_imported_package(
    package_root: str | Path,
    *,
    hf_model_dir: str | Path,
    full_package: bool = False,
) -> _VerifiedQwen35Package:
    """Create a private verified handle for one imported-package model load."""
    if full_package:
        root, _hf_model = _canonical_hf_model(package_root, hf_model_dir)
        root_entries = {
            path.name
            for path in root.iterdir()
            if path.name in QWEN35_IMPORTED_RUNTIME_ROOT_ENTRIES
            or path.name.startswith(("policy_preprocessor_step_", "policy_postprocessor_step_"))
        }
        unexpected = sorted({path.name for path in root.iterdir()} - root_entries)
        if unexpected:
            raise RuntimeError(f"Qwen3.5 imported package contains unbound root entries: {unexpected}.")
        snapshot = _snapshot_qwen35_package(
            root,
            hf_model_dir=hf_model_dir,
            full_package=True,
            root_entries=root_entries,
        )
        try:
            verify_qwen35_imported_package(snapshot.root, hf_model_dir=snapshot.root / "hf_model")
        except BaseException:
            snapshot.cleanup()
            raise
    else:
        snapshot = snapshot_qwen35_imported_package(package_root, hf_model_dir=hf_model_dir)
    return _VerifiedQwen35Package(
        _VERIFIED_QWEN35_HANDLE_TOKEN,
        snapshot,
        hf_model_path=snapshot.root / "hf_model",
        apply_offset_norm=False,
    )


def _snapshot_verified_qwen35_trained_package(
    package_root: str | Path,
    *,
    hf_model_dir: str | Path,
    verify_snapshot: Callable[[Path], tuple[str | Path, bool]],
) -> _VerifiedQwen35Package:
    """Snapshot a trained package, verify that copy, and return its exact model handle."""
    root, _hf_model = _canonical_hf_model(package_root, hf_model_dir)
    from .trained_package import (
        QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME,
        TRAINED_PACKAGE_INERT_HUB_METADATA,
    )

    try:
        manifest = load_json_object(root / QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME)
    except ValueError as exc:
        raise RuntimeError(f"Cannot inspect Qwen3.5 trained package manifest: {exc}") from exc
    files_sha256 = manifest.get("files_sha256")
    if not isinstance(files_sha256, dict):
        raise RuntimeError("Qwen3.5 trained package manifest files_sha256 must be an object.")
    root_entries = {
        "config.json",
        "hf_model",
        QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME,
        *(name for name in TRAINED_PACKAGE_INERT_HUB_METADATA if (root / name).exists()),
    }
    for relative in files_sha256:
        path = Path(relative) if isinstance(relative, str) else Path()
        if not isinstance(relative, str) or not relative or path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Qwen3.5 trained package manifest contains an unsafe file path.")
        root_entries.add(path.parts[0])
    unexpected_root_entries = sorted({path.name for path in root.iterdir()} - root_entries)
    if unexpected_root_entries:
        raise RuntimeError(
            f"Qwen3.5 trained package contains unbound root entries: {unexpected_root_entries}."
        )
    snapshot = _snapshot_qwen35_package(
        root,
        hf_model_dir=hf_model_dir,
        full_package=True,
        root_entries=root_entries,
        allow_source_lineage=True,
    )
    try:
        verified_hf_model, apply_offset_norm = verify_snapshot(snapshot.root)
        _root, canonical_hf_model = _canonical_hf_model(snapshot.root, verified_hf_model)
    except BaseException:
        snapshot.cleanup()
        raise
    return _VerifiedQwen35Package(
        _VERIFIED_QWEN35_HANDLE_TOKEN,
        snapshot,
        hf_model_path=canonical_hf_model,
        apply_offset_norm=bool(apply_offset_norm),
    )
