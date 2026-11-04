"""Shared outer-package manifests for trained Perceptron ISAAC checkpoints."""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint_integrity import (
    canonical_json_sha256,
    file_sha256,
    is_hf_snapshot_blob_link,
    load_json_object,
    require_sha256,
)
from .mk1_checkpoint_contract import SafetensorsInventory, read_and_validate_safetensors_index

if TYPE_CHECKING:
    from .configuration_perceptron_isaac import PerceptronIsaacConfig

TRAINED_PACKAGE_INERT_HUB_METADATA = frozenset({".gitattributes", "README.md"})
MK1_TRAINED_PACKAGE_MANIFEST_FILENAME = "mk1_trained_package_manifest.json"
MK1_TRAINED_PACKAGE_MANIFEST_SCHEMA = "perceptron_isaac_mk1_trained_package_manifest_v1"
MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD = "mk1_trained_package_manifest_sha256"
MK1_MODEL_IMPORT_FILENAME = "mk1_model_import.json"
MK1_CANONICAL_TRAINED_PACKAGE_MANIFEST_SCHEMA_V2 = (
    "perceptron_isaac_mk1_canonical_trained_package_manifest_v2"
)
MK1_CANONICAL_TRAINED_PACKAGE_MANIFEST_SCHEMA = "perceptron_isaac_mk1_canonical_trained_package_manifest_v3"
CANONICAL_INNER_MODEL_STORAGE = "canonical_inner"

PackageFilePredicate = Callable[[Path, Path], bool]

PROCESSOR_PACKAGE_PATHS = {
    "fast_processor_path": ("fast_processor", True),
    "native_render_metadata_path": ("native_render_metadata.json", False),
    "native_stats_path": ("isaac_stats.json", False),
    "suite_stats_path": ("isaac_suite_stats.json", False),
    "suite_by_task_index_path": ("isaac_suite_by_task_index.json", False),
}


def parse_trained_tensor_storage_dtypes(metadata: object) -> frozenset[str]:
    """Read the exact v3 storage declaration, never infer a missing declaration."""
    if not isinstance(metadata, dict) or set(metadata) != {"tensor_storage_dtypes"}:
        raise RuntimeError(
            "MK1 trained storage declaration metadata must contain only tensor_storage_dtypes."
        )
    values = metadata["tensor_storage_dtypes"]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or value not in {"BF16", "F32"} for value in values)
        or values != sorted(set(values))
    ):
        raise RuntimeError("MK1 tensor_storage_dtypes must be a sorted distinct nonempty BF16/F32 list.")
    return frozenset(values)


@dataclass(frozen=True)
class VerifiedTrainedTensorStorage:
    """Storage and source generation authenticated by the complete outer checkpoint.

    Inner descriptors retain producer lineage; a trained hf_model is not a standalone ABI.
    Precise shard identities avoid a second payload-hash pass; second-resolution
    sources are rehashed before lazy materialization.
    """

    model_root: Path
    dtypes: frozenset[str]
    source_inventory: SafetensorsInventory
    metadata_digests: Mapping[Path, str]
    coarse_shard_digests: Mapping[Path, str]

    def _verify_shards_unchanged(self) -> None:
        for shard in self.source_inventory.shards:
            self.source_inventory.verify_shard_identity(shard)

    def for_model(self, model_root: Path) -> frozenset[str]:
        if model_root.resolve(strict=True) != self.model_root:
            raise RuntimeError("Verified MK1 tensor storage belongs to a different model directory.")
        self.verify_source_unchanged()
        return self.dtypes

    def verify_source_unchanged(self) -> None:
        """Check the authenticated generation without rereading weight payloads."""
        self._verify_shards_unchanged()
        # These small files are read again by the loader; sidecar snapshots do not
        # authorize a changed config, tokenizer, or index at the weight-backed root.
        for path, expected in self.metadata_digests.items():
            try:
                actual = file_sha256(path)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Authenticated MK1 model metadata changed: {path.name}") from exc
            if not hmac.compare_digest(actual, expected):
                raise RuntimeError(f"Authenticated MK1 model metadata changed: {path.name}")

    def verify_for_materialization(self) -> None:
        """Rehash second-resolution shards; their timestamps can hide same-second writes.

        Precise identities keep the no-extra-payload-read path. Coarse sources incur
        one extra authenticated weight read, without copying or casting tensors.
        """
        self.verify_source_unchanged()
        for shard, expected in self.coarse_shard_digests.items():
            try:
                actual = file_sha256(shard)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Authenticated MK1 shard changed: {shard.name}") from exc
            if not hmac.compare_digest(actual, expected):
                raise RuntimeError(f"Authenticated MK1 shard digest mismatch: {shard.name}")
        if self.coarse_shard_digests:
            # Metadata must remain bound through the potentially long payload read.
            self.verify_source_unchanged()

    def require_header_dtypes(self, inventory: SafetensorsInventory) -> None:
        self._verify_shards_unchanged()
        actual = frozenset(header.dtype for header in inventory.tensors.values())
        if actual != self.dtypes:
            raise RuntimeError(
                "MK1 trained storage declaration disagrees with tensor header dtype set: "
                f"declared={sorted(self.dtypes)}, headers={sorted(actual)}."
            )


def _processor_configs(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_file()
        and path.name.startswith(("policy_preprocessor", "policy_postprocessor"))
        and path.suffix == ".json"
    ]


def rewrite_trained_processor_paths(package_root: str | Path) -> None:
    """Rewrite serialized ISAAC processor sidecars to package-relative paths in place."""
    root = Path(package_root)
    suite_available = all(
        (root / filename).is_file()
        for filename in (
            "isaac_suite_stats.json",
            "isaac_suite_by_task_index.json",
        )
    )

    def rewrite(value: Any, *, filename: str) -> Any:
        if isinstance(value, list):
            return [rewrite(item, filename=filename) for item in value]
        if not isinstance(value, dict):
            return value
        rewritten = {key: rewrite(item, filename=filename) for key, item in value.items()}
        for key in ("config_toml_path", "stats_path"):
            if key in rewritten:
                rewritten[key] = None
        for key, (relative, is_directory) in PROCESSOR_PACKAGE_PATHS.items():
            if key not in rewritten:
                continue
            if key.startswith("suite_"):
                rewritten[key] = relative if suite_available else None
                continue
            if rewritten[key] in (None, ""):
                continue
            candidate = root / relative
            available = candidate.is_dir() if is_directory else candidate.is_file()
            if not available:
                raise RuntimeError(
                    f"Serialized processor {filename} declares {key}, but package asset {relative} is missing."
                )
            rewritten[key] = relative
        return rewritten

    for config_path in _processor_configs(root):
        try:
            payload = load_json_object(config_path)
        except ValueError as exc:
            raise RuntimeError(f"Cannot rewrite serialized processor paths: {exc}") from exc
        rewritten = rewrite(payload, filename=config_path.name)
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(config_path)


def _outer_files(
    root: Path,
    *,
    manifest_filename: str,
    delegated_directory: str | None,
    label: str,
    is_package_file: PackageFilePredicate | None = None,
) -> dict[str, Path]:
    """Return the exact authenticated outer-file inventory."""
    files: dict[str, Path] = {}
    reserved = {"config.json", manifest_filename}
    for candidate in sorted(root.rglob("*")):
        relative_path = candidate.relative_to(root)
        relative = relative_path.as_posix()
        if relative in TRAINED_PACKAGE_INERT_HUB_METADATA:
            valid = (
                candidate.is_file() and not candidate.is_symlink()
                if is_package_file is None
                else (
                    (not candidate.is_symlink() and is_package_file(root, candidate))
                    or (
                        is_package_file(root, candidate)
                        and is_hf_snapshot_blob_link(candidate, snapshot_root=root)
                    )
                )
            )
            if not valid:
                qualifier = (
                    "regular root file"
                    if is_package_file is None
                    else ("regular package file or an immutable Hub blob link")
                )
                raise RuntimeError(f"{label} Hub metadata must be a {qualifier}: {relative}.")
            continue
        if relative in reserved or (
            delegated_directory is not None and relative_path.parts[0] == delegated_directory
        ):
            continue
        if candidate.is_symlink():
            if (
                is_package_file is None
                or not is_package_file(root, candidate)
                or not is_hf_snapshot_blob_link(candidate, snapshot_root=root)
            ):
                raise RuntimeError(
                    f"{label} contains a symlink outside its delegated model tree: {relative}."
                )
        elif candidate.is_dir():
            continue
        elif not candidate.is_file():
            raise RuntimeError(f"{label} contains a non-regular entry: {relative}.")
        files[relative] = candidate
    return files


def write_trained_package_manifest(
    package_root: str | Path,
    *,
    label: str,
    manifest_filename: str,
    manifest_schema: str,
    digest_field: str,
    identity: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    delegated_directory: str | None = "hf_model",
) -> str:
    """Write an exhaustive outer-package manifest and return its raw digest."""
    root = Path(package_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"{label} root must be a regular directory: {root}.")
    config_path = root / "config.json"
    manifest_path = root / manifest_filename
    if not config_path.is_file() or config_path.is_symlink():
        raise RuntimeError(f"{label} config.json must be a regular root file.")
    if manifest_path.is_symlink():
        raise RuntimeError(f"{label} manifest must not be a symlink: {manifest_path}.")
    try:
        raw_config = load_json_object(config_path)
    except ValueError as exc:
        raise RuntimeError(f"Cannot read {label} config.json: {exc}") from exc
    if digest_field not in raw_config:
        raise RuntimeError(f"{label} config is missing {digest_field}.")
    if any(raw_config.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f"{label} config identity disagrees with manifest generation inputs.")

    canonical_config = dict(raw_config)
    canonical_config[digest_field] = None
    files = _outer_files(
        root,
        manifest_filename=manifest_filename,
        delegated_directory=delegated_directory,
        label=label,
    )
    files_sha256 = {relative: file_sha256(path) for relative, path in files.items()}
    manifest = {
        "schema": manifest_schema,
        **identity,
        **(metadata or {}),
        "config_sha256": canonical_json_sha256(canonical_config),
        "files_sha256": files_sha256,
    }
    temporary = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return file_sha256(manifest_path)


def verify_trained_package_manifest(
    package_root: str | Path,
    *,
    label: str,
    manifest_filename: str,
    manifest_schema: str,
    digest_field: str,
    manifest_digest: Any,
    identity: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    is_package_file: PackageFilePredicate,
    delegated_directory: str | None = "hf_model",
) -> dict[str, Any]:
    """Verify the canonical config and every non-delegated package file."""
    root = Path(package_root).resolve(strict=True)
    try:
        expected_manifest_digest = require_sha256(
            manifest_digest,
            context=f"{label} {digest_field}",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    manifest_path = root / manifest_filename
    config_path = root / "config.json"

    def safe_reserved(path: Path) -> bool:
        return is_package_file(root, path) and (
            not path.is_symlink() or is_hf_snapshot_blob_link(path, snapshot_root=root)
        )

    if not safe_reserved(manifest_path):
        raise RuntimeError(f"{label} is missing {manifest_filename}.")
    if not safe_reserved(config_path):
        raise RuntimeError(f"{label} config.json is not a regular package file.")
    actual_manifest_digest = file_sha256(manifest_path)
    if not hmac.compare_digest(actual_manifest_digest, expected_manifest_digest):
        raise RuntimeError(
            f"{label} manifest digest mismatch: expected {expected_manifest_digest}, "
            f"found {actual_manifest_digest}."
        )
    try:
        manifest = load_json_object(manifest_path)
        raw_config = load_json_object(config_path)
    except ValueError as exc:
        raise RuntimeError(f"Cannot read {label} JSON: {exc}") from exc

    metadata = metadata or {}
    expected_keys = {"schema", *identity, *metadata, "config_sha256", "files_sha256"}
    # Historical ISAAC manifests retain the retired count in both authenticated JSONs.
    # It is not runtime identity, and is never removed before hashing either document.
    if "trained_steps" in manifest:
        expected_keys.add("trained_steps")
    if set(manifest) != expected_keys or manifest.get("schema") != manifest_schema:
        raise RuntimeError(f"{label} manifest fields do not match its schema.")
    if any(manifest.get(key) != value or raw_config.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f"{label} manifest identity disagrees with the policy config.")
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise RuntimeError(f"{label} manifest metadata is unsupported.")
    if raw_config.get(digest_field) != expected_manifest_digest:
        raise RuntimeError(f"{label} config does not bind the loaded manifest digest.")
    canonical_config = dict(raw_config)
    canonical_config[digest_field] = None
    try:
        expected_config_digest = require_sha256(
            manifest.get("config_sha256"),
            context=f"{label} config_sha256",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    actual_config_digest = canonical_json_sha256(canonical_config)
    if not hmac.compare_digest(actual_config_digest, expected_config_digest):
        raise RuntimeError(
            f"{label} canonical config digest mismatch: expected {expected_config_digest}, "
            f"found {actual_config_digest}."
        )

    raw_files = manifest.get("files_sha256")
    if not isinstance(raw_files, dict):
        raise RuntimeError(f"{label} files_sha256 must be an object.")
    expected_files: dict[str, str] = {}
    reserved = {"config.json", manifest_filename}
    for relative, digest in raw_files.items():
        path = Path(relative) if isinstance(relative, str) else Path()
        if (
            not isinstance(relative, str)
            or not relative
            or path.is_absolute()
            or path.as_posix() != relative
            or ".." in path.parts
            or relative in reserved
            or relative in TRAINED_PACKAGE_INERT_HUB_METADATA
            or (delegated_directory is not None and path.parts[0] == delegated_directory)
        ):
            raise RuntimeError(f"{label} files_sha256 contains an invalid path.")
        try:
            expected_files[relative] = require_sha256(
                digest,
                context=f"{label} files_sha256[{relative!r}]",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    actual_files = _outer_files(
        root,
        manifest_filename=manifest_filename,
        delegated_directory=delegated_directory,
        label=label,
        is_package_file=is_package_file,
    )
    if set(actual_files) != set(expected_files):
        raise RuntimeError(
            f"{label} file inventory disagrees with its manifest: "
            f"missing={sorted(set(expected_files) - set(actual_files))}, "
            f"unlisted={sorted(set(actual_files) - set(expected_files))}."
        )
    for relative, expected_digest in expected_files.items():
        actual_digest = file_sha256(actual_files[relative])
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise RuntimeError(
                f"{label} digest mismatch for {relative}: expected {expected_digest}, found {actual_digest}."
            )
    return manifest


def finalize_canonical_trained_package(
    config: PerceptronIsaacConfig,
    package_root: str | Path,
    *,
    mk1_source_model_import_sha256: str | None,
) -> str:
    """Bind native inner weights to their source identity and close the config digest cycle."""
    if not mk1_source_model_import_sha256:
        raise RuntimeError("Cannot finalize an MK1 package without source model-import identity.")
    root = Path(package_root).resolve(strict=True)
    inventory = read_and_validate_safetensors_index(root / "hf_model")
    storage_metadata = {"tensor_storage_dtypes": sorted({h.dtype for h in inventory.tensors.values()})}
    parse_trained_tensor_storage_dtypes(storage_metadata)
    config.mk1_trained_package_manifest_sha256 = None
    config._save_pretrained(root)
    digest = write_trained_package_manifest(
        root,
        label="MK1 canonical trained package",
        manifest_filename=MK1_TRAINED_PACKAGE_MANIFEST_FILENAME,
        manifest_schema=MK1_CANONICAL_TRAINED_PACKAGE_MANIFEST_SCHEMA,
        digest_field=MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
        identity={
            "artifact_kind": config.artifact_kind,
            "mk1_source_model_import_sha256": mk1_source_model_import_sha256,
        },
        metadata={"model_storage": CANONICAL_INNER_MODEL_STORAGE, "metadata": storage_metadata},
        delegated_directory=None,
    )
    config.mk1_trained_package_manifest_sha256 = digest
    config._save_pretrained(root)

    try:
        raw_config = load_json_object(root / "config.json")
        manifest = load_json_object(root / MK1_TRAINED_PACKAGE_MANIFEST_FILENAME)
    except ValueError as exc:
        raise RuntimeError(f"Cannot re-read finalized MK1 canonical trained package: {exc}") from exc
    canonical_config = dict(raw_config)
    canonical_config[MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD] = None
    if (
        raw_config.get(MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD) != digest
        or file_sha256(root / MK1_TRAINED_PACKAGE_MANIFEST_FILENAME) != digest
        or manifest.get("config_sha256") != canonical_json_sha256(canonical_config)
    ):
        raise RuntimeError(
            "MK1 canonical trained package config/manifest binding changed before publication."
        )
    return digest
