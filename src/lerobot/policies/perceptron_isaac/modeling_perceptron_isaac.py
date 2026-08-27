"""Perceptron Isaac VLA policy for LeRobot native eval/serving.

LeRobot processors own observation normalization, mharmony packing, stats
normalization, and action unnormalization. The policy loads the HF VLA, owns
live rollout window state for online serving, invokes ``sample_action`` on
native mharmony streams, and returns normalized action chunks.
"""

from __future__ import annotations

import builtins
import hmac
import json
import logging
import shutil
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from huggingface_hub import save_torch_state_dict
from torch import Tensor
from transformers import AutoModelForCausalLM

from lerobot.lerobot_types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.hub import resolve_hub_snapshot

from ..pretrained import PreTrainedPolicy
from .checkpoint_integrity import (
    file_sha256,
    is_hf_snapshot_blob_link,
    load_json_object,
    require_sha256,
    snapshot_directory,
)
from .configuration_perceptron_isaac import PerceptronIsaacConfig
from .isaac_stats import (
    IsaacNormalizationStats,
    normalize_isaac_actions,
    require_closed_loop_safe_isaac_stats,
)
from .loss_perceptron_isaac import (
    ISAAC_FLOW_LOSS_MASK_KEY,
    ISAAC_FLOW_NOISE_KEY,
    ISAAC_FLOW_TIMESTEPS_KEY,
    perceptron_isaac_loss_denominators,
    perceptron_isaac_training_loss,
)
from .mharmony_native import (
    IsaacMharmonyRenderMetadata,
    load_native_isaac_stats,
    load_native_render_metadata,
)
from .processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY,
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_PREVIOUS_ACTIONS_KEY,
    PERCEPTRON_ISAAC_RENDER_META_KEY,
    PERCEPTRON_ISAAC_SERVING_STATS_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacMharmonyPackProcessorStep,
)
from .trained_package import (
    AUTHENTICATED_IMPORT_MODEL_STORAGE,
    CANONICAL_INNER_MODEL_STORAGE,
    MK1_CANONICAL_TRAINED_PACKAGE_MANIFEST_SCHEMA,
    MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
    MK1_TRAINED_PACKAGE_MANIFEST_FILENAME,
    MK1_TRAINED_PACKAGE_MANIFEST_SCHEMA,
    PROCESSOR_PACKAGE_PATHS,
    QWEN35_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
    QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME,
    QWEN35_TRAINED_PACKAGE_MANIFEST_SCHEMA,
    finalize_canonical_trained_package,
    rewrite_trained_processor_paths,
    stage_trained_processor_assets,
    verify_trained_package_manifest,
)

T = TypeVar("T", bound="PerceptronIsaacPolicy")
logger = logging.getLogger(__name__)
_MK1_BASE_VOCAB_SIZE = 248_044


def _load_json_object(
    path: Path,
    label: str,
    duplicate_context: str,
    object_error: str | None = None,
) -> dict[str, Any]:
    try:
        return load_json_object(path)
    except ValueError as exc:
        message = str(exc)
        if "duplicate JSON key" in message or "contains duplicate key" in message:
            marker = "duplicate JSON key" if "duplicate JSON key" in message else "contains duplicate key"
            duplicate = message.split(marker, 1)[1].split(" in JSON", 1)[0].rstrip(".")
            raise RuntimeError(f"{duplicate_context} contains duplicate key{duplicate}.") from exc
        if "must contain an object" in message:
            raise RuntimeError(object_error or f"{label} must contain a JSON object.") from exc
        raise RuntimeError(f"Cannot read {label}: {exc}") from exc


def _verify_sha256(
    expected: Any,
    context: str,
    actual: str | Path | None = None,
    mismatch_error: str | Callable[[str, str], str] | None = None,
) -> str:
    try:
        expected = require_sha256(expected, context=context)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if isinstance(actual, Path):
        actual = file_sha256(actual)
    if actual is not None and not hmac.compare_digest(actual, expected):
        if callable(mismatch_error):
            mismatch_error = mismatch_error(expected, actual)
        raise RuntimeError(mismatch_error or f"{context} mismatch: expected {expected}, found {actual}.")
    return expected


def _is_portable_isaac05_repository(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and config.get("model_type") == "isaac_0_5"


class PerceptronIsaacPolicy(PreTrainedPolicy):
    """LeRobot training/eval wrapper for checkpoint-selected native ISAAC VLAs."""

    name = "perceptron_isaac"
    config_class = PerceptronIsaacConfig

    def __init__(self, config: PerceptronIsaacConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._isaac_model = None
        self._stats = None
        self._native_render_metadata = None
        self._verified_runtime_assets = None
        self._verified_external_runtime_assets = []
        self._verified_processor_assets = None
        self._verified_external_processor_assets = []
        # Checkpoint-serialized normalization stats adopted from the restored
        # preprocessor pipeline (see _maybe_adopt_serving_stats). Set once per loaded
        # checkpoint and deliberately NOT cleared by reset(): the checkpoint identity
        # does not change between episodes, only per-episode rollout state does.
        self._adopted_serving_stats: IsaacNormalizationStats | None = None
        # Set once, outside reset(): parameter freezing must happen before the model is
        # wrapped for distributed training and must not be re-applied per step.
        self._training_parameters_configured = False

        self.reset()

        # Pre-load only when inference dtype (bf16) is also the training storage dtype.
        # With fp32 storage the load is deferred so the backbone is never materialized twice;
        # both ``predict_action_chunk`` and ``forward`` load lazily on first use.
        if config.hf_model_path and not config.train_storage_fp32:
            self._load_backbone()

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        """Save one canonical trained weight inventory plus checkpoint-local runtime assets."""
        source_root = (
            self._verified_runtime_assets.root
            if self._verified_runtime_assets is not None
            else self._local_checkpoint_source_root()
        )
        if source_root is None:
            raise RuntimeError("Perceptron ISAAC dense saves require a loaded checkpoint package.")
        with self._verified_checkpoint_source(source_root) as verified_source:
            self._copy_checkpoint_package(verified_source, save_directory)
        self.config._export_apply_offset_norm = False
        if self._config_declares_mk1(self.config):
            self.config._export_mk1_source_model_import_sha256 = (
                self.config.mk1_model_import_sha256 or self.config.mk1_source_model_import_sha256
            )
            self.config._export_mk1_model_import_sha256 = None
        self.config._save_pretrained(save_directory)
        self._save_canonical_inner_weights(save_directory, state_dict)
        self._canonical_inner_save_root = str(save_directory.resolve())

    def _local_checkpoint_source_root(self) -> Path | None:
        source_value = getattr(self.config, "_checkpoint_source_root", None)
        source_root = Path(str(source_value)) if source_value else None
        if source_root is None or not source_root.is_dir():
            pretrained_path = getattr(self.config, "pretrained_path", None)
            candidate = Path(str(pretrained_path)) if pretrained_path else None
            if candidate is not None and candidate.is_dir():
                source_root = candidate
        return source_root.resolve() if source_root is not None else None

    @staticmethod
    def _is_checkpoint_weight_name(name: str) -> bool:
        return (
            name == "model.safetensors"
            or name == "model.safetensors.index.json"
            or (name.startswith("model-") and name.endswith(".safetensors"))
        )

    def _processor_snapshot_entries(self, root: Path) -> set[str]:
        """Return adapter-root entries needed to restore both processor pipelines."""
        entries: set[str] = set()
        relative_paths = dict(getattr(self.config, "_checkpoint_relative_paths", {}))

        def trusted_processor_path(key: str, raw: str) -> Path | None:
            path = Path(raw)
            if not path.parts or ".." in path.parts:
                raise RuntimeError(f"Serialized processor path {key} escapes its package: {raw}.")
            if not path.is_absolute():
                return root / path
            record = relative_paths.get(key)
            if not isinstance(record, dict):
                raise RuntimeError(f"Serialized processor path {key} is untrusted: {raw}.")
            relative = record.get("relative")
            owner = record.get("root")
            absolute = record.get("absolute")
            if not all(isinstance(value, str) for value in (relative, owner, absolute)):
                raise RuntimeError(f"Serialized processor path {key} is untrusted: {raw}.")
            relative_path = Path(relative)
            if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
                raise RuntimeError(f"Serialized processor path {key} is untrusted: {raw}.")
            owner_root = Path(owner)
            expected = owner_root / relative_path
            if str(expected) != absolute or absolute != raw:
                raise RuntimeError(f"Serialized processor path {key} is untrusted: {raw}.")
            # Adapter-owned assets belong in its processor snapshot. Verified
            # base-owned assets are retained separately by records_by_owner.
            return expected if owner_root == root else None

        def collect(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    collect(item)
                return
            if not isinstance(value, dict):
                return
            state_file = value.get("state_file")
            if isinstance(state_file, str) and state_file.strip():
                state_path = Path(state_file)
                if state_path.is_absolute() or not state_path.parts or ".." in state_path.parts:
                    raise RuntimeError(f"Serialized processor state_file escapes its package: {state_file}.")
                if not (root / state_path).is_file():
                    raise RuntimeError(f"Serialized processor state_file is missing: {state_file}.")
                entries.add(state_path.parts[0])
            for key, item in value.items():
                if key in PROCESSOR_PACKAGE_PATHS and isinstance(item, str) and item.strip():
                    path = trusted_processor_path(key, item)
                    if path is not None and path.exists():
                        entries.add(path.relative_to(root).parts[0])
                collect(item)

        for path in sorted(root.iterdir()):
            if not (
                path.is_file()
                and path.name.startswith(("policy_preprocessor", "policy_postprocessor"))
                and path.suffix == ".json"
            ):
                continue
            entries.add(path.name)
            try:
                collect(load_json_object(path))
            except ValueError as exc:
                raise RuntimeError(f"Cannot retain serialized processor config: {exc}") from exc
        if (root / "policy_normalization.json").is_file():
            entries.add("policy_normalization.json")
        return entries

    def retain_pretrained_processor_assets(self, checkpoint_root: str | Path) -> None:
        """Keep adapter processors and their sidecars alive beyond a mutable source path."""
        root = Path(checkpoint_root).resolve(strict=True)
        entries = self._processor_snapshot_entries(root)
        relative_paths = dict(getattr(self.config, "_checkpoint_relative_paths", {}))
        processor_path_attrs = set(PROCESSOR_PACKAGE_PATHS) | {"config_toml_path", "stats_path"}
        records_by_owner: dict[Path, list[tuple[str, dict[str, str]]]] = {}
        for attr, record in relative_paths.items():
            if attr not in processor_path_attrs or not isinstance(record, dict):
                continue
            relative = record.get("relative")
            owner_value = record.get("root")
            if not isinstance(relative, str) or not isinstance(owner_value, str):
                continue
            owner = Path(owner_value).resolve()
            records_by_owner.setdefault(owner, []).append((attr, record))
            if owner == root:
                entries.add(Path(relative).parts[0])

        snapshot = snapshot_directory(root, prefix="lerobot-isaac-processors-", root_entries=entries)
        external_snapshots = []
        replacements: dict[str, str] = {}
        try:
            updated_paths = dict(relative_paths)
            for owner, records in records_by_owner.items():
                if owner == root:
                    owner_snapshot = snapshot
                else:
                    owner_snapshot = snapshot_directory(
                        owner,
                        prefix="lerobot-isaac-processor-sidecars-",
                        root_entries={Path(record["relative"]).parts[0] for _, record in records},
                    )
                    external_snapshots.append(owner_snapshot)
                for attr, record in records:
                    candidate = owner_snapshot.root / record["relative"]
                    if not candidate.exists():
                        raise RuntimeError(f"Retained processor path {attr} disappeared from its package.")
                    absolute = str(candidate)
                    replacements[str(record.get("absolute", ""))] = absolute
                    setattr(self.config, attr, absolute)
                    updated_paths[attr] = {
                        "absolute": absolute,
                        "relative": record["relative"],
                        "root": str(owner_snapshot.root),
                    }

            stable_paths = {
                key: str(getattr(self.config, key))
                for key in processor_path_attrs
                if getattr(self.config, key, None)
            }

            def rewrite(value: Any) -> Any:
                if isinstance(value, list):
                    return [rewrite(item) for item in value]
                if not isinstance(value, dict):
                    return replacements.get(value, value) if isinstance(value, str) else value
                return {
                    key: stable_paths[key] if key in stable_paths and isinstance(item, str) else rewrite(item)
                    for key, item in value.items()
                }

            for config_path in sorted(snapshot.root.glob("policy_*processor*.json")):
                payload = rewrite(load_json_object(config_path))
                config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            self.config._checkpoint_relative_paths = updated_paths
            self.config._pretrained_processor_root = str(snapshot.root)
        except BaseException:
            snapshot.cleanup()
            for external in external_snapshots:
                external.cleanup()
            raise

        previous = self._verified_processor_assets
        previous_external = self._verified_external_processor_assets
        self._verified_processor_assets = snapshot
        self._verified_external_processor_assets = external_snapshots
        if previous is not None:
            previous.cleanup()
        for external in previous_external:
            external.cleanup()

    def _retain_verified_runtime_assets(self, verified_root: Path) -> None:
        """Keep verified sidecars stable without retaining a second copy of model weights."""
        root = Path(verified_root)
        root_entries = {
            path.name
            for path in root.iterdir()
            if path.name != "base_model" and not self._is_checkpoint_weight_name(path.name)
        }
        model_root = root / "hf_model"
        model_entries = (
            {path.name for path in model_root.iterdir() if not self._is_checkpoint_weight_name(path.name)}
            if model_root.is_dir()
            else set()
        )
        snapshot = snapshot_directory(
            root,
            prefix="lerobot-isaac-runtime-",
            root_entries=root_entries,
            directory_entries={"hf_model": model_entries} if "hf_model" in root_entries else None,
        )
        external_snapshots = []
        try:
            relative_paths = dict(getattr(self.config, "_checkpoint_relative_paths", {}))
            source_root_value = getattr(self.config, "_checkpoint_source_root", None)
            source_root = Path(str(source_root_value)).resolve() if source_root_value else root.resolve()
            records_by_owner: dict[Path, list[tuple[str, dict[str, str]]]] = {}
            for attr, record in relative_paths.items():
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("absolute"), str)
                    or not isinstance(record.get("relative"), str)
                ):
                    continue
                owner_value = record.get("root")
                owner = Path(owner_value).resolve() if isinstance(owner_value, str) else source_root
                records_by_owner.setdefault(owner, []).append((attr, record))

            rebound_paths: dict[str, dict[str, str]] = {}
            for owner, records in records_by_owner.items():
                if owner == source_root:
                    owner_snapshot = snapshot
                else:
                    owner_snapshot = snapshot_directory(
                        owner,
                        prefix="lerobot-isaac-owned-runtime-",
                        root_entries={Path(record["relative"]).parts[0] for _, record in records},
                    )
                    external_snapshots.append(owner_snapshot)
                for attr, record in records:
                    candidate = owner_snapshot.root / record["relative"]
                    if not candidate.exists():
                        raise RuntimeError(
                            f"Retained checkpoint-local path {attr} disappeared from its owning package."
                        )
                    absolute = str(candidate)
                    setattr(self.config, attr, absolute)
                    rebound_paths[attr] = {
                        "absolute": absolute,
                        "relative": record["relative"],
                        "root": str(owner_snapshot.root),
                    }
            self.config._checkpoint_relative_paths = rebound_paths
            self._set_checkpoint_local_vocab_path(self.config, force=True)
        except BaseException:
            snapshot.cleanup()
            for external in external_snapshots:
                external.cleanup()
            raise
        previous = self._verified_runtime_assets
        previous_external = self._verified_external_runtime_assets
        self._verified_runtime_assets = snapshot
        self._verified_external_runtime_assets = external_snapshots
        if previous is not None:
            previous.cleanup()
        for external in previous_external:
            external.cleanup()

    @contextmanager
    def _verified_checkpoint_source(self, source_root: Path):
        """Yield one stable, verified package root and release its private snapshot."""
        runtime_assets = self._verified_runtime_assets
        if runtime_assets is not None and source_root == runtime_assets.root:
            yield runtime_assets.root
            return
        if self._config_declares_mk1(self.config):
            from lerobot.configs import PreTrainedConfig

            with snapshot_directory(source_root, prefix="lerobot-mk1-save-") as snapshot:
                source_config = PreTrainedConfig.from_pretrained(snapshot.root)
                self._resolve_checkpoint_local_paths(source_config, snapshot.root)
                self._verify_packaged_contract_digests(source_config, snapshot.root)
                yield snapshot.root
            return

        verified_package = getattr(self.config, "_qwen35_verified_package", None)
        if verified_package is None:
            from lerobot.configs import PreTrainedConfig

            with snapshot_directory(source_root, prefix="lerobot-qwen35-save-") as snapshot:
                source_config = PreTrainedConfig.from_pretrained(snapshot.root)
                self._resolve_checkpoint_local_paths(source_config, snapshot.root)
                if source_config.qwen35_trained_package_manifest_sha256 is not None:
                    self._verify_qwen35_trained_package_manifest(source_config, snapshot.root)
                else:
                    self._verify_packaged_contract_digests(source_config, snapshot.root)
                nested_verified = getattr(source_config, "_qwen35_verified_package", None)
                if nested_verified is None:
                    yield snapshot.root
                    return
                try:
                    yield nested_verified.root
                finally:
                    nested_verified.cleanup()
            return
        try:
            yield verified_package.root
        finally:
            verified_package.cleanup()
            if getattr(self.config, "_qwen35_verified_package", None) is verified_package:
                delattr(self.config, "_qwen35_verified_package")

    def _save_canonical_inner_weights(
        self,
        save_directory: Path,
        state_dict: dict[str, Tensor] | None,
    ) -> None:
        policy_state = self.state_dict() if state_dict is None else state_dict
        prefixes = ("_isaac_model.", "module._isaac_model.")
        prefix = next(
            (item for item in prefixes if policy_state and all(key.startswith(item) for key in policy_state)),
            None,
        )
        if prefix is None:
            unexpected = sorted(key for key in policy_state if not key.startswith(prefixes))
            raise RuntimeError(
                "Perceptron ISAAC dense state must contain only native backbone tensors; "
                f"unexpected={unexpected[:10]}."
            )
        inner_state = {key.removeprefix(prefix): tensor for key, tensor in policy_state.items()}
        model_root = save_directory / "hf_model"
        if not model_root.is_dir() or model_root.is_symlink():
            raise RuntimeError("Perceptron ISAAC dense save is missing a regular checkpoint-local hf_model.")
        for directory in (save_directory, model_root):
            for weight_path in (
                directory / "model.safetensors",
                directory / "model.safetensors.index.json",
                *directory.glob("model-*.safetensors"),
            ):
                weight_path.unlink(missing_ok=True)
        save_torch_state_dict(inner_state, model_root, max_shard_size="5GB")
        if self._config_declares_mk1(self.config):
            from .mk1_checkpoint_contract import (
                MK1_TEST_GEOMETRY_MARKER,
                Mk1CheckpointContractError,
                finalize_mk1_runtime_checkpoint_layout,
                validate_mk1_checkpoint,
            )

            finalize_mk1_runtime_checkpoint_layout(model_root)
            try:
                inner_config = load_json_object(model_root / "config.json")
                validate_mk1_checkpoint(
                    model_root,
                    allow_test_only_reduced_geometry=bool(inner_config.get(MK1_TEST_GEOMETRY_MARKER, False)),
                )
            except (ValueError, Mk1CheckpointContractError) as exc:
                raise RuntimeError(f"Saved MK1 runtime checkpoint is invalid: {exc}") from exc
            lineage = (
                (
                    save_directory / "isaac_import_provenance.json",
                    save_directory / "source_isaac_import_provenance.json",
                ),
                (model_root / "mk1_model_import.json", model_root / "source_mk1_model_import.json"),
            )
        else:
            from .qwen35_checkpoint import (
                QWEN35_CONVERSION_PROVENANCE_FILE,
                QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE,
                write_qwen35_weight_convention,
            )

            write_qwen35_weight_convention(model_root)
            lineage = (
                (
                    save_directory / "isaac_import_provenance.json",
                    save_directory / "source_isaac_import_provenance.json",
                ),
                (
                    save_directory / "isaac_export_provenance.json",
                    save_directory / "source_isaac_export_provenance.json",
                ),
                (
                    model_root / QWEN35_CONVERSION_PROVENANCE_FILE,
                    model_root / QWEN35_SOURCE_CONVERSION_PROVENANCE_FILE,
                ),
            )
        for source, destination in lineage:
            if source.is_file():
                destination.unlink(missing_ok=True)
                source.replace(destination)

    @classmethod
    def _verify_saved_package(cls, root: str | Path) -> PerceptronIsaacConfig:
        from lerobot.configs import PreTrainedConfig

        saved_config = PreTrainedConfig.from_pretrained(root)
        cls._resolve_checkpoint_local_paths(saved_config, root)
        if (
            not cls._config_declares_mk1(saved_config)
            and saved_config.qwen35_trained_package_manifest_sha256 is not None
        ):
            cls._verify_qwen35_trained_package_manifest(saved_config, Path(root))
        else:
            cls._verify_packaged_contract_digests(saved_config, Path(root))
        return saved_config

    def finalize_pretrained_package(self, save_directory: str | Path) -> None:
        """Bind final package contents, or embed a local PEFT base, after all writers finish."""
        root = Path(save_directory)
        adapter_config_path = root / "adapter_config.json"
        if self.config.use_peft and adapter_config_path.is_file():
            stage_trained_processor_assets(root, self.config)
        rewrite_trained_processor_paths(root)
        if self.config.use_peft and adapter_config_path.is_file():
            self._finalize_local_peft_base(root, adapter_config_path)
            return
        if getattr(self, "_canonical_inner_save_root", None) != str(root.resolve()):
            return
        if self._config_declares_mk1(self.config):
            model_import_digest = getattr(
                self.config,
                "_export_mk1_source_model_import_sha256",
                self.config.mk1_source_model_import_sha256,
            )
            finalize_canonical_trained_package(
                self.config,
                root,
                family="mk1",
                mk1_source_model_import_sha256=model_import_digest,
            )
            return
        finalize_canonical_trained_package(self.config, root, family="qwen35")

    def hub_delete_patterns(self) -> str:
        """Mirror the exact authenticated ISAAC package inventory on every Hub push."""
        return "*"

    def hub_ignore_patterns(self) -> None:
        """Upload the exact file inventory authenticated by the outer package manifest."""
        return None

    def _finalize_local_peft_base(self, root: Path, adapter_config_path: Path) -> None:
        try:
            adapter_config = load_json_object(adapter_config_path)
        except ValueError as exc:
            raise RuntimeError(f"Cannot finalize PEFT adapter config: {exc}") from exc

        def write_adapter_config() -> None:
            temporary = adapter_config_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(adapter_config_path)

        embed_base = getattr(self.config, "_embed_base_on_save", None)
        if embed_base is False:
            original_reference = getattr(self.config, "_peft_base_reference", None)
            if not original_reference:
                raise RuntimeError("Remote PEFT base provenance is missing its Hub repo ID.")
            embedded = root / "base_model"
            if embedded.is_symlink() or (embedded.exists() and not embedded.is_dir()):
                raise RuntimeError("Stale embedded PEFT base_model is not a regular directory.")
            if embedded.is_dir():
                shutil.rmtree(embedded)
            adapter_config["base_model_name_or_path"] = str(original_reference)
            adapter_config["revision"] = getattr(self.config, "_peft_base_revision", None)
            write_adapter_config()
            return

        source_root = self._local_checkpoint_source_root()
        if source_root is None:
            raise RuntimeError("PEFT package finalization requires the resolved base checkpoint package.")
        embedded = root / "base_model"
        if embedded.is_symlink() or (embedded.exists() and not embedded.is_dir()):
            raise RuntimeError("Existing embedded PEFT base_model must be a regular directory.")
        with self._verified_checkpoint_source(source_root) as verified_source:
            if embedded.is_dir():
                self._verify_saved_package(embedded)
                if self._package_file_digests(verified_source) != self._package_file_digests(embedded):
                    raise RuntimeError(
                        "Existing embedded PEFT base does not match the configured source package."
                    )
            else:
                shutil.copytree(verified_source, embedded)
        self._verify_saved_package(embedded)
        adapter_config["base_model_name_or_path"] = "base_model"
        adapter_config["revision"] = None
        write_adapter_config()

    @staticmethod
    def _copy_checkpoint_package(source_root: Path, destination: Path) -> None:
        """Materialize runtime assets without copying either stale trained weight inventory."""
        if source_root == destination.resolve():
            return

        def ignore(directory: str, names: list[str]) -> set[str]:
            current = Path(directory).resolve()
            at_package_root = current == source_root
            at_model_root = current == source_root / "hf_model"
            return {
                name
                for name in names
                if (at_package_root and name == "config.json")
                or (
                    (at_package_root or at_model_root)
                    and (
                        name == "model.safetensors"
                        or name == "model.safetensors.index.json"
                        or (name.startswith("model-") and name.endswith(".safetensors"))
                    )
                )
            }

        shutil.copytree(source_root, destination, dirs_exist_ok=True, ignore=ignore)

    @staticmethod
    def _package_file_digests(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def adapt_async_observation(
        self, observation: dict[str, Any], *, timestep: int, fps: float
    ) -> dict[str, Any]:
        """Anchor the policy's prompt clock to episode time for chunk-rate serving.

        The async policy server runs one predict per chunk (~1/s at 30 actions),
        so the policy's internal frame clock (one frame per predict call) would
        advance ~30x slower than episode time. The observation timestep (index
        of the last executed action) over the control fps is the episode-relative,
        monotonic model time the checkpoint was trained with. Called by the
        policy server through a policy-independent hook before preprocessing.
        """
        adapted = dict(observation)
        adapted[PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY] = [max(int(timestep), 0) / float(fps)]
        return adapted

    def validate_async_features(self, lerobot_features: dict[str, Any]) -> None:
        """Validate the robot handshake against the checkpoint contract at load.

        Fails the async server's readiness path immediately (instead of at the
        first render) when mharmony is absent or the robot's state/action width,
        camera count, or image size do not match the serialized checkpoint.
        """
        from .mharmony_adapter import assert_mharmony_available

        assert_mharmony_available()

        problems: list[str] = []

        state = lerobot_features.get(OBS_STATE)
        if state is None:
            problems.append(f"missing {OBS_STATE} feature")
        elif tuple(state.get("shape", ())) != (self.config.proprio_dim,):
            problems.append(
                f"{OBS_STATE} shape {tuple(state.get('shape', ()))} != ({self.config.proprio_dim},)"
            )

        action = lerobot_features.get(ACTION)
        if action is None:
            problems.append("missing action feature in handshake")
        elif tuple(action.get("shape", ())) != (self.config.action_dim,):
            problems.append(f"action shape {tuple(action.get('shape', ()))} != ({self.config.action_dim},)")

        image_keys = sorted(key for key in lerobot_features if key.startswith(OBS_IMAGES))
        if len(image_keys) != len(self.config.camera_order):
            problems.append(
                f"expected {len(self.config.camera_order)} camera features "
                f"({self.config.camera_order}), got {image_keys}"
            )
        height, width = (int(value) for value in self.config.image_size)
        for key in image_keys:
            shape = tuple(lerobot_features[key].get("shape", ()))
            if shape not in ((height, width, 3), (3, height, width)):
                problems.append(f"{key} shape {shape} incompatible with image_size ({height}, {width})")

        if problems:
            raise ValueError("perceptron_isaac checkpoint contract validation failed: " + "; ".join(problems))

    def reset(self):
        """Clear per-episode action state."""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._online_image_queue: deque[np.ndarray] = deque(maxlen=max(1, int(self.config.n_obs_steps)))
        self._online_state_queue: deque[np.ndarray] = deque(maxlen=max(1, int(self.config.n_obs_steps)))
        self._online_action_queue: deque[np.ndarray | None] = deque(
            maxlen=max(1, int(self.config.n_obs_steps))
        )
        self._pending_executed_action: np.ndarray | None = None
        self._record_action_for_history = False
        self._online_render_step: PerceptronIsaacMharmonyPackProcessorStep | None = None
        self._last_norm_chunk: Tensor | None = None
        self._frame_index: int = -1
        # Settle frames are counted separately so they never advance the episode clock.
        self._settle_index: int = 0
        self._last_external_timestamp_seconds: float | None = None
        self._flow_seed_index: int = -1

    def record_executed_action(self, action: Tensor | np.ndarray) -> None:
        """Record one postprocessed action for the next causal observation."""
        if not self.config.action_conditioning or not self._record_action_for_history:
            return
        if self._pending_executed_action is not None:
            raise RuntimeError("record_executed_action called twice before the next observation.")
        self._ensure_native_metadata()
        action_array = (
            action.detach().cpu().numpy() if isinstance(action, Tensor) else np.asarray(action)
        ).astype(np.float32, copy=False)
        action_vector = action_array.reshape(-1)
        expected_shape = (int(self.config.action_dim),)
        if action_vector.shape != expected_shape:
            raise ValueError(f"Executed action must have shape {expected_shape}, got {action_array.shape}.")
        if not np.isfinite(action_vector).all():
            raise ValueError("Executed action must contain only finite values.")
        normalized = normalize_isaac_actions(action_vector[None, :], self._stats.action)[0]
        clip_max = float(self.config.fast_clip_normalized_max)
        self._pending_executed_action = np.clip(normalized, -clip_max, clip_max).astype(
            np.float32, copy=False
        )
        self._record_action_for_history = False

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Return shell optimizer groups after enforcing the native training guard."""
        self._require_native_training_supported()
        if self._isaac_model is None:
            self._load_backbone(training=True)
        self._configure_training_parameters()
        if self.config.use_peft:
            # PEFT is applied in-place to this lazy inner model. Reassert the final
            # contract here, immediately before optimizer groups are collected.
            for name, parameter in self._isaac_model.named_parameters():
                if "action_expert" in name:
                    parameter.requires_grad_("modules_to_save.default" in name)
                elif "lora_" in name:
                    parameter.requires_grad_(True)
        self._promote_trainable_parameters_to_fp32(self._isaac_model)
        groups: dict[str, list[Tensor]] = {"vlm": [], "vision": [], "expert": []}
        for name, parameter in self._isaac_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "action_expert" in name:
                groups["expert"].append(parameter)
            elif name.startswith("model.visual."):
                groups["vision"].append(parameter)
            else:
                groups["vlm"].append(parameter)
        result: list[dict[str, Any]] = []
        for name, lr in (
            ("vlm", self.config.optimizer_lr),
            ("vision", self.config.optimizer_vit_lr),
            ("expert", self.config.optimizer_action_expert_lr),
        ):
            if groups[name]:
                result.append({"params": groups[name], "lr": lr})
        return result

    def wrap_with_peft(self, peft_config=None, peft_cli_overrides: dict | None = None):
        """Materialize the lazy ISAAC backbone before PEFT discovers target modules."""
        if self._isaac_model is None:
            self._load_backbone(training=False)
        peft_model = super().wrap_with_peft(peft_config=peft_config, peft_cli_overrides=peft_cli_overrides)
        # ISAAC configures trainability while lazily loading its native backbone. Make the
        # final PEFT contract explicit: adapters and exactly one PEFT-owned expert copy.
        for name, parameter in peft_model.named_parameters():
            if "action_expert" in name:
                parameter.requires_grad_("modules_to_save.default" in name)
            elif "lora_" in name:
                parameter.requires_grad_(True)
        self._promote_trainable_parameters_to_fp32(peft_model)
        return peft_model

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """LoRA the VLM/vision linears and fully train/save the action expert."""
        vlm_linear_leaves = (
            r"q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
            r"in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj|proj|"
            r"linear_fc1|linear_fc2|qkv"
        )
        return {
            "target_modules": (
                rf"_isaac_model\.model\.(?:language_model|visual)\..*\.({vlm_linear_leaves})$"
            ),
            # The outer and inner modules are both named ``action_expert``. Target the
            # fully-qualified inner module so PEFT does not recursively clone both.
            "modules_to_save": ["_isaac_model.model.action_expert.action_expert"],
        }

    def _promote_trainable_parameters_to_fp32(self, module: torch.nn.Module) -> None:
        """Keep optimizer-owned parameters in FP32 without expanding the frozen backbone."""
        if not self.config.train_storage_fp32:
            return
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32:
                # This runs before optimizer/FSDP/DDP construction, so replacing parameter storage is safe.
                parameter.data = parameter.data.to(dtype=torch.float32)

        non_fp32 = [
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if non_fp32:
            raise RuntimeError(f"ISAAC trainable parameters must use FP32 storage; first: {non_fp32[:5]}")

    def _resolve_device(self) -> str:
        """Per-rank CUDA device under accelerate multi-GPU, else config.device."""
        import os

        d = str(self.config.device)
        if d.startswith("cuda") and os.environ.get("LOCAL_RANK") is not None and torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return d

    def _model_dtype(self) -> torch.dtype:
        """Dtype of the loaded backbone, falling back to bf16 for parameterless stand-ins."""
        try:
            return next(p.dtype for p in self._isaac_model.parameters() if p.is_floating_point())
        except (AttributeError, StopIteration):
            return torch.bfloat16

    @staticmethod
    def _config_declares_mk1(config: PerceptronIsaacConfig) -> bool:
        """Select MK1 from authenticated metadata or the inner checkpoint config."""
        if (
            config.artifact_kind == "neutral_debug"
            or config.mk1_model_import_sha256 is not None
            or config.mk1_source_model_import_sha256 is not None
            or config.mk1_trained_package_manifest_sha256 is not None
        ):
            return True

        model_path = getattr(config, "hf_model_path", None)
        config_path = Path(str(model_path)) / "config.json" if model_path else None
        if config_path is not None and config_path.is_file():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return False
            if not isinstance(raw, dict):
                return False
            if raw.get("model_type") == "isaac_0_5":
                return True
            genesis_vla = raw.get("genesis_vla")
            return bool(
                raw.get("model_type") == "qwen3_5_moe"
                and isinstance(genesis_vla, dict)
                and genesis_vla.get("backbone_family") == "mk1_qwen3_6_moe"
            )

        return False

    def _load_backbone(self, *, training: bool = False) -> None:
        if not self.config.hf_model_path:
            raise RuntimeError("PerceptronIsaacConfig.hf_model_path is required to load the Isaac HF VLA.")
        model_dir = Path(self.config.hf_model_path)
        if _is_portable_isaac05_repository(model_dir):
            if training:
                self._require_native_training_supported()
            device = self._resolve_device()
            self._isaac_model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map={"": str(device)},
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            return
        if not self._config_declares_mk1(self.config):
            from .modeling_qwen35_vla import (
                _load_qwen35_vla_from_verified_package,
                load_qwen35_vla_from_hf,
            )

            # Serving knobs only. A checkpoint-stamped action_expert block is
            # authoritative for expert geometry; historical checkpoints fall back
            # to the loader's default contract.
            action_expert_overrides = {
                "action_horizon": int(self.config.chunk_size),
                "num_inference_steps": int(self.config.num_inference_steps),
            }
            dense_fp32_training = (
                training and self.config.train_storage_fp32 and not self.config.train_expert_only
            )
            model_dtype = torch.float32 if dense_fp32_training else torch.bfloat16
            model_path = Path(self.config.hf_model_path).resolve()
            verified_package = getattr(self.config, "_qwen35_verified_package", None)
            load_options = {
                "vector_max_states": self.config.vector_max_states,
                "action_expert_overrides": action_expert_overrides,
                "dtype": model_dtype,
                "device": self._resolve_device(),
                "strict": True,
                "apply_offset_norm": bool(getattr(self.config, "apply_offset_norm", True)),
            }
            if verified_package is None:
                model, _ = load_qwen35_vla_from_hf(model_path, **load_options)
            else:
                try:
                    self._retain_verified_runtime_assets(verified_package.root)
                except BaseException:
                    verified_package.cleanup()
                    delattr(self.config, "_qwen35_verified_package")
                    raise
                delattr(self.config, "_qwen35_verified_package")
                model, _ = _load_qwen35_vla_from_verified_package(
                    verified_package,
                    action_expert=None,
                    **load_options,
                )
            self._isaac_model = model
            if training:
                self._configure_training_parameters()
            return

        if training:
            self._require_native_training_supported()
        from .mk1_checkpoint_contract import Mk1CheckpointContractError, validate_mk1_checkpoint
        from .modeling_mk1_vla import TORCH_SDPA_ATTENTION_BACKEND, load_mk1_vla_from_hf
        from .modeling_qwen36_moe import DETERMINISTIC_ROUTE_REDUCTION

        model_dtype = torch.bfloat16
        model_dir = Path(self.config.hf_model_path)
        allow_test_geometry = self.config.artifact_kind == "neutral_debug"
        try:
            contract, _ = validate_mk1_checkpoint(
                model_dir,
                allow_test_only_reduced_geometry=allow_test_geometry,
            )
        except Mk1CheckpointContractError as exc:
            raise RuntimeError(f"ISAAC requires a valid MK1 composite checkpoint: {exc}") from exc
        if (
            self.config.artifact_kind != contract.artifact.artifact_kind
            or self.config.trained_steps != contract.artifact.trained_steps
        ):
            raise RuntimeError(
                "ISAAC policy/checkpoint artifact identity mismatch: "
                f"policy=({self.config.artifact_kind!r}, {self.config.trained_steps}), "
                f"checkpoint=({contract.artifact.artifact_kind!r}, {contract.artifact.trained_steps})."
            )
        if self.config.vector_max_states != contract.vector_encoder.max_states:
            raise RuntimeError(
                "ISAAC policy/checkpoint vector width mismatch: "
                f"policy={self.config.vector_max_states}, checkpoint={contract.vector_encoder.max_states}."
            )
        if self.config.max_action_dim != contract.action_expert.action_dim:
            raise RuntimeError(
                "ISAAC policy/checkpoint maximum action width mismatch: "
                f"policy={self.config.max_action_dim}, checkpoint={contract.action_expert.action_dim}."
            )
        if self.config.max_action_horizon != contract.action_expert.action_horizon:
            raise RuntimeError(
                "ISAAC policy/checkpoint maximum action horizon mismatch: "
                f"policy={self.config.max_action_horizon}, "
                f"checkpoint={contract.action_expert.action_horizon}."
            )
        if self.config.action_dim > contract.action_expert.action_dim:
            raise RuntimeError(
                "ISAAC policy action_dim exceeds the checkpoint maximum: "
                f"policy={self.config.action_dim}, checkpoint={contract.action_expert.action_dim}."
            )
        if self.config.chunk_size > contract.action_expert.action_horizon:
            raise RuntimeError(
                "ISAAC policy chunk_size exceeds the checkpoint maximum: "
                f"policy={self.config.chunk_size}, checkpoint={contract.action_expert.action_horizon}."
            )
        if bool(getattr(self.config, "apply_offset_norm", False)):
            raise RuntimeError("MK1 checkpoints forbid the legacy apply_offset_norm conversion.")
        try:
            render_metadata = load_native_render_metadata(self.config)
            contract.validate_coord_reserved_token_groups(render_metadata.mharmony_reserved_token_groups)
        except (ValueError, Mk1CheckpointContractError) as exc:
            raise RuntimeError(f"ISAAC MK1 coordinate/rendering contract mismatch: {exc}") from exc
        model, _, loaded_contract = load_mk1_vla_from_hf(
            model_dir,
            dtype=model_dtype,
            device=self._resolve_device(),
            allow_test_only_reduced_geometry=allow_test_geometry,
            attention_backend=TORCH_SDPA_ATTENTION_BACKEND,
            route_reduction=DETERMINISTIC_ROUTE_REDUCTION,
        )
        if loaded_contract != contract:
            raise RuntimeError("MK1 checkpoint contract changed between preflight and model loading.")
        self._isaac_model = model
        self._native_render_metadata = render_metadata
        if training:
            self._configure_training_parameters()

    def _configure_training_parameters(self) -> None:
        if self._isaac_model is None:
            raise RuntimeError("ISAAC backbone must be loaded before training setup.")
        # Freezing is a one-time setup step that must happen before the model is wrapped.
        # ``forward`` also calls this on every training step, and once FSDP has flat-sharded
        # the parameters they are no longer leaf tensors, so re-applying ``requires_grad_``
        # raises "you can only change requires_grad flags of leaf variables".
        if self._training_parameters_configured:
            return
        self._training_parameters_configured = True
        from .modeling_qwen35_vla import setup_qwen35_vla_for_training

        setup_qwen35_vla_for_training(
            self._isaac_model,
            train_expert_only=bool(self.config.train_expert_only),
            train_samples_per_chunk=int(self.config.train_samples_per_chunk),
            rtc_max_delay_steps=int(self.config.flow_rtc_max_delay_steps),
            rtc_delay_sampling=str(self.config.flow_rtc_delay_sampling),
            dual_timestep_ratio=float(self.config.flow_dual_timestep_ratio),
            mask_padded_action_rows=bool(self.config.flow_mask_padded_action_rows),
        )
        if self.config.freeze_input_embeddings:
            input_embeddings = self._isaac_model.get_input_embeddings()
            if input_embeddings is None:
                raise RuntimeError("freeze_input_embeddings=true but ISAAC exposes no input embeddings.")
            lm_head_params = {id(param) for param in self._isaac_model.lm_head.parameters()}
            if any(id(param) in lm_head_params for param in input_embeddings.parameters()):
                raise RuntimeError("ISAAC input embeddings unexpectedly share parameters with lm_head.")
            for parameter in input_embeddings.parameters():
                # Defensive: a parameter already wrapped by FSDP/DDP is a non-leaf view and
                # cannot take a requires_grad flag. The freeze applies before wrapping.
                if parameter.is_leaf:
                    parameter.requires_grad_(False)
        self._promote_trainable_parameters_to_fp32(self._isaac_model)

    def _require_native_training_supported(self) -> None:
        if self.config.hf_model_path and self._config_declares_mk1(self.config):
            raise RuntimeError(
                "Isaac-0.5 training is not supported yet: production training requires FP32 "
                "parameter storage, while the grouped_mm null-MoE dispatcher requires BF16. "
                "Use the training repository for now; LeRobot Isaac-0.5 remains inference-only."
            )

    def _maybe_adopt_serving_stats(self, batch: dict[str, Any]) -> None:
        """Adopt checkpoint-serialized normalization stats handed over by the pack step.

        At online single-step serving the saved preprocessor skips rendering (the
        anchor is injected only after preprocessing) and attaches its restored stats
        to the batch instead. Those serialized stats are the training-time ground
        truth, so every policy-owned consumer -- the online render step and the
        settle branch -- must use them in preference to independently path-loading
        ``config.native_stats_path`` (which, for PEFT finetunes, historically pointed
        at the *base* package and normalized proprio in the wrong frame).
        """
        stats = batch.get(PERCEPTRON_ISAAC_SERVING_STATS_KEY)
        if stats is None or stats is self._adopted_serving_stats:
            return
        if not isinstance(stats, IsaacNormalizationStats):
            raise TypeError(
                f"{PERCEPTRON_ISAAC_SERVING_STATS_KEY!r} must carry IsaacNormalizationStats, "
                f"got {type(stats).__name__}."
            )
        require_closed_loop_safe_isaac_stats(stats)
        self._adopted_serving_stats = stats
        self._stats = stats
        # Render metadata may derive its clock from the stats; rebuild it lazily.
        self._native_render_metadata = None
        step = self._online_render_step
        if step is not None and step._stats is not stats:
            step._stats = stats
            step._stream_builder = None

    def _ensure_native_metadata(self) -> None:
        if self._stats is None:
            self._stats = self._adopted_serving_stats or load_native_isaac_stats(self.config)
        if self._native_render_metadata is None:
            self._native_render_metadata = load_native_render_metadata(
                self.config,
                stats_fps=self._stats.target_fps,
            )

    def _ensure_renderer(self) -> None:
        """Compatibility hook for callers/tests: rendering is owned by processors."""
        self._ensure_native_metadata()

    def _require_deployable_action_artifact(self) -> None:
        if self.config.artifact_kind != "trained_policy":
            raise RuntimeError(
                "Perceptron Isaac action serving rejects neutral_debug checkpoints. "
                "This zero-step artifact validates construction and parity only and must never control a robot."
            )

    def _ensure_online_render_step(self) -> PerceptronIsaacMharmonyPackProcessorStep:
        """Create the stateless pack/render processor used after policy-owned online windowing."""
        device = self._resolve_device()
        if self._online_render_step is None:
            # Forward the FULL normalization/render contract, mirroring
            # make_perceptron_isaac_pre_post_processors. Dropping fields here is not
            # benign: a missing normalize_gripper/feature-name pair silently
            # normalizes gripper dimensions the checkpoint expects raw, and a missing
            # target_fps disables the wrong-stats-file fps check.
            self._online_render_step = PerceptronIsaacMharmonyPackProcessorStep(
                enabled=True,
                inference_backend=str(self.config.inference_backend),
                config_toml_path=self.config.config_toml_path,
                stats_path=self.config.stats_path,
                native_render_metadata_path=self.config.native_render_metadata_path,
                native_stats_path=self.config.native_stats_path,
                fast_processor_path=self.config.fast_processor_path,
                fast_processor_tree_sha256=self.config.fast_processor_tree_sha256,
                mharmony_version=self.config.mharmony_version,
                action_dim=int(self.config.action_dim),
                proprio_dim=int(self.config.proprio_dim),
                vector_max_states=int(self.config.vector_max_states),
                dataset_name=str(self.config.dataset_name),
                robot_type=str(self.config.robot_type),
                action_conditioning=bool(self.config.action_conditioning),
                action_conditioning_role=str(self.config.action_conditioning_role),
                mistake_conditioning=bool(self.config.mistake_conditioning),
                camera_order=list(self.config.camera_order),
                image_size=tuple(self.config.image_size),
                image_preprocessing=str(self.config.image_preprocessing),
                image_keys=[f"observation.images.{cam}" for cam in self.config.camera_order],
                allow_image_key_fallback=bool(self.config.allow_image_key_fallback),
                normalize_language=bool(self.config.normalize_task_text),
                n_obs_steps=int(self.config.n_obs_steps),
                chunk_size=int(self.config.chunk_size),
                objective=str(self.config.objective),
                patch_size=int(self.config.render_patch_size),
                max_num_patches=self.config.render_max_num_patches,
                min_num_patches=self.config.render_min_num_patches,
                pixel_shuffle_scale=int(self.config.render_pixel_shuffle_scale),
                temporal_patch_size=int(self.config.render_temporal_patch_size),
                device=device,
                dtype="bfloat16",
                target_fps=self.config.target_fps,
                normalization_profile_id=self.config.normalization_profile_id,
                normalization_profile_scope=self.config.normalization_profile_scope,
                normalization_validation_status=self.config.normalization_validation_status,
                render_metadata=(
                    IsaacMharmonyRenderMetadata.from_json_file(
                        self.config.native_render_metadata_path
                    ).to_json_dict()
                    if self.config.native_render_metadata_path
                    and Path(self.config.native_render_metadata_path).is_file()
                    else None
                ),
                train_clip_normalized_actions=bool(self.config.train_clip_normalized_actions),
                clip_normalized_max=float(self.config.clip_normalized_max),
                fast_clip_normalized_max=float(self.config.fast_clip_normalized_max),
                train_skip_outlier_threshold=float(self.config.train_skip_outlier_threshold),
                train_max_sequence_length=int(self.config.train_max_sequence_length),
                normalize_gripper=bool(self.config.normalize_gripper),
                action_feature_names=list(self.config.action_feature_names or []) or None,
                state_feature_names=list(self.config.state_feature_names or []) or None,
            )
            if self._adopted_serving_stats is not None:
                # The checkpoint's serialized stats outrank any config-path load,
                # and must survive the per-Ready() reset that rebuilds this step.
                self._online_render_step._stats = self._adopted_serving_stats
        else:
            self._online_render_step.device = device
        return self._online_render_step

    def _update_online_rollout_state(self, batch: dict[str, Any]) -> None:
        """Append the current live observation to the policy-owned history window."""
        render_step = self._ensure_online_render_step()
        batch_size = render_step._infer_batch_size(batch, None)
        if batch_size != 1:
            raise NotImplementedError("Perceptron Isaac online rollout state supports batch_size=1 only.")
        images = render_step._extract_image_tensor(batch, batch_size)
        states = render_step._extract_state_tensor(batch, batch_size)
        if images.shape[:2] != states.shape[:2]:
            raise ValueError(f"Image/state window shape mismatch: {images.shape[:2]} vs {states.shape[:2]}.")
        if self.config.action_conditioning and images.shape[1] != 1:
            raise ValueError(
                "Action-conditioned online rollout requires one observation per select_action call."
            )
        for idx in range(images.shape[1]):
            self._online_image_queue.append(images[0, idx])
            self._online_state_queue.append(states[0, idx])
            if self.config.action_conditioning:
                self._online_action_queue.append(
                    None if self._frame_index == 0 else self._pending_executed_action
                )
                self._pending_executed_action = None
            else:
                self._online_action_queue.append(None)

    def _online_anchor_timestamp_seconds(self) -> float:
        self._ensure_native_metadata()
        fps = (
            getattr(self._native_render_metadata, "target_fps", None)
            or getattr(self._stats, "target_fps", None)
            or 20.0
        )
        fps = float(fps) if float(fps) > 0.0 else 20.0
        if self._frame_index < 0:
            raise RuntimeError("Perceptron Isaac model time requested before the first episode frame.")
        # Floor the anchor at (k-1)/fps like Genesis does, so the k rendered observation
        # timestamps stay distinct instead of collapsing to duplicates at episode start.
        n_obs_steps = int(getattr(self._native_render_metadata, "n_obs_steps", 1) or 1)
        min_anchor_steps = max(0, n_obs_steps - 1)
        return float(max(self._frame_index, min_anchor_steps)) / fps

    def _external_anchor_timestamp_seconds(self, batch: dict[str, Any]) -> float | None:
        """Read and validate an explicitly supplied episode-relative model time.

        The policy clock remains frame based for normal online rollout. Offline
        replay may override it with an explicit timestamp, but processors never
        synthesize or retain time and RNG state is not used as a clock.
        """
        raw = None
        for key in (
            PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY,
            "anchor_timestamp_seconds",
            "timestamp_seconds",
            "timestamp",
        ):
            if batch.get(key) is not None:
                raw = batch[key]
                break
        if raw is None:
            return None
        if torch.is_tensor(raw):
            values = raw.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
        else:
            values = np.asarray(raw, dtype=np.float64).reshape(-1).tolist()
        if len(values) != 1:
            raise ValueError(
                f"Perceptron Isaac online timestamp override requires exactly one value; got {len(values)}."
            )
        timestamp = float(values[0])
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                f"Perceptron Isaac timestamp override must be finite and nonnegative; got {timestamp!r}."
            )
        previous = self._last_external_timestamp_seconds
        if previous is not None and timestamp < previous:
            raise ValueError(
                "Perceptron Isaac timestamp override must be monotonic within an episode; "
                f"got {timestamp} after {previous}."
            )
        self._last_external_timestamp_seconds = timestamp
        return timestamp

    @staticmethod
    def _online_task_source(batch: dict[str, Any]) -> Any:
        task = batch.get("task")
        if task is None:
            task = batch.get("observation.language")
        if task is None:
            task = batch.get("language_instruction")
        return task

    def _render_online_stream_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Render a stream from policy-owned online history and explicit policy-owned anchor timing."""
        if not self._online_image_queue or not self._online_state_queue:
            raise RuntimeError("Perceptron Isaac online render requested before any observation was queued.")
        render_step = self._ensure_online_render_step()
        image_keys = render_step._resolve_image_keys(batch)
        image_values = list(self._online_image_queue)
        state_values = list(self._online_state_queue)
        action_values = list(self._online_action_queue)
        target_steps = max(1, int(self.config.n_obs_steps))
        if len(image_values) < target_steps:
            pad_count = target_steps - len(image_values)
            image_values = [image_values[0]] * pad_count + image_values
            state_values = [state_values[0]] * pad_count + state_values
            action_values = [None] * pad_count + action_values
        image_window = np.stack(image_values, axis=0)
        state_window = np.stack(state_values, axis=0)
        observation = {
            image_key: image_window[None, :, camera_idx] for camera_idx, image_key in enumerate(image_keys)
        }
        observation[OBS_STATE] = state_window[None, ...]
        task_source = self._online_task_source(batch)
        if task_source is not None:
            observation["task"] = task_source
        external_anchor = self._external_anchor_timestamp_seconds(batch)
        complementary = {
            "task": task_source,
            PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY: [
                external_anchor if external_anchor is not None else self._online_anchor_timestamp_seconds()
            ],
            PERCEPTRON_ISAAC_PREVIOUS_ACTIONS_KEY: [action_values],
        }
        transition = {
            TransitionKey.OBSERVATION: observation,
            TransitionKey.ACTION: None,
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.INFO: {},
            TransitionKey.COMPLEMENTARY_DATA: complementary,
        }
        rendered = render_step(transition)
        rendered_complementary = rendered[TransitionKey.COMPLEMENTARY_DATA]
        out = dict(batch)
        out[PERCEPTRON_ISAAC_STREAM_KEY] = rendered_complementary[PERCEPTRON_ISAAC_STREAM_KEY]
        out[PERCEPTRON_ISAAC_RENDER_META_KEY] = rendered_complementary[PERCEPTRON_ISAAC_RENDER_META_KEY]
        out["task"] = rendered_complementary.get("task", task_source)
        return out

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        action_prefix: Tensor | None = None,
        prefix_length: int | None = None,
        **kwargs,
    ) -> Tensor:
        """Run the flow expert on a native mharmony stream.

        Returns normalized actions shaped ``[B, chunk_size, action_dim]``. The
        postprocessor is responsible for action unnormalization and clipping.
        """
        self._require_deployable_action_artifact()
        # Fail fast on the RTC inference engine: it calls this method directly (bypassing
        # select_action's online history + settle clock) and passes RTC-specific kwargs that this
        # policy does not implement. The MolmoAct expert is not RTC-trained; use --inference.type=sync.
        if "inference_delay" in kwargs or "prev_chunk_left_over" in kwargs:
            raise NotImplementedError(
                "perceptron_isaac does not support RTC inference (the MolmoAct expert is not "
                "RTC-trained and this policy owns its own observation clock). Use "
                "--inference.type=sync."
            )
        self.eval()
        if self._isaac_model is None:
            self._load_backbone()
        model = self._isaac_model
        assert model is not None
        self._maybe_adopt_serving_stats(batch)
        self._ensure_native_metadata()

        stream = batch.get(PERCEPTRON_ISAAC_STREAM_KEY)
        if stream is None:
            self._frame_index += 1
            self._update_online_rollout_state(batch)
            batch = self._render_online_stream_batch(batch)
            stream = batch[PERCEPTRON_ISAAC_STREAM_KEY]
        # Follow the loaded model's dtype rather than assuming bf16: an fp32 backbone
        # (train_storage_fp32) is evaluated in-process during training.
        stream = stream.to(device=self._resolve_device(), dtype=self._model_dtype())

        n_flow = max(1, int(getattr(self.config, "num_flow_samples", 1) or 1))

        def _sample() -> Tensor:
            return model.sample_action(
                stream,
                num_steps=self.config.num_inference_steps,
                action_dim=self.config.action_dim,
                num_action_steps=self.config.chunk_size,
                action_prefix=action_prefix,
                prefix_length=prefix_length,
                num_flow_samples=n_flow,
            )

        def _run_samples() -> Tensor:
            device = torch.device(self._resolve_device())
            compute_dtype = getattr(torch, self.config.dtype)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=compute_dtype)
                if device.type in {"cuda", "cpu"} and compute_dtype in {torch.bfloat16, torch.float16}
                else nullcontext()
            )
            with autocast_context:
                return _sample()

        import os

        seed_base = self.config.flow_seed_base
        if seed_base is None:
            seed_base_env = os.environ.get("ISAAC_FLOW_SEED_BASE")
            seed_base = int(seed_base_env) if seed_base_env is not None else None
        if seed_base is not None:
            self._flow_seed_index = int(getattr(self, "_flow_seed_index", -1)) + 1
            seed = int(seed_base) + self._flow_seed_index
            dev = torch.device(self._resolve_device())
            use_cuda = dev.type == "cuda" and torch.cuda.is_available()
            if use_cuda:
                device_index = dev.index if dev.index is not None else torch.cuda.current_device()
                with torch.random.fork_rng(devices=[device_index]), torch.cuda.device(device_index):
                    torch.cuda.manual_seed(seed)
                    actions_norm = _run_samples()
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    actions_norm = _run_samples()
        else:
            actions_norm = _run_samples()

        if not bool(torch.isfinite(actions_norm).all()):
            raise RuntimeError("Perceptron Isaac model returned non-finite normalized actions.")
        self._last_norm_chunk = actions_norm[0].detach()
        return actions_norm[:, :, : self.config.action_dim].float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs) -> Tensor:
        """Serve one normalized action per call; refill chunks through ``predict_action_chunk``."""
        self._require_deployable_action_artifact()
        self.eval()
        if ACTION in batch:
            batch = {k: v for k, v in batch.items() if k != ACTION}

        # Adopt before the settle branch below: idle actions are normalized with
        # policy-level stats, which must be the checkpoint's serialized stats rather
        # than a config-path load.
        self._maybe_adopt_serving_stats(batch)

        has_stream = PERCEPTRON_ISAAC_STREAM_KEY in batch
        settle = int(getattr(self.config, "num_settle_steps", 0) or 0)
        # Genesis settles inside the environment reset and counts only executed post-settle
        # steps in its model clock, so idle settle frames must not advance the episode clock.
        if self._settle_index < settle:
            self._settle_index += 1
            # Idle settle frames must not enter the model observation window. The first
            # post-settle observation starts the policy-owned history and is padded there.
            self._ensure_native_metadata()
            idle = np.zeros((1, self.config.action_dim), dtype=np.float32)
            idle[:, -1] = float(self.config.settle_gripper)
            idle_norm = normalize_isaac_actions(idle, self._stats.action)
            return torch.from_numpy(idle_norm).float()

        if (
            self.config.action_conditioning
            and not has_stream
            and self._frame_index >= 0
            and self._pending_executed_action is None
        ):
            raise RuntimeError(
                "Action-conditioned rollout requires record_executed_action after each selected action."
            )
        self._frame_index += 1
        if not has_stream:
            self._update_online_rollout_state(batch)

        if len(self._queues[ACTION]) == 0:
            action_prefix, prefix_length = None, None
            rtc_p = int(getattr(self.config, "rtc_prefix_length", 0) or 0)
            prev = getattr(self, "_last_norm_chunk", None)
            if rtc_p > 0 and prev is not None:
                start = self.config.n_action_steps
                if start + rtc_p <= prev.shape[0]:
                    action_prefix = prev[start : start + rtc_p].unsqueeze(0)
                    prefix_length = rtc_p
            model_batch = batch if has_stream else self._render_online_stream_batch(batch)
            actions = self.predict_action_chunk(
                model_batch, action_prefix=action_prefix, prefix_length=prefix_length
            )
            actions = actions[:, : self.config.n_action_steps]
            self._queues[ACTION].extend(actions.transpose(0, 1))
        selected_action = self._queues[ACTION].popleft()
        self._record_action_for_history = bool(self.config.action_conditioning and not has_stream)
        return selected_action

    def get_loss_denominators(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Return local valid counts before model forward for exact update-wide normalization."""
        stream = batch.get(PERCEPTRON_ISAAC_STREAM_KEY)
        if stream is None:
            raise ValueError(
                f"ISAAC training requires the native preprocessor output {PERCEPTRON_ISAAC_STREAM_KEY!r}."
            )
        flow_fixture_tensors = self._aligned_flow_fixture_tensors(batch, len(stream.streams))
        return perceptron_isaac_loss_denominators(
            stream,
            loss_plan=str(self.config.loss_plan),
            action_dim=int(self.config.max_action_dim),
            action_horizon=int(self.config.chunk_size),
            train_samples_per_chunk=int(self.config.train_samples_per_chunk),
            flow_mask_padded_action_rows=bool(self.config.flow_mask_padded_action_rows),
            exclude_non_agent_roles=bool(self.config.exclude_non_agent_roles),
            flow_loss_mask=flow_fixture_tensors[ISAAC_FLOW_LOSS_MASK_KEY],
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        loss_denominators: dict[str, Tensor] | None = None,
        loss_accumulation_scale: int = 1,
    ) -> tuple[Tensor, dict]:
        self._require_native_training_supported()
        stream = batch.get(PERCEPTRON_ISAAC_STREAM_KEY)
        if stream is None:
            raise ValueError(
                f"ISAAC training requires the native preprocessor output {PERCEPTRON_ISAAC_STREAM_KEY!r}."
            )
        if self._isaac_model is None:
            self._load_backbone(training=True)
        self._configure_training_parameters()
        stream = stream.to(device=self._resolve_device(), dtype=self._model_dtype())
        model_output = self._isaac_model.train_forward(stream)
        flow_fixture_tensors = self._aligned_flow_fixture_tensors(batch, len(stream.streams))
        loss_output = perceptron_isaac_training_loss(
            model_output,
            stream,
            loss_plan=str(self.config.loss_plan),
            exclude_non_agent_roles=bool(self.config.exclude_non_agent_roles),
            flow_matching_detach_vlm_activations=bool(self.config.flow_matching_detach_vlm_activations),
            train_samples_per_chunk=int(self.config.train_samples_per_chunk),
            flow_mask_padded_action_rows=bool(self.config.flow_mask_padded_action_rows),
            softmax_auxiliary_loss_scale=float(self.config.softmax_auxiliary_loss_scale),
            flow_timesteps=flow_fixture_tensors[ISAAC_FLOW_TIMESTEPS_KEY],
            flow_noise=flow_fixture_tensors[ISAAC_FLOW_NOISE_KEY],
            flow_loss_mask=flow_fixture_tensors[ISAAC_FLOW_LOSS_MASK_KEY],
            global_denominators=loss_denominators,
            accumulation_scale=loss_accumulation_scale,
            reduction=reduction,
        )
        metrics = {
            name: value.detach().float().item() if torch.is_tensor(value) else float(value)
            for name, value in loss_output.metrics.items()
        }
        return loss_output.loss, metrics

    @staticmethod
    def _aligned_flow_fixture_tensors(
        batch: dict[str, Any], stream_batch_size: int
    ) -> dict[str, Tensor | None]:
        """Keep explicit oracle tensors aligned when the renderer skips action outliers."""
        kept_raw = batch.get(PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY)
        kept = None if kept_raw is None else torch.as_tensor(kept_raw, dtype=torch.long).reshape(-1)
        if kept is not None:
            if kept.numel() != stream_batch_size or bool((kept < 0).any()):
                raise ValueError(
                    "ISAAC kept-sample indices must be nonnegative and match the rendered stream batch."
                )
            if kept.unique().numel() != kept.numel() or not bool((kept[1:] > kept[:-1]).all()):
                raise ValueError("ISAAC kept-sample indices must be unique and strictly increasing.")

        aligned: dict[str, Tensor | None] = {}
        for key in (ISAAC_FLOW_TIMESTEPS_KEY, ISAAC_FLOW_NOISE_KEY, ISAAC_FLOW_LOSS_MASK_KEY):
            value = batch.get(key)
            if value is None:
                aligned[key] = None
                continue
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            if kept is not None and tensor.ndim >= 2 and tensor.shape[1] != stream_batch_size:
                if kept.numel() and int(kept.max()) >= tensor.shape[1]:
                    raise ValueError(
                        f"ISAAC fixture {key!r} cannot select kept indices {kept.tolist()} "  # nosec B608
                        f"from batch axis of size {tensor.shape[1]}."
                    )
                tensor = tensor.index_select(1, kept.to(device=tensor.device))
            aligned[key] = tensor
        return aligned

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PerceptronIsaacConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """Load a local or Hub-hosted LeRobot checkpoint and its checkpoint-local assets."""
        config_was_provided = config is not None
        if config is None:
            from lerobot.configs import PreTrainedConfig

            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        checkpoint_root = cls._resolve_checkpoint_root(
            pretrained_name_or_path,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        cls._resolve_checkpoint_local_paths(
            config,
            checkpoint_root,
            allow_pre_resolved_absolute=config_was_provided,
        )
        cls._verify_packaged_contract_digests(config, checkpoint_root)
        cls._set_checkpoint_local_vocab_path(config)
        policy = cls(config)
        cls._maybe_load_finetuned_weights(policy, checkpoint_root, strict=strict)
        # PEFT discovers target modules immediately after this returns. Training adapters with
        # FP32 expert storage therefore need the mixed-storage backbone materialized here.
        if config.use_peft and policy._isaac_model is None:
            policy._load_backbone(training=False)
        policy.eval()
        return policy

    @staticmethod
    def _verify_packaged_contract_digests(
        config: PerceptronIsaacConfig,
        checkpoint_root: Path | None,
    ) -> None:
        """Verify the checkpoint-family-specific integrity record before allocation.

        Existing Qwen3.5 packages retain the PR #4 import/export provenance
        contract. MK1 packages use their exhaustive package and model manifests.
        """
        if checkpoint_root is None:
            return
        root = Path(checkpoint_root).resolve()
        previous_qwen_package = getattr(config, "_qwen35_verified_package", None)
        if previous_qwen_package is not None:
            previous_qwen_package.cleanup()
            delattr(config, "_qwen35_verified_package")
        model_path = getattr(config, "hf_model_path", None)
        if model_path is not None and _is_portable_isaac05_repository(Path(model_path)):
            expected_adapter = getattr(config, "deployment_adapter_sha256", None)
            if not expected_adapter:
                raise RuntimeError("Portable Isaac-0.5 package requires deployment_adapter_sha256.")
            adapter_path = root / "isaac_deployment_adapter.json"
            if not PerceptronIsaacPolicy._is_package_file(root, adapter_path):
                raise RuntimeError(
                    f"Portable Isaac-0.5 package is missing its deployment adapter: {adapter_path}."
                )
            _verify_sha256(
                expected_adapter,
                "deployment_adapter_sha256",
                adapter_path,
                mismatch_error=lambda expected, actual: (
                    "Portable Isaac-0.5 deployment adapter digest mismatch: "
                    f"expected {expected}, found {actual}."
                ),
            )
            return
        is_mk1_package = PerceptronIsaacPolicy._config_declares_mk1(config)
        if is_mk1_package:
            if config.artifact_kind == "neutral_debug":
                PerceptronIsaacPolicy._verify_neutral_debug_manifest(config, root)
                return

            verified_model_import_digest = PerceptronIsaacPolicy._verify_trained_package_manifest(
                config, root
            )
            if verified_model_import_digest is not None:
                PerceptronIsaacPolicy._verify_mk1_model_import(
                    config,
                    root,
                    verified_record_digest=verified_model_import_digest,
                    root_weights_bound=True,
                )
            return

        checkpoint_root = Path(checkpoint_root)

        if config.qwen35_trained_package_manifest_sha256 is not None:
            from .qwen35_checkpoint import _snapshot_verified_qwen35_trained_package

            configured_model = Path(str(config.hf_model_path))
            if not configured_model.is_absolute():
                configured_model = root / configured_model

            def verify_snapshot(snapshot_root: Path) -> tuple[str | Path, bool]:
                from lerobot.configs import PreTrainedConfig

                snapshot_config = PreTrainedConfig.from_pretrained(snapshot_root)
                PerceptronIsaacPolicy._resolve_checkpoint_local_paths(snapshot_config, snapshot_root)
                PerceptronIsaacPolicy._verify_qwen35_trained_package_manifest(snapshot_config, snapshot_root)
                return snapshot_config.hf_model_path, bool(snapshot_config.apply_offset_norm)

            config._qwen35_verified_package = _snapshot_verified_qwen35_trained_package(
                root,
                hf_model_dir=configured_model,
                verify_snapshot=verify_snapshot,
            )
            return

        import_provenance = checkpoint_root / "isaac_import_provenance.json"
        export_provenance = checkpoint_root / "isaac_export_provenance.json"
        if import_provenance.is_file() and export_provenance.is_file():
            raise RuntimeError("ISAAC package cannot contain both import and export provenance.")
        if import_provenance.is_file() or export_provenance.is_file():
            kind = "imported" if import_provenance.is_file() else "exported"
            raise RuntimeError(
                f"Qwen3.5 {kind} Perceptron packages require the exhaustive outer trained-package "
                "manifest; its config digest or manifest is missing."
            )
        if getattr(config, "deployment_adapter_sha256", None):
            raise RuntimeError(
                "Qwen3.5 ISAAC package has deployment identity but no authenticated package provenance."
            )

    @staticmethod
    def _verify_qwen35_trained_package_manifest(
        config: PerceptronIsaacConfig,
        root: Path,
    ) -> None:
        from .qwen35_checkpoint import (
            validate_qwen35_hf_model_entries,
            verify_qwen35_weight_convention,
        )

        try:
            manifest = load_json_object(root / QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME)
        except ValueError as exc:
            raise RuntimeError(f"Cannot read Qwen3.5 trained package manifest: {exc}") from exc
        model_storage = manifest.get("model_storage")
        if model_storage not in {CANONICAL_INNER_MODEL_STORAGE, AUTHENTICATED_IMPORT_MODEL_STORAGE}:
            raise RuntimeError("Qwen3.5 trained package has unsupported model_storage metadata.")
        model_root = PerceptronIsaacPolicy._require_canonical_hf_model_path(
            config,
            root,
            context="Qwen3.5",
        )
        validate_qwen35_hf_model_entries(
            model_root,
            allow_source_lineage=model_storage == CANONICAL_INNER_MODEL_STORAGE,
        )
        if model_storage == CANONICAL_INNER_MODEL_STORAGE:
            verify_qwen35_weight_convention(model_root)
        verify_trained_package_manifest(
            root,
            label="Qwen3.5 trained package",
            manifest_filename=QWEN35_TRAINED_PACKAGE_MANIFEST_FILENAME,
            manifest_schema=QWEN35_TRAINED_PACKAGE_MANIFEST_SCHEMA,
            digest_field=QWEN35_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
            manifest_digest=config.qwen35_trained_package_manifest_sha256,
            identity={
                "artifact_kind": config.artifact_kind,
                "trained_steps": config.trained_steps,
            },
            metadata={"model_storage": model_storage},
            is_package_file=PerceptronIsaacPolicy._is_package_file,
            delegated_directory=("hf_model" if model_storage == AUTHENTICATED_IMPORT_MODEL_STORAGE else None),
        )
        expected_adapter_digest = _verify_sha256(
            config.deployment_adapter_sha256,
            "Qwen3.5 trained package deployment_adapter_sha256",
        )
        _verify_sha256(
            expected_adapter_digest,
            "Qwen3.5 trained package deployment_adapter_sha256",
            root / "isaac_deployment_adapter.json",
        )
        if model_storage == AUTHENTICATED_IMPORT_MODEL_STORAGE:
            from .qwen35_checkpoint import verify_qwen35_imported_package

            verify_qwen35_imported_package(root, hf_model_dir=model_root)
        elif (root / "isaac_import_provenance.json").exists() or (
            root / "hf_model" / "rmsnorm_conversion.json"
        ).exists():
            raise RuntimeError("Canonical Qwen3.5 trained weights retain stale import provenance.")

    @staticmethod
    def _require_canonical_hf_model_path(
        config: PerceptronIsaacConfig,
        root: Path,
        *,
        context: str,
    ) -> Path:
        model_path = getattr(config, "hf_model_path", None)
        if not model_path:
            raise RuntimeError(f"{context} trained package requires hf_model_path.")
        expected = root / "hf_model"
        configured_path = Path(str(model_path))
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        try:
            configured = configured_path.resolve(strict=True)
            packaged = expected.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"{context} trained package hf_model directory cannot be resolved.") from exc
        if configured != packaged or not expected.is_dir() or expected.is_symlink():
            raise RuntimeError(
                f"{context} trained package hf_model_path must resolve to package-local hf_model."
            )
        return expected

    @staticmethod
    def _verify_trained_package_manifest(config: PerceptronIsaacConfig, root: Path) -> str | None:
        """Verify every trained package asset outside the delegated HF model tree."""
        from .checkpoint_import import MK1_MODEL_IMPORT_FILENAME

        context = "MK1 trained package"
        manifest_path = root / MK1_TRAINED_PACKAGE_MANIFEST_FILENAME
        try:
            raw_manifest = load_json_object(manifest_path)
        except ValueError as exc:
            raise RuntimeError(f"Cannot read {context} manifest: {exc}") from exc
        manifest_schema = raw_manifest.get("schema")
        if manifest_schema == MK1_CANONICAL_TRAINED_PACKAGE_MANIFEST_SCHEMA:
            model_import_digest = _verify_sha256(
                config.mk1_source_model_import_sha256,
                f"{context} mk1_source_model_import_sha256",
            )
            verify_trained_package_manifest(
                root,
                label="MK1 canonical trained package",
                manifest_filename=MK1_TRAINED_PACKAGE_MANIFEST_FILENAME,
                manifest_schema=MK1_CANONICAL_TRAINED_PACKAGE_MANIFEST_SCHEMA,
                digest_field=MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
                manifest_digest=config.mk1_trained_package_manifest_sha256,
                identity={
                    "artifact_kind": config.artifact_kind,
                    "trained_steps": config.trained_steps,
                    "mk1_source_model_import_sha256": model_import_digest,
                },
                metadata={"model_storage": CANONICAL_INNER_MODEL_STORAGE},
                is_package_file=PerceptronIsaacPolicy._is_package_file,
                delegated_directory=None,
            )
            expected_adapter_digest = _verify_sha256(
                config.deployment_adapter_sha256,
                f"{context} deployment_adapter_sha256",
            )
            adapter_path = root / "isaac_deployment_adapter.json"
            _verify_sha256(
                expected_adapter_digest,
                f"{context} deployment_adapter_sha256",
                adapter_path,
            )
            PerceptronIsaacPolicy._require_canonical_hf_model_path(config, root, context="MK1")
            return None

        expected_adapter_digest = getattr(config, "deployment_adapter_sha256", None)
        if not expected_adapter_digest:
            raise RuntimeError("A trained MK1 Perceptron Isaac package requires deployment_adapter_sha256.")
        expected_adapter_digest = _verify_sha256(
            expected_adapter_digest, f"{context} deployment_adapter_sha256"
        )
        adapter_path = root / "isaac_deployment_adapter.json"
        if not PerceptronIsaacPolicy._is_package_file(root, adapter_path):
            raise RuntimeError(
                "A trained MK1 Perceptron Isaac package is missing its local deployment adapter: "
                f"{adapter_path}."
            )
        expected_model_import_digest = _verify_sha256(
            getattr(config, "mk1_model_import_sha256", None),
            f"{context} mk1_model_import_sha256",
        )
        expected_model_root = PerceptronIsaacPolicy._require_canonical_hf_model_path(
            config, root, context="MK1"
        )
        model_import_path = expected_model_root / MK1_MODEL_IMPORT_FILENAME
        if not PerceptronIsaacPolicy._is_package_file(root, model_import_path):
            raise RuntimeError(f"A trained MK1 package is missing {MK1_MODEL_IMPORT_FILENAME}.")
        verified_model_import_digest = file_sha256(model_import_path)
        _verify_sha256(
            expected_model_import_digest,
            f"{context} model-import record digest",
            verified_model_import_digest,
        )
        if config.artifact_kind != "trained_policy":
            raise RuntimeError("MK1 trained package manifest identity disagrees with the policy config.")

        manifest = verify_trained_package_manifest(
            root,
            label=context,
            manifest_filename=MK1_TRAINED_PACKAGE_MANIFEST_FILENAME,
            manifest_schema=MK1_TRAINED_PACKAGE_MANIFEST_SCHEMA,
            digest_field=MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD,
            manifest_digest=getattr(config, MK1_TRAINED_PACKAGE_MANIFEST_DIGEST_FIELD, None),
            identity={
                "artifact_kind": "trained_policy",
                "trained_steps": config.trained_steps,
                "mk1_model_import_sha256": expected_model_import_digest,
            },
            is_package_file=PerceptronIsaacPolicy._is_package_file,
            delegated_directory="hf_model",
        )
        raw_files = manifest["files_sha256"]
        if not raw_files:
            raise RuntimeError("MK1 trained package files_sha256 must be a non-empty object.")

        def adapter_mismatch(expected: str, actual: str) -> str:
            return (
                f"Perceptron Isaac deployment adapter digest mismatch at {adapter_path}: "
                f"expected {expected}, found {actual}. The package was modified after import."
            )

        _verify_sha256(
            expected_adapter_digest,
            f"{context} deployment_adapter_sha256",
            adapter_path,
            mismatch_error=adapter_mismatch,
        )
        _verify_sha256(
            expected_adapter_digest,
            f"{context} deployment_adapter_sha256",
            raw_files[adapter_path.name],
            mismatch_error=adapter_mismatch,
        )
        return verified_model_import_digest

    @staticmethod
    def _verify_mk1_model_import(
        config: PerceptronIsaacConfig,
        root: Path,
        *,
        verified_record_digest: str | None = None,
        root_weights_bound: bool = False,
    ) -> None:
        """Verify the exhaustive MK1 model import record before allocating weights."""
        from .mk1_checkpoint_contract import (
            Mk1CheckpointContractError,
            read_mk1_config,
            validate_mk1_tokenizer,
        )

        context = "MK1 model import"
        record_name = "mk1_model_import.json"
        model_path = getattr(config, "hf_model_path", None)
        if not model_path:
            raise RuntimeError("A trained Perceptron Isaac package requires hf_model_path.")
        root_weight_overrides = {
            candidate.name
            for candidate in root.iterdir()
            if candidate.name == "model.safetensors"
            or candidate.name == "model.safetensors.index.json"
            or (candidate.name.startswith("model-") and candidate.name.endswith(".safetensors"))
        }
        if root_weight_overrides and not root_weights_bound:
            raise RuntimeError(
                "Trained MK1 packages forbid unbound root-level weight overrides; "
                f"found {sorted(root_weight_overrides)}. Re-import them into the hashed hf_model inventory."
            )
        model_root = Path(str(model_path))
        record_path = model_root / record_name
        if not PerceptronIsaacPolicy._is_package_file(root, record_path):
            raise RuntimeError(f"A trained MK1 package is missing {record_name}.")
        expected_record_digest = getattr(config, "mk1_model_import_sha256", None)
        if not expected_record_digest:
            raise RuntimeError("A trained MK1 package requires mk1_model_import_sha256.")
        if verified_record_digest is None:
            verified_record_digest = file_sha256(record_path)
        _verify_sha256(expected_record_digest, f"{context} record digest", verified_record_digest)

        record = _load_json_object(record_path, record_name, f"{context} JSON")
        record_keys = {
            "schema",
            "algorithm",
            "backbone_family",
            "rmsnorm_weight_convention",
            "artifact",
            "source_files_sha256",
            "files_sha256",
            "qwen35_vocab",
            "shards",
        }
        if set(record) != record_keys:
            raise RuntimeError("MK1 model import record fields do not match schema version 2.")
        if (
            record["schema"] != "perceptron_isaac_mk1_model_import_v2"
            or record["algorithm"] != "byte_for_byte_safetensors_copy"
            or record["backbone_family"] != "mk1_qwen3_6_moe"
            or record["rmsnorm_weight_convention"] != "zero_centered_1_plus_weight"
        ):
            raise RuntimeError("MK1 model import record identity is unsupported.")

        def digest_map(value: Any, *, label: str) -> dict[str, str]:
            if not isinstance(value, dict) or not value:
                raise RuntimeError(f"MK1 model import {label} must be a non-empty object.")
            result: dict[str, str] = {}
            for filename, digest in value.items():
                path = Path(filename) if isinstance(filename, str) else Path()
                if (
                    not isinstance(filename, str)
                    or not filename
                    or path.is_absolute()
                    or path.name != filename
                    or path.as_posix() != filename
                    or ".." in path.parts
                ):
                    raise RuntimeError(f"MK1 model import {label} contains an invalid filename.")
                result[filename] = _verify_sha256(digest, f"{context} {label}[{filename!r}]")
            return result

        files = digest_map(record["files_sha256"], label="files_sha256")
        source_files = digest_map(record["source_files_sha256"], label="source_files_sha256")
        if record_name in files or record_name in source_files:
            raise RuntimeError("MK1 model import record must not include itself in a digest map.")

        actual_files: set[str] = set()
        for candidate in model_root.iterdir():
            if candidate.is_dir() or not PerceptronIsaacPolicy._is_package_file(root, candidate):
                raise RuntimeError(f"MK1 model directory contains a non-package file: {candidate}.")
            actual_files.add(candidate.name)
        if actual_files != set(files) | {record_name}:
            raise RuntimeError(
                "MK1 model file inventory disagrees with its import record: "
                f"missing={sorted(set(files) - actual_files)}, "
                f"unlisted={sorted(actual_files - set(files) - {record_name})}."
            )
        for filename, expected in files.items():
            _verify_sha256(
                expected,
                f"{context} files_sha256[{filename!r}]",
                model_root / filename,
                f"{context} digest mismatch for {filename}.",
            )

        required_files = {
            "config.json",
            "tokenizer.json",
            "vocab.json",
            "model.safetensors.index.json",
        }
        missing_required = required_files - set(files)
        missing_source = (required_files - {"vocab.json"}) - set(source_files)
        if missing_required or missing_source:
            raise RuntimeError(
                "MK1 model import record does not bind all required assets: "
                f"packaged_missing={sorted(missing_required)}, source_missing={sorted(missing_source)}."
            )

        index = _load_json_object(
            model_root / "model.safetensors.index.json", "model.safetensors.index.json", f"{context} JSON"
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("MK1 model index must contain a non-empty weight_map.")
        shard_counts: dict[str, int] = {}
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                raise RuntimeError("MK1 model index contains an invalid weight_map entry.")
            shard_counts[shard_name] = shard_counts.get(shard_name, 0) + 1
        indexed_shards = set(shard_counts)
        recorded_shards = record["shards"]
        if not isinstance(recorded_shards, list) or not recorded_shards:
            raise RuntimeError("MK1 model import shards must be a non-empty list.")
        if [item.get("filename") for item in recorded_shards if isinstance(item, dict)] != sorted(
            indexed_shards
        ):
            raise RuntimeError("MK1 model import shard ordering or coverage disagrees with the index.")
        for item in recorded_shards:
            if not isinstance(item, dict) or set(item) != {
                "filename",
                "source_sha256",
                "output_sha256",
                "tensor_count",
            }:
                raise RuntimeError("MK1 model import contains a malformed shard record.")
            filename = item["filename"]
            count = item["tensor_count"]
            if (
                filename not in indexed_shards
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count != shard_counts[filename]
                or files.get(filename) != _verify_sha256(item["output_sha256"], f"{context} shard output")
                or source_files.get(filename)
                != _verify_sha256(item["source_sha256"], f"{context} shard source")
            ):
                raise RuntimeError(f"MK1 model import shard record disagrees for {filename!r}.")
        packaged_safetensors = {name for name in files if name.endswith(".safetensors")}
        if packaged_safetensors != indexed_shards:
            raise RuntimeError("MK1 model import safetensors files disagree with the index.")

        for filename in {"config.json", "tokenizer.json", "model.safetensors.index.json"} | indexed_shards:
            if source_files.get(filename) != files[filename]:
                raise RuntimeError(f"MK1 byte-copy provenance disagrees for {filename}.")

        vocab = record["qwen35_vocab"]
        if not isinstance(vocab, dict) or set(vocab) != {
            "schema",
            "algorithm",
            "token_count",
            "tokenizer_sha256",
            "vocab_sha256",
        }:
            raise RuntimeError("MK1 qwen35_vocab record is malformed.")
        if (
            vocab["schema"] != "perceptron_isaac_qwen35_vocab_v1"
            or vocab["algorithm"] != "extract_tokenizer_json_model_vocab"
            or vocab["token_count"] != _MK1_BASE_VOCAB_SIZE
            or _verify_sha256(vocab["tokenizer_sha256"], f"{context} qwen35 tokenizer")
            != files["tokenizer.json"]
            or _verify_sha256(vocab["vocab_sha256"], f"{context} qwen35 vocab") != files["vocab.json"]
        ):
            raise RuntimeError("MK1 qwen35_vocab record disagrees with packaged tokenizer assets.")

        artifact = record["artifact"]
        expected_artifact = {
            "schema_version": 1,
            "artifact_kind": "trained_policy",
            "trained_steps": config.trained_steps,
        }
        if (
            artifact != expected_artifact
            or not isinstance(config.trained_steps, int)
            or config.trained_steps <= 0
        ):
            raise RuntimeError("MK1 model import artifact identity disagrees with the package config.")
        raw_config = _load_json_object(model_root / "config.json", "MK1 config.json", f"{context} JSON")
        if raw_config.get("genesis_artifact") != artifact:
            raise RuntimeError("MK1 inner config artifact identity disagrees with its import record.")

        try:
            contract = read_mk1_config(model_root)
            validate_mk1_tokenizer(model_root, contract)
        except Mk1CheckpointContractError as exc:
            raise RuntimeError(f"MK1 inner config contract validation failed: {exc}") from exc
        if (
            config.vector_max_states != contract.vector_encoder.max_states
            or config.max_action_dim != contract.action_expert.action_dim
            or config.max_action_horizon != contract.action_expert.action_horizon
        ):
            raise RuntimeError("MK1 outer checkpoint maxima disagree with the inner model contract.")
        if (
            config.action_dim > config.max_action_dim
            or config.chunk_size > config.max_action_horizon
            or config.proprio_dim > config.vector_max_states
        ):
            raise RuntimeError("MK1 served policy geometry exceeds the checkpoint-owned maxima.")

    @staticmethod
    def _verify_neutral_debug_manifest(config: PerceptronIsaacConfig, root: Path) -> None:
        context = "Neutral-debug manifest"
        manifest_name = "mk1_neutral_debug_manifest.json"
        manifest_path = root / manifest_name
        if not PerceptronIsaacPolicy._is_package_file(root, manifest_path):
            raise RuntimeError(f"Neutral-debug package is missing {manifest_name}.")

        manifest = _load_json_object(
            manifest_path,
            context.lower(),
            context,
            f"{context} fields do not match schema version 1.",
        )
        expected_keys = {
            "schema_version",
            "artifact_kind",
            "trained_steps",
            "source_config_sha256",
            "source_index_sha256",
            "model_materialization",
            "files_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected_keys:
            raise RuntimeError("Neutral-debug manifest fields do not match schema version 1.")
        if (
            manifest["schema_version"] != 1
            or manifest["artifact_kind"] != "neutral_debug"
            or manifest["trained_steps"] != 0
            or config.artifact_kind != "neutral_debug"
            or config.trained_steps != 0
        ):
            raise RuntimeError("Neutral-debug manifest identity disagrees with the package config.")

        files = manifest["files_sha256"]
        if not isinstance(files, dict) or not files:
            raise RuntimeError("Neutral-debug manifest files_sha256 must be a non-empty object.")
        expected_files: dict[str, str] = {}
        for relative, expected_digest in files.items():
            if not isinstance(relative, str) or not relative:
                raise RuntimeError("Neutral-debug manifest contains an invalid file path.")
            raw_path = Path(relative)
            if raw_path.is_absolute() or raw_path.as_posix() != relative or ".." in raw_path.parts:
                raise RuntimeError(f"Neutral-debug manifest path is not canonical and local: {relative!r}.")
            expected_files[relative] = _verify_sha256(
                expected_digest,
                f"{context} files_sha256[{relative!r}]",
            )
        if manifest_name in expected_files:
            raise RuntimeError("Neutral-debug manifest must not hash itself.")

        actual_files: set[str] = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                if not PerceptronIsaacPolicy._is_package_file(root, candidate):
                    raise RuntimeError(f"Neutral-debug package contains an escaping symlink: {candidate}.")
                actual_files.add(candidate.relative_to(root).as_posix())
                continue
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise RuntimeError(f"Neutral-debug package contains a non-regular entry: {candidate}.")
            actual_files.add(candidate.relative_to(root).as_posix())
        if actual_files != set(expected_files) | {manifest_name}:
            raise RuntimeError(
                "Neutral-debug package file inventory disagrees with its manifest: "
                f"missing={sorted(set(expected_files) - actual_files)}, "
                f"unlisted={sorted(actual_files - set(expected_files) - {manifest_name})}."
            )
        for relative, expected_digest in expected_files.items():
            _verify_sha256(
                expected_digest,
                f"{context} files_sha256[{relative!r}]",
                root / relative,
                f"Neutral-debug package digest mismatch for {relative}.",
            )

        source_config = _verify_sha256(manifest["source_config_sha256"], f"{context} source_config_sha256")
        source_index = _verify_sha256(manifest["source_index_sha256"], f"{context} source_index_sha256")
        if expected_files.get("hf_model/config.json") != source_config:
            raise RuntimeError("Neutral-debug source config digest disagrees with the packaged model config.")
        if expected_files.get("hf_model/model.safetensors.index.json") != source_index:
            raise RuntimeError("Neutral-debug source index digest disagrees with the packaged model index.")

        materialization = manifest["model_materialization"]
        model_files = {
            Path(relative).name for relative in expected_files if Path(relative).parent == Path("hf_model")
        }
        if (
            not isinstance(materialization, dict)
            or set(materialization) != model_files
            or any(method not in {"copy", "hardlink"} for method in materialization.values())
        ):
            raise RuntimeError("Neutral-debug model materialization inventory is malformed.")

    @staticmethod
    def _is_package_file(root: Path, path: Path) -> bool:
        """Accept package-local files and immutable HF snapshot links only."""
        try:
            root = root.resolve(strict=True)
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        if not resolved.is_file():
            return False
        if resolved.is_relative_to(root):
            return True
        return is_hf_snapshot_blob_link(path)

    @staticmethod
    def _resolve_checkpoint_root(
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> Path:
        from huggingface_hub.errors import HfHubHTTPError

        try:
            return resolve_hub_snapshot(
                pretrained_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )
        except HfHubHTTPError as exc:
            raise FileNotFoundError(
                f"Perceptron Isaac checkpoint not found locally or on the Hugging Face Hub: "
                f"{pretrained_name_or_path}"
            ) from exc

    @classmethod
    def resolve_checkpoint_config_paths(
        cls, config: PerceptronIsaacConfig, checkpoint_root: str | Path
    ) -> None:
        """Resolve the config's relative sidecar paths against a checkpoint root.

        Public hook for PEFT loaders: ``from_pretrained`` runs against the BASE
        package (``peft_config.base_model_name_or_path``), so the adapter config's
        relative paths must be resolved against the ADAPTER checkpoint root first.
        Without this, a relative ``native_stats_path: isaac_stats.json`` in a
        finetune's config.json resolves to the base package's stats file and
        silently reinstates the serving-stats split-brain. Resolution is
        idempotent: paths made absolute here are left untouched by the later
        base-root pass inside ``from_pretrained``.
        """
        cls._resolve_checkpoint_local_paths(
            config,
            checkpoint_root,
            allow_missing_relative=True,
            record_source_root=False,
        )

    @staticmethod
    def _resolve_checkpoint_local_paths(
        config: PerceptronIsaacConfig,
        pretrained_name_or_path: str | Path,
        *,
        allow_pre_resolved_absolute: bool = False,
        allow_missing_relative: bool = False,
        record_source_root: bool = True,
    ) -> None:
        """Resolve and validate checkpoint-local sidecars.

        Public loader hooks may run twice for PEFT: once against the adapter and
        again against its base. In that path, already-resolved absolute values are
        intentionally left untouched. Direct package validation stays strict.
        """
        root = Path(pretrained_name_or_path)
        if not root.is_dir():
            return
        root = root.resolve()
        if record_source_root:
            config._checkpoint_source_root = str(root)
        relative_paths = dict(getattr(config, "_checkpoint_relative_paths", {}))
        directory_attrs = {"hf_model_path", "fast_processor_path"}
        # Every path-typed config field, deliberately: a partial list is the kind of
        # invariant that regrows the split-brain the next time a consumer is added.
        # The suite pair is training-time routing input rather than a serving sidecar,
        # but it is saved checkpoint-local for the same reason (see
        # PerceptronIsaacConfig._save_pretrained), so it resolves the same way.
        for attr in (
            "hf_model_path",
            "config_toml_path",
            "stats_path",
            "native_render_metadata_path",
            "native_stats_path",
            "fast_processor_path",
            "suite_stats_path",
            "suite_by_task_index_path",
        ):
            raw = getattr(config, attr, None)
            if raw is None or not str(raw).strip():
                continue
            path = Path(str(raw))
            if path.is_absolute():
                record = relative_paths.get(attr)
                if isinstance(record, dict) and record.get("absolute") == str(path):
                    continue
                if allow_pre_resolved_absolute and getattr(
                    config, "_allow_trusted_absolute_checkpoint_paths", False
                ):
                    continue
                raise RuntimeError(f"Checkpoint-local path {attr} must be relative, got {path}.")
            unresolved_candidate = root / path
            if ".." in path.parts:
                portable_parent = root.parent
                expected_portable_path = None
                if attr == "hf_model_path" and path == Path(".."):
                    expected_portable_path = portable_parent
                elif attr == "fast_processor_path" and path == Path("../fast_processor_pinned"):
                    expected_portable_path = portable_parent / "fast_processor_pinned"
                try:
                    resolved_portable_path = unresolved_candidate.resolve(strict=True)
                except (OSError, RuntimeError):
                    resolved_portable_path = None
                if (
                    expected_portable_path is not None
                    and resolved_portable_path == expected_portable_path
                    and _is_portable_isaac05_repository(portable_parent)
                ):
                    absolute = str(resolved_portable_path)
                    relative_paths[attr] = {
                        "absolute": absolute,
                        "relative": path.as_posix(),
                        "root": str(root),
                    }
                    setattr(config, attr, absolute)
                    continue
                raise RuntimeError(f"Checkpoint-local path {attr} escapes the package root: {path}.")
            try:
                candidate = unresolved_candidate.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                if allow_missing_relative:
                    continue
                raise RuntimeError(f"Checkpoint-local path {attr} does not exist: {path}.") from exc
            if attr in directory_attrs:
                expected_type_matches = (
                    candidate.is_dir()
                    and unresolved_candidate.is_dir()
                    and not unresolved_candidate.is_symlink()
                )
                if expected_type_matches:
                    for child in unresolved_candidate.rglob("*"):
                        if child.is_symlink() and not PerceptronIsaacPolicy._is_package_file(root, child):
                            raise RuntimeError(
                                f"Checkpoint-local directory {attr} contains an escaping symlink: {child}."
                            )
            else:
                expected_type_matches = PerceptronIsaacPolicy._is_package_file(root, unresolved_candidate)
            if not expected_type_matches:
                expected_type = "directory" if attr in directory_attrs else "file"
                raise RuntimeError(f"Checkpoint-local path {attr} must reference a {expected_type}: {path}.")
            absolute = str(unresolved_candidate)
            relative_paths[attr] = {
                "absolute": absolute,
                "relative": path.as_posix(),
                "root": str(root),
            }
            setattr(config, attr, absolute)
        config._checkpoint_relative_paths = relative_paths

    @staticmethod
    def _set_checkpoint_local_vocab_path(
        config: PerceptronIsaacConfig,
        *,
        force: bool = False,
    ) -> None:
        """Publish the package vocab only after its integrity record has passed."""
        import os

        verified_package = getattr(config, "_qwen35_verified_package", None)
        if (
            not force
            and verified_package is None
            and not PerceptronIsaacPolicy._config_declares_mk1(config)
            and os.environ.get("QWEN35_VOCAB_PATH")
        ):
            return
        hf_model_path = getattr(verified_package, "hf_model_path", None) or getattr(
            config, "hf_model_path", None
        )
        if hf_model_path:
            vocab = Path(str(hf_model_path)) / "vocab.json"
            if vocab.is_file():
                os.environ["QWEN35_VOCAB_PATH"] = str(vocab)

    @staticmethod
    def _maybe_load_finetuned_weights(
        policy: PerceptronIsaacPolicy,
        pretrained_name_or_path: str | Path,
        *,
        strict: bool = False,
    ) -> None:
        """Load LeRobot-saved finetuned policy weights when present."""
        import logging

        from safetensors.torch import load_file

        from .qwen35_checkpoint import safetensors_checkpoint_layout

        p = Path(pretrained_name_or_path)
        if not p.is_dir():
            return
        if not any(
            (
                (p / "model.safetensors").is_file(),
                (p / "model.safetensors.index.json").is_file(),
                any(p.glob("model-*.safetensors")),
            )
        ):
            # Imported packages carry no LeRobot-saved weights. Materializing the backbone
            # here would pin eval dtype (bf16); leave it unloaded so a later training load
            # honors train_storage_fp32.
            return
        if not (p / "model.safetensors.index.json").is_file() and any(p.glob("model-*.safetensors")):
            raise RuntimeError("Outer sharded weights require model.safetensors.index.json.")
        try:
            shards, indexed_keys = safetensors_checkpoint_layout(p)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"Outer weight inventory contains unreferenced or missing files: {exc}"
            ) from exc
        state: dict[str, Tensor] = {}
        for shard in shards:
            shard_state = load_file(str(shard))
            if indexed_keys is not None:
                expected_keys = indexed_keys[shard.name]
                actual_keys = set(shard_state)
                if actual_keys != expected_keys:
                    raise RuntimeError(
                        f"Outer weight shard {shard.name} disagrees with its index: "
                        f"missing={sorted(expected_keys - actual_keys)}, "
                        f"unindexed={sorted(actual_keys - expected_keys)}."
                    )
            duplicate_keys = set(state).intersection(shard_state)
            if duplicate_keys:
                raise RuntimeError(
                    f"Outer weight shards contain duplicate tensors: {sorted(duplicate_keys)}."
                )
            state.update(shard_state)
        if policy._isaac_model is None:
            policy._load_backbone(training=not policy._config_declares_mk1(policy.config))
        overlap = set(policy.state_dict()).intersection(state)
        if not overlap:
            raise RuntimeError(
                "perceptron_isaac.from_pretrained found finetuned weights but no keys matched the policy."
            )
        missing, unexpected = policy.load_state_dict(state, strict=strict)
        logging.getLogger(__name__).info(
            "perceptron_isaac.from_pretrained: loaded finetuned weights (%d tensors; missing=%d unexpected=%d)",
            len(state),
            len(missing),
            len(unexpected),
        )
