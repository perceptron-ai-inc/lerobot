"""Processor pipelines for the Perceptron Isaac policy.

Native eval/serving uses a LeRobot processor pipeline: observations are
renamed, windowed, lowered to mharmony, packed into an Isaac stream, moved to the
policy device, and normalized actions are unnormalized on the way back to CPU.
"""

import json
import logging
import re
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies.processor_utils import (
    RETAINED_SAMPLE_INDICES_KEY,
    coerce_batch_text,
    coerce_rgb_uint8,
    flatten_feature_names,
    to_numpy as _to_numpy,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_perceptron_isaac import PerceptronIsaacConfig
from .fast_processor import load_fast_action_processor
from .isaac_stats import (
    IsaacNormalizationStats,
    IsaacStatsBlock,
    apply_gripper_passthrough,
    isaac_stats_from_state_dict,
    isaac_stats_to_state_dict,
    isaac_stats_value_differences,
    load_isaac_stats,
    load_isaac_suite_stats,
    require_closed_loop_safe_isaac_profile,
    require_closed_loop_safe_isaac_stats,
    unnormalize_isaac_actions,
)
from .mharmony_contract import SUPPORTED_MHARMONY_VERSION, normalize_mharmony_version_marker
from .mharmony_native import (
    IsaacActionOutlierError,
    IsaacMharmonyRenderMetadata,
    IsaacNativeMharmonyRenderer,
    hydrate_isaac_training_proprio_contract,
)

PERCEPTRON_ISAAC_STREAM_KEY = "perceptron_isaac_stream"
PERCEPTRON_ISAAC_RENDER_META_KEY = "perceptron_isaac_render_meta"
PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY = "perceptron_isaac_anchor_timestamp_seconds"
PERCEPTRON_ISAAC_PREVIOUS_ACTIONS_KEY = "perceptron_isaac_previous_actions"
# Backward-compatible policy-specific alias for the shared filtered-batch protocol.
PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY = RETAINED_SAMPLE_INDICES_KEY
# When online single-step serving skips rendering (the anchor arrives only after
# preprocessing), the saved pack step hands its restored normalization stats to the
# policy through this batch key, so the policy-owned online render step packs proprio
# with the checkpoint's serialized stats instead of independently path-loading
# config.native_stats_path (which, for PEFT finetunes, points at the base package).
PERCEPTRON_ISAAC_SERVING_STATS_KEY = "perceptron_isaac_serving_stats"

_TASK_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,\u2026"
_TASK_TRAILING_CLOSERS = "\"'\u201d\u2019)]}"
_TASK_SURROUNDING_DELIMITERS = "\"'`\u201c\u201d\u2018\u2019[](){}"
_TASK_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)
_CAMERA_FALLBACK_SUFFIXES = {
    "image": ("image", "agentview_image", "agentview", "front_image"),
    "wrist_image": ("wrist_image", "image2", "robot0_eye_in_hand_image", "eye_in_hand", "wrist"),
}
_PACK_CONFIG_METADATA = {"pack_config": True}
_LIST_CONFIG_FIELD_NAMES = frozenset({"camera_order", "image_size", "image_keys"})


def _adopt_dataset_state_feature_names(config: PerceptronIsaacConfig, dataset_meta: Any | None) -> None:
    """Capture non-strict dataset state layout so gripper masks can be derived and saved."""
    if dataset_meta is None or config.strict_hardware_feature_contract:
        return
    features = getattr(dataset_meta, "features", None)
    if not isinstance(features, dict):
        return
    state_feature = features.get(OBS_STATE)
    if state_feature is None:
        return
    raw_names = (
        state_feature.get("names")
        if isinstance(state_feature, dict)
        else getattr(state_feature, "names", None)
    )
    names = flatten_feature_names(raw_names, empty_as_none=True)
    if names is None:
        return
    if len(names) != int(config.proprio_dim):
        raise ValueError(
            f"Dataset {OBS_STATE} names cover {len(names)} dimensions, but Perceptron Isaac "
            f"expects proprio_dim={config.proprio_dim}."
        )
    config.state_feature_names = names


def _dataset_stats_to_isaac_stats(dataset_stats, config, *, dataset_fps):
    """Convert LeRobot dataset quantiles into native Isaac normalization stats."""
    action_raw = dataset_stats.get("action")
    proprio_raw = dataset_stats.get("observation.state") or dataset_stats.get("state")
    if action_raw is None or proprio_raw is None:
        raise ValueError("Perceptron Isaac fine-tuning requires action and observation.state dataset stats.")

    def block(raw, name):
        if "q01" not in raw or "q99" not in raw:
            raise ValueError(f"Dataset stats for {name} are missing q01/q99; recompute dataset stats.")
        q01 = np.asarray(raw["q01"], dtype=np.float32).reshape(-1)
        q99 = np.asarray(raw["q99"], dtype=np.float32).reshape(-1)
        minimum = np.asarray(raw["min"], dtype=np.float32).reshape(-1) if "min" in raw else None
        maximum = np.asarray(raw["max"], dtype=np.float32).reshape(-1) if "max" in raw else None
        if config.joint_signs is not None and config.joint_offsets is not None:
            n = len(config.joint_signs)
            if q01.size < n:
                raise ValueError(
                    f"Dataset {name} stats dim {q01.size} is smaller than joint transform dim {n}."
                )
            signs = np.asarray(config.joint_signs, dtype=np.float32)
            offsets = np.asarray(config.joint_offsets, dtype=np.float32)
            lo, hi = signs * q01[:n] + offsets, signs * q99[:n] + offsets
            q01, q99 = q01.copy(), q99.copy()
            q01[:n], q99[:n] = np.minimum(lo, hi), np.maximum(lo, hi)
            if minimum is not None and maximum is not None:
                lo, hi = signs * minimum[:n] + offsets, signs * maximum[:n] + offsets
                minimum, maximum = minimum.copy(), maximum.copy()
                minimum[:n], maximum[:n] = np.minimum(lo, hi), np.maximum(lo, hi)
        return IsaacStatsBlock(q01=q01, q99=q99, minimum=minimum, maximum=maximum)

    action = block(action_raw, "action")
    proprio = block(proprio_raw, "observation.state")
    if action.dim != int(config.action_dim) or proprio.dim != int(config.proprio_dim):
        raise ValueError(
            "Dataset stats dimensions do not match the Isaac policy contract: "
            f"action={action.dim}/{config.action_dim}, proprio={proprio.dim}/{config.proprio_dim}."
        )
    # The finetune-dataset quantiles are DIFFERENT VALUES from the base profile, so they
    # must not reuse the base's profile_id verbatim: two bundles sharing an id with
    # different numbers makes every label-keyed identity check lie. Values remain the
    # only trustworthy comparison (isaac_stats_value_differences), but the label should
    # not collide on top of that.
    base_profile_id = config.normalization_profile_id
    if base_profile_id and not str(base_profile_id).endswith("#finetune_dataset"):
        profile_id = f"{base_profile_id}#finetune_dataset"
    else:
        profile_id = base_profile_id
    stats = IsaacNormalizationStats(
        action=action,
        proprio=proprio,
        # The ISAAC contract owns the time base (see _frame_index_anchors): converted LIBERO
        # datasets misdeclare fps=10 for 20 Hz demos, and stamping that misdeclared rate here
        # makes _ensure_renderer reject the normalization stats this function just built.
        # Prefer the packaged contract fps; _frame_index_anchors independently checks the
        # declared dataset rate against that contract.
        target_fps=(
            float(config.target_fps)
            if config.target_fps is not None
            else (float(dataset_fps) if dataset_fps is not None else None)
        ),
        action_horizon=int(config.chunk_size),
        schema="lerobot_dataset_quantiles",
        profile_id=profile_id,
        profile_scope="finetune_dataset",
        validation_status="dataset_stats",
    )
    # Finetuning takes this branch, not the path-load one: _install_dataset_stats assigns
    # step._stats directly, so a mask applied only at load time would never reach training
    # or the saved checkpoint. Apply it here, where every dataset-stats consumer converges.
    if not bool(getattr(config, "normalize_gripper", True)):
        stats = apply_gripper_passthrough(
            stats,
            action_feature_names=list(config.action_feature_names or []),
            state_feature_names=list(config.state_feature_names or []),
        )
    return stats


def _sync_normalization_labels(step: Any, stats: IsaacNormalizationStats) -> None:
    """Keep the step's JSON-config labels truthful about the stats actually installed.

    Without this, a finetune restored from a base package saves step configs still
    advertising the base lineage (`normalization_profile_scope`, `validation_status`)
    while the authoritative safetensors state carries finetune-dataset stats.
    """
    step.normalization_profile_id = stats.profile_id
    step.normalization_profile_scope = stats.profile_scope
    step.normalization_validation_status = stats.validation_status


def _install_dataset_stats(preprocessor, postprocessor, stats):
    for step in preprocessor.steps:
        if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep):
            step._stats = stats
            step._stream_builder = None
            _sync_normalization_labels(step, stats)
    for step in postprocessor.steps:
        if isinstance(step, PerceptronIsaacActionUnnormalizeProcessorStep):
            step._stats = stats
            _sync_normalization_labels(step, stats)


def _record_config_stats_export(config: PerceptronIsaacConfig, preprocessor) -> None:
    """Hand the live pipeline stats to the config for checkpoint-local export at save.

    ``_export_native_stats`` is a plain (non-field) attribute, so it never serializes
    into config.json itself; ``PerceptronIsaacConfig._save_pretrained`` consumes it to
    write a checkpoint-local ``isaac_stats.json`` and a relative ``native_stats_path``,
    keeping every saved finetune self-contained instead of pointing at the base
    package's stats file.
    """
    for step in preprocessor.steps:
        if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep) and step._stats is not None:
            config._export_native_stats = step._stats
            return


def normalize_perceptron_isaac_task_text(text: str) -> str:
    """Normalize task text before Genesis prompt assembly."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_TASK_SURROUNDING_DELIMITERS).strip()
        for pattern in _TASK_PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_TASK_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_TASK_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_TASK_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    return normalized.lower()


def build_perceptron_isaac_observation_window(
    images: Any,
    states: Any,
    camera_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Convert stacked image/state tensors to Genesis ObservationFrame dictionaries."""
    image_arr = _to_numpy(images)
    state_arr = _to_numpy(states).astype(np.float32, copy=False)
    if image_arr.ndim != 5:
        raise ValueError(
            "Perceptron Isaac expected stacked images shaped [T, N_cam, C, H, W] or [T, N_cam, H, W, C], "
            f"got {image_arr.shape}."
        )
    if state_arr.ndim != 2:
        raise ValueError(f"Perceptron Isaac expected stacked states shaped [T, D], got {state_arr.shape}.")
    if image_arr.shape[0] != state_arr.shape[0]:
        raise ValueError(f"Image/state timestep mismatch: {image_arr.shape[0]} vs {state_arr.shape[0]}.")
    if image_arr.shape[1] != len(camera_order):
        raise ValueError(
            f"Image camera axis has {image_arr.shape[1]} entries, but camera_order has {len(camera_order)}."
        )

    frames: list[dict[str, Any]] = []
    for t in range(image_arr.shape[0]):
        frame_imgs = {}
        for ci, cam in enumerate(camera_order):
            frame_imgs[cam] = coerce_rgb_uint8(
                image_arr[t, ci],
                squeeze_leading_singletons=True,
                shape_error="Unsupported image shape for Perceptron Isaac: {shape}.",
            )
        frames.append({"images": frame_imgs, "proprio": state_arr[t].reshape(-1)})
    return frames


def _dtype_from_name(name: str):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported Isaac render dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]


def _maybe_passthrough_grippers(step: Any, stats):
    """Apply the gripper passthrough mask to stats a step just read from a path.

    Stats restored from serialized state already carry their mask, so this is only for the
    path-load branches. `getattr` defaults keep it inert for steps that do not expose the
    fields, matching how this module already reads optional subclass attributes.
    """
    if getattr(step, "normalize_gripper", True):
        return stats
    action_names = getattr(step, "action_feature_names", None)
    state_names = getattr(step, "state_feature_names", None)
    if not action_names or not state_names:
        raise ValueError(
            f"{type(step).__name__} has normalize_gripper=False but no "
            "action_feature_names/state_feature_names, so the gripper dimensions cannot be "
            "located. Pass both, or leave normalize_gripper=True."
        )
    return apply_gripper_passthrough(
        stats, action_feature_names=action_names, state_feature_names=state_names
    )


@ProcessorStepRegistry.register(name="perceptron_isaac_render")
@dataclass
class PerceptronIsaacRenderProcessorStep(ProcessorStep):
    """Shared native mharmony pack implementation for ISAAC inference and training."""

    _serialize_pack_fields: ClassVar[bool] = False

    enabled: bool = False
    inference_backend: str = "native_mharmony"
    config_toml_path: str | None = None
    stats_path: str | None = None
    camera_order: list[str] = field(default_factory=lambda: ["image", "wrist_image"])
    image_size: tuple[int, int] = (256, 256)
    # Geometry contract for off-size inputs: "letterbox" scales by one factor then
    # center-pads, "stretch" resizes both dimensions exactly. Uses the same primitive as
    # the camera deployment path so training and serving geometry cannot drift; inputs
    # already at `image_size` pass through untouched (byte-identical).
    image_preprocessing: str = "stretch"
    image_keys: list[str] = field(default_factory=list)
    allow_image_key_fallback: bool = True
    normalize_language: bool = True
    n_obs_steps: int = 3
    chunk_size: int = 30
    objective: str = "flow"
    patch_size: int = 16
    max_num_patches: int | None = None
    min_num_patches: int | None = None
    pixel_shuffle_scale: int = 2
    temporal_patch_size: int = 2
    device: str = "cpu"
    dtype: str = "bfloat16"
    stream_key: str = PERCEPTRON_ISAAC_STREAM_KEY
    metadata_key: str = PERCEPTRON_ISAAC_RENDER_META_KEY
    target_fps: float | None = None
    normalization_profile_id: str | None = None
    normalization_profile_scope: str | None = None
    normalization_validation_status: str | None = None
    # Multi-suite training only. `suite_stats_path` maps scope -> normalization block and
    # `suite_by_task_index_path` maps the dataset's task_index -> scope; set both or neither.
    # Inference is single-scope by contract and leaves these unset.
    suite_stats_path: str | None = None
    suite_by_task_index_path: str | None = None
    native_render_metadata_path: str | None = field(default=None, metadata=_PACK_CONFIG_METADATA)
    native_stats_path: str | None = field(default=None, metadata=_PACK_CONFIG_METADATA)
    fast_processor_path: str | None = field(default=None, metadata=_PACK_CONFIG_METADATA)
    fast_processor_tree_sha256: str | None = field(default=None, metadata=_PACK_CONFIG_METADATA)
    mharmony_version: str = field(default=SUPPORTED_MHARMONY_VERSION, metadata=_PACK_CONFIG_METADATA)
    action_dim: int = field(default=7, metadata=_PACK_CONFIG_METADATA)
    proprio_dim: int = field(default=8, metadata=_PACK_CONFIG_METADATA)
    vector_max_states: int = field(default=128, metadata=_PACK_CONFIG_METADATA)
    dataset_name: str = field(default="libero", metadata=_PACK_CONFIG_METADATA)
    robot_type: str = field(default="generic", metadata=_PACK_CONFIG_METADATA)
    action_conditioning: bool = field(default=False, metadata=_PACK_CONFIG_METADATA)
    action_conditioning_role: str = field(default="user", metadata=_PACK_CONFIG_METADATA)
    mistake_conditioning: bool = field(default=False, metadata=_PACK_CONFIG_METADATA)
    render_metadata: dict[str, Any] | None = field(default=None, metadata=_PACK_CONFIG_METADATA)
    train_clip_normalized_actions: bool = field(default=True, metadata=_PACK_CONFIG_METADATA)
    clip_normalized_max: float = field(default=10.0, metadata=_PACK_CONFIG_METADATA)
    # FAST tokenizer input cap; Genesis lowers FAST and Flow with independent caps.
    fast_clip_normalized_max: float = field(default=1.0, metadata=_PACK_CONFIG_METADATA)
    train_skip_outlier_threshold: float = field(default=20.0, metadata=_PACK_CONFIG_METADATA)
    train_max_sequence_length: int = field(default=4096, metadata=_PACK_CONFIG_METADATA)
    # False marks the gripper dimensions passthrough on the loaded stats. Only applies to
    # stats read from a path: stats restored from serialized state already carry the mask.
    normalize_gripper: bool = field(default=True, metadata=_PACK_CONFIG_METADATA)
    action_feature_names: list[str] | None = field(default=None, metadata=_PACK_CONFIG_METADATA)
    state_feature_names: list[str] | None = field(default=None, metadata=_PACK_CONFIG_METADATA)

    def __post_init__(self) -> None:
        self.mharmony_version = normalize_mharmony_version_marker(self.mharmony_version)
        require_closed_loop_safe_isaac_profile(
            profile_id=self.normalization_profile_id,
            profile_scope=self.normalization_profile_scope,
            validation_status=self.normalization_validation_status,
        )
        if bool(self.suite_stats_path) != bool(self.suite_by_task_index_path):
            raise ValueError(
                "Perceptron Isaac per-suite normalization needs suite_stats_path and "
                "suite_by_task_index_path together; got "
                f"suite_stats_path={self.suite_stats_path!r}, "
                f"suite_by_task_index_path={self.suite_by_task_index_path!r}."
            )
        if self.image_preprocessing not in {"stretch", "letterbox"}:
            raise ValueError(
                "Perceptron Isaac image_preprocessing must be 'stretch' or 'letterbox'; "
                f"got {self.image_preprocessing!r}."
            )
        self._stream_builder = None
        self._stats = None
        self._fast_processor = None
        self._suite_stats = None
        self._suite_by_task_index = None
        # Run-local, factory-injected: the training dataset's declared fps, used to
        # cross-check the checkpoint-owned clock in _frame_index_anchors. Never serialized.
        self._dataset_declared_fps = None
        self._warned_dataset_fps_mismatch = False

    def reset(self) -> None:
        return None

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for config_field in dataclass_fields(self):
            name = config_field.name
            if not self._serialize_pack_fields and config_field.metadata.get("pack_config"):
                continue
            value = getattr(self, name)
            config[name] = list(value) if name in _LIST_CONFIG_FIELD_NAMES else value
        if self._serialize_pack_fields and self.enabled:
            config["stats_path"] = None
            config["native_stats_path"] = None
            if self.render_metadata is not None:
                config["native_render_metadata_path"] = None
        return config

    @staticmethod
    def _observation_image_keys(observation: dict[str, Any]) -> list[str]:
        return sorted(key for key in observation if str(key).startswith(f"{OBS_IMAGES}."))

    def _resolve_image_keys(self, observation: dict[str, Any]) -> list[str]:
        if self.image_keys:
            missing = [key for key in self.image_keys if key not in observation]
            if not missing:
                return list(self.image_keys)
            if not self.allow_image_key_fallback:
                raise ValueError(f"Perceptron Isaac image_keys missing from observation: {missing}.")

        keys = self._observation_image_keys(observation)
        if not keys:
            raise ValueError("Perceptron Isaac requires at least one image observation.")

        resolved: list[str] = []
        for cam in self.camera_order:
            suffixes = _CAMERA_FALLBACK_SUFFIXES.get(cam, (cam,))
            match = next((key for key in keys if key.rsplit(".", 1)[-1] in suffixes), None)
            if match is not None:
                resolved.append(match)
        if len(resolved) == len(self.camera_order):
            return resolved
        if self.allow_image_key_fallback and len(keys) >= len(self.camera_order):
            # Positional slice of alphabetically-sorted keys: 'arm'/'base' would feed the scene
            # slot from 'arm' purely because it sorts first. Suffix matching above already
            # covers every known naming, so reaching here means the cameras are genuinely
            # unrecognized and the mapping is a guess. Say so loudly -- a silent miswire is
            # indistinguishable from a working eval until the success rate collapses.
            guess = keys[: len(self.camera_order)]
            logging.warning(
                "Perceptron Isaac could not match cameras %s by name; falling back to "
                "alphabetical order %s. Verify this mapping is correct -- set image_keys "
                "explicitly, or allow_image_key_fallback=false to make this an error.",
                list(self.camera_order),
                dict(zip(self.camera_order, guess, strict=True)),
            )
            return guess
        missing_cameras = [cam for cam in self.camera_order if not any(key.endswith(cam) for key in keys)]
        raise ValueError(
            f"Could not resolve Perceptron Isaac cameras {missing_cameras} from observation keys {keys}."
        )

    def _infer_batch_size(self, observation: dict[str, Any], action: Tensor | None) -> int:
        if action is not None:
            return int(action.shape[0]) if action.ndim > 1 else 1
        state = observation.get(OBS_STATE)
        if torch.is_tensor(state) or isinstance(state, np.ndarray):
            ndim = getattr(state, "ndim", 0)
            if ndim >= 2:
                return int(state.shape[0])
            return 1
        for key in self._resolve_image_keys(observation):
            value = observation[key]
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                return int(value.shape[0]) if getattr(value, "ndim", 0) >= 4 else 1
        return 1

    def _extract_image_tensor(self, observation: dict[str, Any], batch_size: int) -> np.ndarray:
        values = [_to_numpy(observation[key]) for key in self._resolve_image_keys(observation)]
        per_camera: list[np.ndarray] = []
        for value in values:
            arr = value
            if arr.ndim == 5:
                # [B, T, C, H, W] / [B, T, H, W, C]
                if arr.shape[0] != batch_size:
                    raise ValueError(f"Image batch size {arr.shape[0]} does not match {batch_size}.")
                per_camera.append(arr)
            elif arr.ndim == 4:
                if batch_size == 1 and arr.shape[0] != 1 and arr.shape[1] in {1, 3, 4}:
                    per_camera.append(arr[None, ...])  # [T, C, H, W] -> [B, T, C, H, W]
                elif batch_size == 1 and arr.shape[0] != 1 and arr.shape[-1] in {1, 3, 4}:
                    per_camera.append(arr[None, ...])  # [T, H, W, C] -> [B, T, H, W, C]
                else:
                    if arr.shape[0] != batch_size:
                        raise ValueError(f"Image batch size {arr.shape[0]} does not match {batch_size}.")
                    per_camera.append(arr[:, None, ...])  # [B, C, H, W] / [B, H, W, C] -> [B, T, ...]
            elif arr.ndim == 3:
                if batch_size != 1:
                    raise ValueError("Unbatched image can only be used with batch_size=1.")
                per_camera.append(arr[None, None, ...])
            else:
                raise ValueError(f"Unsupported Isaac image shape {arr.shape}.")
        per_camera = [self._conform_image_geometry(arr) for arr in per_camera]
        images = np.stack(per_camera, axis=2)  # [B, T, N_cam, ...]
        self._validate_image_tensor(images)
        return images

    def _conform_image_geometry(self, images: np.ndarray) -> np.ndarray:
        """Bring one camera's [B, T, C, H, W] / [B, T, H, W, C] stack to ``image_size``.

        Applies the checkpoint's saved geometry contract (``image_preprocessing``) with the
        same primitive the camera deployment path uses, so dataset-resolution training
        frames and camera-resolution serving frames cannot drift. Contract-size input is
        returned untouched, which is what the serving byte-parity gates see.
        """
        expected = tuple(int(x) for x in self.image_size)
        channels_first = int(images.shape[-3]) in {1, 3, 4}
        if not channels_first and int(images.shape[-1]) not in {1, 3, 4}:
            return images  # undecidable layout; the size validator reports it
        height, width = (
            (int(images.shape[-2]), int(images.shape[-1]))
            if channels_first
            else (int(images.shape[-3]), int(images.shape[-2]))
        )
        if (height, width) == expected:
            return images

        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvf

        from lerobot.transforms import letterbox_resize_image

        tensor = torch.from_numpy(np.ascontiguousarray(images))
        if not channels_first:
            tensor = tensor.movedim(-1, -3)
        if self.image_preprocessing == "letterbox":
            resized = letterbox_resize_image(
                tensor, expected, fill=0.0, interpolation=InterpolationMode.BILINEAR, antialias=True
            )
        else:
            resized = tvf.resize_image(
                tensor, list(expected), interpolation=InterpolationMode.BILINEAR, antialias=True
            )
        if not channels_first:
            resized = resized.movedim(-3, -1)
        return resized.contiguous().numpy()

    def _validate_image_tensor(self, images: np.ndarray) -> None:
        expected_h, expected_w = tuple(int(x) for x in self.image_size)
        height, width = int(images.shape[-2]), int(images.shape[-1])
        channels_first = images.ndim >= 5 and int(images.shape[-3]) in {1, 3, 4}
        if not channels_first and images.ndim >= 5 and int(images.shape[-1]) in {1, 3, 4}:
            height, width = int(images.shape[-3]), int(images.shape[-2])
        if (height, width) != (expected_h, expected_w):
            raise ValueError(
                "Perceptron Isaac render processor requires observations matching training resolution; "
                f"got {(height, width)}, expected {(expected_h, expected_w)}."
            )

    def _extract_state_tensor(self, observation: dict[str, Any], batch_size: int) -> np.ndarray:
        if OBS_STATE not in observation:
            raise ValueError("Perceptron Isaac requires observation.state.")
        state = _to_numpy(observation[OBS_STATE]).astype(np.float32, copy=False)
        if state.ndim == 3:
            if state.shape[0] != batch_size:
                raise ValueError(f"State batch size {state.shape[0]} does not match {batch_size}.")
            return state
        if state.ndim == 2:
            if batch_size == 1 and state.shape[0] != 1:
                return state[None, ...]  # [T, D] -> [B, T, D]
            if state.shape[0] != batch_size:
                raise ValueError(f"State batch size {state.shape[0]} does not match {batch_size}.")
            return state[:, None, :]
        if state.ndim == 1:
            if batch_size != 1:
                raise ValueError("Unbatched state can only be used with batch_size=1.")
            return state[None, None, :]
        raise ValueError(f"Unsupported Isaac state shape {state.shape}.")

    def _ensure_renderer(self) -> None:
        if self._stream_builder is not None:
            return
        if self.inference_backend != "native_mharmony":
            raise ValueError(
                "Perceptron Isaac pack only supports inference_backend='native_mharmony', "
                f"got {self.inference_backend!r}."
            )
        if self._stats is None:
            stats_path = self.native_stats_path or self.stats_path
            if not stats_path:
                raise RuntimeError(
                    "Perceptron Isaac pack requires serialized stats state or native_stats_path."
                )
            self._stats = _maybe_passthrough_grippers(self, load_isaac_stats(stats_path))
            require_closed_loop_safe_isaac_stats(self._stats)
        self._validate_target_fps(self._stats.target_fps, artifact="normalization stats")
        if self.render_metadata is not None:
            metadata = IsaacMharmonyRenderMetadata(**self.render_metadata)
        elif self.native_render_metadata_path:
            metadata = IsaacMharmonyRenderMetadata.from_json_file(self.native_render_metadata_path)
        else:
            metadata = IsaacMharmonyRenderMetadata(
                camera_order=list(self.camera_order),
                image_size=[int(self.image_size[0]), int(self.image_size[1])],
                n_obs_steps=int(self.n_obs_steps),
                action_horizon=int(self.chunk_size),
                action_dim=int(self.action_dim),
                proprio_dim=int(self.proprio_dim),
                vector_max_states=int(self.vector_max_states),
                target_fps=self._stats.target_fps,
                dataset_name=str(self.dataset_name),
                robot_type=str(self.robot_type),
                action_conditioning=bool(self.action_conditioning),
                action_conditioning_role=str(self.action_conditioning_role),
                mistake_conditioning=bool(self.mistake_conditioning),
                patch_size=int(self.patch_size),
                max_num_patches=self.max_num_patches if self.max_num_patches is not None else 576,
                min_num_patches=self.min_num_patches,
                pixel_shuffle_scale=int(self.pixel_shuffle_scale),
                temporal_patch_size=int(self.temporal_patch_size),
                mharmony_version=self.mharmony_version,
            )
        self._validate_target_fps(metadata.target_fps, artifact="render metadata")
        self._stream_builder = IsaacNativeMharmonyRenderer(metadata=metadata, stats=self._stats)

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        if self._stats is None:
            stats_path = self.native_stats_path or self.stats_path
            if not stats_path:
                return {}
            self._stats = _maybe_passthrough_grippers(self, load_isaac_stats(stats_path))
        # Never freeze an ablation-labeled profile into a saved package.
        require_closed_loop_safe_isaac_stats(self._stats)
        return isaac_stats_to_state_dict(self._stats)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if not state:
            return
        stats = isaac_stats_from_state_dict(state)
        # The labels now survive the round trip, so re-check them on the way back in rather
        # than trusting a bundle that hydration would otherwise skip validating.
        require_closed_loop_safe_isaac_stats(stats)
        self._stats = stats
        self._stream_builder = None

    def _validate_target_fps(self, actual: float | None, *, artifact: str) -> None:
        if self.target_fps is None:
            return
        if actual is None or abs(float(actual) - float(self.target_fps)) > 1e-9:
            raise ValueError(
                f"ISAAC {artifact} target_fps={actual!r} does not match packaged "
                f"target_fps={self.target_fps!r}."
            )

    @staticmethod
    def _anchor_numeric_values(
        raw: Any,
        batch_size: int,
        *,
        field_name: str,
    ) -> list[float] | None:
        if raw is None:
            return None
        if torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        elif isinstance(raw, np.ndarray):
            arr = raw
        elif isinstance(raw, (list, tuple)):
            if raw and any(
                torch.is_tensor(item) or isinstance(item, (np.ndarray, list, tuple)) for item in raw
            ):
                values = []
                for item in raw:
                    item_arr = item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item)
                    values.append(float(item_arr.reshape(-1)[-1]))
                arr = np.asarray(values, dtype=np.float32)
            else:
                arr = np.asarray(raw)
        else:
            arr = np.asarray(raw)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 0:
            values = [float(arr.item())]
        elif arr.ndim >= 2:
            values = [float(value) for value in arr.reshape(arr.shape[0], -1)[:, -1].tolist()]
        else:
            flat = arr.reshape(-1)
            values = [float(flat[-1])] if batch_size == 1 else [float(value) for value in flat.tolist()]

        if len(values) == 1 and batch_size != 1:
            values = values * batch_size
        if len(values) != batch_size:
            raise ValueError(f"Expected {batch_size} {field_name} values, got {len(values)}.")
        return values

    def _target_fps(self, *, default: float | None = None) -> float | None:
        self._ensure_renderer()
        fps = None
        if self._stream_builder is not None:
            metadata = getattr(self._stream_builder, "metadata", None)
            stats = getattr(self._stream_builder, "stats", None)
            fps = getattr(metadata, "target_fps", None) or getattr(stats, "target_fps", None)
        if fps is None:
            return default
        fps = float(fps)
        return fps if fps > 0.0 else default

    def _has_explicit_anchor_source(self, complementary: dict[str, Any]) -> bool:
        return any(
            complementary.get(key) is not None
            for key in (
                PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY,
                "anchor_timestamp_seconds",
                "frame_index",
                "timestamp",
            )
        )

    def _ensure_suite_stats(self) -> bool:
        """Load the scope -> stats table and the task_index -> scope map. True when routing is on."""
        if not self.suite_stats_path:
            return False
        if self._suite_stats is None:
            suite_stats = {
                scope: _maybe_passthrough_grippers(self, block)
                for scope, block in load_isaac_suite_stats(self.suite_stats_path).items()
            }
            self._require_suite_geometry_matches_contract(suite_stats)
            raw = json.loads(Path(self.suite_by_task_index_path).read_text())
            suite_by_task_index = {int(k): str(v) for k, v in raw.items()}
            missing = sorted(set(suite_by_task_index.values()) - set(suite_stats))
            if missing:
                raise ValueError(
                    f"Task index map references scopes with no normalization block: {missing}. "
                    f"Known scopes: {sorted(suite_stats)}."
                )
            # Publish only after both loads and the validation pass, so a raise here cannot
            # leave the step half-initialized (stats set, map None) for a retried call.
            self._suite_stats = suite_stats
            self._suite_by_task_index = suite_by_task_index
        return True

    def _require_suite_geometry_matches_contract(
        self, suite_stats: dict[str, IsaacNormalizationStats]
    ) -> None:
        """Reject a suite table that contradicts the checkpoint contract, before the first render.

        Per-suite routing hands ``build_training`` a block from this table INSTEAD of the
        step's serialized stats, so the package's own stats are bypassed for every sample.
        Differing quantiles are the entire point -- LIBERO suites are genuinely not
        interchangeable -- so comparing values here would be wrong. Geometry is a different
        matter: an ``action_horizon`` other than the chunk the model emits, or an
        ``action_dim``/``target_fps`` other than the checkpoint's, means the run is training
        on a contract the checkpoint cannot serve.

        Checked here rather than in ``_require_restored_stats_match_config_paths`` because
        that guard runs only on the serving/eval load, where per-suite routing is inert; this
        runs on the training render path, which is the only consumer.
        """
        reference = self._stats
        problems: list[str] = []
        for scope, block in sorted(suite_stats.items()):
            if int(block.action_horizon) != int(self.chunk_size):
                problems.append(
                    f"{scope}: action_horizon {block.action_horizon} != chunk_size {self.chunk_size}"
                )
            if reference is None:
                continue
            if int(block.action_dim) != int(reference.action_dim):
                problems.append(
                    f"{scope}: action_dim {block.action_dim} != checkpoint {reference.action_dim}"
                )
            if (
                block.target_fps is not None
                and reference.target_fps is not None
                and abs(float(block.target_fps) - float(reference.target_fps)) > 1e-6
            ):
                problems.append(
                    f"{scope}: target_fps {block.target_fps} != checkpoint {reference.target_fps}"
                )
        if problems:
            raise ValueError(
                "Perceptron Isaac per-suite normalization table disagrees with the checkpoint "
                f"contract: {'; '.join(problems)}. Rebuild it against this checkpoint "
                "with action_horizon equal to the checkpoint chunk_size."
            )

    def _suite_stats_for(self, complementary: dict[str, Any], batch_size: int):
        """Per-sample normalization stats, or None when per-suite routing is off."""
        if not self._ensure_suite_stats():
            return None
        task_indices = self._anchor_numeric_values(
            complementary.get("task_index"), batch_size, field_name="task_index"
        )
        if task_indices is None:
            raise ValueError(
                "Perceptron Isaac per-suite normalization requires task_index in the batch; "
                "without it a sample cannot be attributed to a LIBERO suite."
            )
        resolved = []
        for value in task_indices:
            key = int(round(value))
            scope = self._suite_by_task_index.get(key)
            if scope is None:
                raise ValueError(
                    f"task_index {key} has no LIBERO suite mapping; "
                    f"known indices {min(self._suite_by_task_index)}..{max(self._suite_by_task_index)}."
                )
            resolved.append(self._suite_stats[scope])
        return resolved

    def _frame_index_anchors(self, complementary: dict[str, Any], batch_size: int) -> list[float] | None:
        """Derive episode-relative model time from the frame counter and the checkpoint's fps.

        Prefer this over the dataset's stored ``timestamp``. Converted LeRobot LIBERO datasets
        (``lerobot/libero_spatial_image``, ``allenai/MolmoAct2-LIBERO-Dataset``) declare
        ``fps=10`` for demos robosuite recorded at 20 Hz, so their ``timestamp``
        (``frame_index / declared_fps``) runs 2x long -- a 116-frame episode claims 11.6 s
        instead of 5.8 s. Online rollout clocks itself as ``frame_index / target_fps``, so
        trusting the declared rate would train the model on one time base and serve it on
        another. The ISAAC contract owns the time base via ``target_fps``; when a dataset's
        declared rate does agree, this returns exactly the stored timestamp.
        """
        fps = self._target_fps()
        if fps is None:
            return None
        frames = self._anchor_numeric_values(
            complementary.get("frame_index"), batch_size, field_name="frame_index"
        )
        if frames is None:
            return None
        declared = self._dataset_declared_fps
        if declared is not None and abs(float(declared) - fps) > 1e-3:
            if str(getattr(self, "dataset_name", "")) == "libero":
                # The converted LIBERO datasets named above misdeclare their rate; the
                # contract clock deliberately overrides, but say so once instead of silently.
                if not self._warned_dataset_fps_mismatch:
                    self._warned_dataset_fps_mismatch = True
                    logging.warning(
                        "Dataset declares fps=%s but the ISAAC contract owns target_fps=%s; "
                        "anchoring model time at frame_index/%s (converted LIBERO datasets "
                        "misdeclare their recording rate).",
                        declared,
                        fps,
                        fps,
                    )
            else:
                raise ValueError(
                    f"Dataset declares fps={declared} but the checkpoint contract owns "
                    f"target_fps={fps}: frame-index anchors would train on a "
                    f"{float(declared) / fps:.3g}x-scaled episode clock relative to the data's "
                    "real timeline. Fix the dataset's declared fps if it is wrong, or finetune "
                    "a checkpoint whose target_fps matches the recording rate."
                )
        return [float(frame) / fps for frame in frames]

    def _anchor_timestamps(self, complementary: dict[str, Any], batch_size: int) -> list[float | None]:
        explicit_anchors = self._anchor_numeric_values(
            complementary.get(PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY),
            batch_size,
            field_name=PERCEPTRON_ISAAC_ANCHOR_TIMESTAMPS_KEY,
        )
        if explicit_anchors is None:
            explicit_anchors = self._anchor_numeric_values(
                complementary.get("anchor_timestamp_seconds"),
                batch_size,
                field_name="anchor_timestamp_seconds",
            )
        if explicit_anchors is not None:
            anchors = [float(value) for value in explicit_anchors]
        else:
            frame_anchors = self._frame_index_anchors(complementary, batch_size)
            if frame_anchors is not None:
                anchors = frame_anchors
            else:
                timestamps = self._anchor_numeric_values(
                    complementary.get("timestamp"), batch_size, field_name="timestamp"
                )
                if timestamps is None:
                    return [None] * batch_size
                anchors = [float(value) for value in timestamps]

        for anchor in anchors:
            if not np.isfinite(anchor) or anchor < 0.0:
                raise ValueError(
                    f"Perceptron Isaac external timestamps must be finite and nonnegative; got {anchor!r}."
                )
        return anchors

    def _render_streams(
        self,
        *,
        windows: list[list[dict[str, Any]]],
        tasks: list[str],
        actions: Tensor | None,
        complementary: dict[str, Any],
        anchors: list[float | None],
    ):
        self._ensure_renderer()
        if actions is not None:
            if any(anchor is None for anchor in anchors):
                raise ValueError(
                    "ISAAC training requires an explicit episode-relative dataset timestamp; "
                    "the stateless processor never synthesizes model time."
                )
            if actions.ndim != 3:
                raise ValueError(
                    f"ISAAC training actions must be shaped [B, H, D], got {tuple(actions.shape)}."
                )
            if actions.shape[0] != len(windows):
                raise ValueError(
                    f"ISAAC action batch {actions.shape[0]} does not match observation batch {len(windows)}."
                )
            fast_processor = self._ensure_fast_processor()
            pad_rows = self._action_padding_rows(complementary, len(windows))
            suite_stats = self._suite_stats_for(complementary, len(windows))
            streams = []
            kept_indices = []
            dtype = _dtype_from_name(self.dtype)
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            max_sequence_length = int(self.train_max_sequence_length)
            # build_training uses max_sequence_length only for its final fail-loud check;
            # passing this sentinel moves that check up here where over-length can be
            # handled without a raise.
            unchecked_length = 2**31 - 1

            def render_sample(index: int, *, skip_outlier_threshold: float, max_sequence_length: int):
                return self._stream_builder.build_training(
                    stats=None if suite_stats is None else suite_stats[index],
                    observation_window=windows[index],
                    prompt=tasks[index],
                    action_chunk=actions[index].detach().cpu().numpy(),
                    fast_processor=fast_processor,
                    action_is_pad=pad_rows[index],
                    clip_normalized_actions=bool(self.train_clip_normalized_actions),
                    clip_normalized_max=float(self.clip_normalized_max),
                    fast_clip_normalized_max=float(self.fast_clip_normalized_max),
                    skip_outlier_threshold=skip_outlier_threshold,
                    max_sequence_length=max_sequence_length,
                    device=self.device,
                    dtype=dtype,
                    patch_size=self.patch_size,
                    max_num_patches=self.max_num_patches,
                    min_num_patches=self.min_num_patches,
                    pixel_shuffle_scale=self.pixel_shuffle_scale,
                    temporal_patch_size=self.temporal_patch_size,
                    anchor_timestamp_seconds=float(anchors[index]),
                )

            # In a distributed run a data-dependent raise on one rank aborts it alone while
            # its peers advance into the loss's dist.all_reduce, hanging the job until the
            # NCCL timeout -- so there over-length samples are skipped like outliers, with
            # the length enforced here rather than inside build_training. Single-process
            # runs keep the renderer's fail-loud over-length error.
            for index in range(len(windows)):
                try:
                    stream = render_sample(
                        index,
                        skip_outlier_threshold=float(self.train_skip_outlier_threshold),
                        max_sequence_length=unchecked_length if distributed else max_sequence_length,
                    )
                except IsaacActionOutlierError:
                    continue
                if distributed and stream.shape[1] > max_sequence_length:
                    logging.warning(
                        "Skipping ISAAC training sample %d: stream uses %d tokens, exceeding "
                        "max_sequence_length=%d.",
                        index,
                        int(stream.shape[1]),
                        max_sequence_length,
                    )
                    continue
                streams.append(stream)
                kept_indices.append(index)
            if distributed and not self._all_distributed_ranks_have_usable_samples(bool(streams)):
                # Every rank participates in this validation collective before any rank can enter
                # denominator/model collectives. Fail together instead of keeping an outlier with
                # the threshold disabled or letting one rank raise alone and hang its peers.
                raise IsaacActionOutlierError(
                    "At least one distributed rank has no usable samples after ISAAC outlier and "
                    f"max_sequence_length={max_sequence_length} filtering. Aborting this update on "
                    "every rank before loss collectives; verify normalization stats, suite routing, "
                    "and the sequence-length budget."
                )
            if not streams:
                # No collectives to keep in lockstep: an all-outlier microbatch means the
                # normalization stats do not match the data, and training on it clipped
                # would only converge to a silently wrong policy.
                raise IsaacActionOutlierError(
                    "Every ISAAC training sample in this microbatch exceeded "
                    f"train_skip_outlier_threshold={self.train_skip_outlier_threshold}. This "
                    "usually means the normalization stats do not match the dataset (wrong "
                    "stats file or wrong per-suite routing)."
                )
            complementary[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] = kept_indices
            return self._stream_builder.collate(streams)
        if len(windows) != 1:
            raise NotImplementedError("Perceptron Isaac eval rendering currently supports batch_size=1.")
        dtype = _dtype_from_name(self.dtype)
        fast_processor = (
            self._ensure_fast_processor() if self._stream_builder.metadata.action_conditioning else None
        )
        stream = self._stream_builder.build(
            observation_window=windows[0],
            prompt=tasks[0],
            fast_processor=fast_processor,
            device=self.device,
            dtype=dtype,
            patch_size=self.patch_size,
            max_num_patches=self.max_num_patches,
            min_num_patches=self.min_num_patches,
            pixel_shuffle_scale=self.pixel_shuffle_scale,
            temporal_patch_size=self.temporal_patch_size,
            scene_events=None,
            anchor_timestamp_seconds=anchors[0],
        )
        if stream.shape[1] > int(self.train_max_sequence_length):
            raise ValueError(
                f"ISAAC inference stream uses {stream.shape[1]} tokens, exceeding the "
                f"checkpoint-trained max_sequence_length={self.train_max_sequence_length}."
            )
        return stream

    def _all_distributed_ranks_have_usable_samples(self, locally_usable: bool) -> bool:
        """Collectively decide batch validity on a backend-compatible device."""
        backend = str(torch.distributed.get_backend()).lower()
        status_device = torch.device(self.device) if "nccl" in backend else torch.device("cpu")
        status = torch.tensor(int(locally_usable), device=status_device, dtype=torch.int32)
        torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MIN)
        return bool(status.item())

    def _ensure_fast_processor(self):
        if self._fast_processor is not None:
            return self._fast_processor
        path = getattr(self, "fast_processor_path", None)
        tree_sha256 = getattr(self, "fast_processor_tree_sha256", None)
        if not path or not tree_sha256:
            raise RuntimeError(
                "ISAAC training requires fast_processor_path and fast_processor_tree_sha256 "
                "from an imported checkpoint package."
            )
        self._fast_processor = load_fast_action_processor(path, expected_tree_sha256=tree_sha256)
        return self._fast_processor

    def _action_padding_rows(self, complementary: dict[str, Any], batch_size: int) -> list[list[bool]]:
        raw = complementary.get("action_is_pad")
        if raw is None:
            return [[False] * int(self.chunk_size) for _ in range(batch_size)]
        array = _to_numpy(raw).astype(bool, copy=False)
        if array.ndim == 1:
            if batch_size != 1:
                raise ValueError("Unbatched action_is_pad is valid only for batch_size=1.")
            array = array[None, :]
        expected = (batch_size, int(self.chunk_size))
        if array.shape != expected:
            raise ValueError(f"ISAAC action_is_pad must have shape {expected}, got {array.shape}.")
        return [[bool(value) for value in row] for row in array.tolist()]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        if not isinstance(observation, dict):
            raise ValueError("Perceptron Isaac render expected an observation dictionary.")
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        action_raw = transition.get(TransitionKey.ACTION)
        action = torch.as_tensor(action_raw, dtype=torch.float32) if action_raw is not None else None
        if action is not None and action.ndim == 2:
            action = action.unsqueeze(1)
        if action is None and self.suite_stats_path:
            raise RuntimeError(
                "Perceptron Isaac per-suite normalization is training-only: a merged multi-suite "
                "checkpoint has no single correct state/action normalization frame for inference. "
                "Create a single-suite serving package with that suite's stats before evaluation or deployment."
            )

        batch_size = self._infer_batch_size(observation, action)
        if action is None and batch_size != 1:
            raise NotImplementedError(
                f"Perceptron Isaac eval/serving currently supports batch_size=1; got batch_size={batch_size}."
            )
        images = self._extract_image_tensor(observation, batch_size)
        states = self._extract_state_tensor(observation, batch_size)
        if images.shape[:2] != states.shape[:2]:
            raise ValueError(f"Image/state window shape mismatch: {images.shape[:2]} vs {states.shape[:2]}.")

        task_source = complementary.get("task")
        if task_source is None:
            task_source = observation.get("task")
        if task_source is None:
            task_source = observation.get("observation.language")
        if task_source is None:
            task_source = complementary.get("language_instruction")
        tasks = coerce_batch_text(
            task_source,
            batch_size,
            item_label="Isaac task strings",
            normalizer=normalize_perceptron_isaac_task_text if self.normalize_language else None,
        )
        complementary["task"] = tasks

        if action is None and images.shape[1] == 1 and not self._has_explicit_anchor_source(complementary):
            if self._stats is not None:
                # Rendering is handed to the policy-owned online step; hand it the
                # restored stats as well so both sides normalize with one source.
                complementary[PERCEPTRON_ISAAC_SERVING_STATS_KEY] = self._stats
            complementary[self.metadata_key] = {
                "task": tasks,
                "camera_order": list(self.camera_order),
                "image_keys": self._resolve_image_keys(observation),
                "anchor_timestamp_seconds": None,
                "n_obs_steps": int(images.shape[1]),
                "state_shape": tuple(int(x) for x in states.shape),
                "image_shape": tuple(int(x) for x in images.shape),
                "action_shape": None,
                "objective": "flow",
                "render_skipped": True,
                "skip_reason": "online_single_step_without_anchor",
            }
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
            return transition

        windows = [
            build_perceptron_isaac_observation_window(images[idx], states[idx], tuple(self.camera_order))
            for idx in range(batch_size)
        ]
        previous_action_batches = complementary.get(PERCEPTRON_ISAAC_PREVIOUS_ACTIONS_KEY)
        if previous_action_batches is not None:
            if not isinstance(previous_action_batches, list) or len(previous_action_batches) != batch_size:
                raise ValueError(
                    f"{PERCEPTRON_ISAAC_PREVIOUS_ACTIONS_KEY} must contain one history per batch item."
                )
            for window, previous_actions in zip(windows, previous_action_batches, strict=True):
                if not isinstance(previous_actions, list) or len(previous_actions) != len(window):
                    raise ValueError(
                        f"{PERCEPTRON_ISAAC_PREVIOUS_ACTIONS_KEY} histories must align with observation windows."
                    )
                for frame, previous_action in zip(window, previous_actions, strict=True):
                    frame["previous_action"] = previous_action
        anchors = self._anchor_timestamps(complementary, batch_size)
        complementary[self.stream_key] = self._render_streams(
            windows=windows,
            tasks=tasks,
            actions=action,
            complementary=complementary,
            anchors=anchors,
        )
        complementary[self.metadata_key] = {
            "task": tasks,
            "camera_order": list(self.camera_order),
            "image_keys": self._resolve_image_keys(observation),
            "anchor_timestamp_seconds": anchors,
            "n_obs_steps": int(images.shape[1]),
            "state_shape": tuple(int(x) for x in states.shape),
            "image_shape": tuple(int(x) for x in images.shape),
            "action_shape": tuple(int(x) for x in action.shape) if action is not None else None,
            "objective": self.objective if action is not None else "flow",
        }
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="perceptron_isaac_mharmony_pack")
@dataclass
class PerceptronIsaacMharmonyPackProcessorStep(PerceptronIsaacRenderProcessorStep):
    """Native mharmony pack step for ISAAC inference and joint training streams."""

    _serialize_pack_fields: ClassVar[bool] = True


@ProcessorStepRegistry.register(name="perceptron_isaac_action_unnormalize")
@dataclass
class PerceptronIsaacActionUnnormalizeProcessorStep(ProcessorStep):
    """Convert native Isaac normalized actions to LIBERO robot actions."""

    enabled: bool = True
    stats_path: str | None = None
    clip_normalized_max: float = 10.0
    clip_action_pose: bool = True
    gripper_binary_to_signed: bool = False
    normalization_profile_id: str | None = None
    normalization_profile_scope: str | None = None
    normalization_validation_status: str | None = None
    # See the pack step: only applies to stats read from `stats_path`.
    normalize_gripper: bool = True
    action_feature_names: list[str] | None = None
    state_feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        require_closed_loop_safe_isaac_profile(
            profile_id=self.normalization_profile_id,
            profile_scope=self.normalization_profile_scope,
            validation_status=self.normalization_validation_status,
        )
        self._stats = None

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stats_path": None if self.enabled else self.stats_path,
            "clip_normalized_max": self.clip_normalized_max,
            "clip_action_pose": self.clip_action_pose,
            "gripper_binary_to_signed": self.gripper_binary_to_signed,
            "normalization_profile_id": self.normalization_profile_id,
            "normalization_profile_scope": self.normalization_profile_scope,
            "normalization_validation_status": self.normalization_validation_status,
            "normalize_gripper": self.normalize_gripper,
            "action_feature_names": self.action_feature_names,
            "state_feature_names": self.state_feature_names,
        }

    def _ensure_stats(self):
        if self._stats is not None:
            return self._stats
        if not self.stats_path:
            raise RuntimeError(
                "PerceptronIsaacActionUnnormalizeProcessorStep requires stats_path when enabled."
            )
        self._stats = _maybe_passthrough_grippers(self, load_isaac_stats(self.stats_path))
        return self._stats

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        if self._stats is None:
            if not self.stats_path:
                return {}
            self._stats = _maybe_passthrough_grippers(self, load_isaac_stats(self.stats_path))
        return isaac_stats_to_state_dict(self._stats)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state:
            self._stats = isaac_stats_from_state_dict(state)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not torch.is_tensor(action):
            raise ValueError(
                f"Perceptron Isaac action postprocessor expected a tensor, got {type(action).__name__}."
            )
        stats = self._ensure_stats()
        action_cpu = action.detach().float().cpu().numpy()
        if action_cpu.shape[-1] != int(stats.action_dim):
            raise ValueError(
                f"Perceptron Isaac normalized action last dim {action_cpu.shape[-1]} != stats dim {stats.action_dim}."
            )
        original_shape = action_cpu.shape
        flat = action_cpu.reshape(-1, original_shape[-1])
        robot = np.asarray(
            unnormalize_isaac_actions(
                flat,
                stats.action,
                clip=True,
                clip_normalized_max=float(self.clip_normalized_max),
            ),
            dtype=np.float32,
        ).reshape(original_shape)
        if self.clip_action_pose and robot.shape[-1] > 1:
            robot[..., :-1] = np.clip(robot[..., :-1], -1.0, 1.0)
        if self.gripper_binary_to_signed:
            robot[..., -1] = 1.0 - 2.0 * robot[..., -1]
        out = transition.copy()
        out[TransitionKey.ACTION] = torch.as_tensor(robot, dtype=torch.float32, device=action.device)
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class _PerceptronIsaacJointFrameTransformStep(ProcessorStep):
    """Shared joint-frame transform; concrete registry entries select the tensor and direction."""

    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None

    _container_key: ClassVar[TransitionKey | None] = None
    _value_key: ClassVar[str] = TransitionKey.ACTION
    _inverse: ClassVar[bool] = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.joint_signs is None or self.joint_offsets is None:
            return transition

        container = transition
        if self._container_key is not None:
            container = transition.get(self._container_key)
            if not isinstance(container, dict) or self._value_key not in container:
                return transition
        elif transition.get(self._value_key) is None:
            return transition

        out = transition.copy()
        if self._container_key is not None:
            container = container.copy()
            out[self._container_key] = container
        else:
            container = out

        values = torch.as_tensor(container[self._value_key], dtype=torch.float32).clone()
        n = len(self.joint_signs)
        signs = values.new_tensor(self.joint_signs)
        offsets = values.new_tensor(self.joint_offsets)
        if self._inverse:
            values[..., :n] = signs * (values[..., :n] - offsets)
        else:
            values[..., :n] = signs * values[..., :n] + offsets
        container[self._value_key] = values
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"joint_signs": self.joint_signs, "joint_offsets": self.joint_offsets}


@ProcessorStepRegistry.register(name="perceptron_isaac_state_frame_transform")
@dataclass
class PerceptronIsaacStateFrameTransformStep(_PerceptronIsaacJointFrameTransformStep):
    """Map observed joint state from the current-LeRobot convention to the checkpoint's (old) frame.

    Runs on every observation BEFORE Isaac history/render/proprio-normalization:
        q_model = signs * q_lerobot + offsets
    No-op when both fields are None. See PerceptronIsaacConfig.joint_signs/joint_offsets.
    """

    _container_key: ClassVar[TransitionKey | None] = TransitionKey.OBSERVATION
    _value_key: ClassVar[str] = OBS_STATE


@ProcessorStepRegistry.register(name="perceptron_isaac_training_action_frame_transform")
@dataclass
class PerceptronIsaacTrainingActionFrameTransformStep(_PerceptronIsaacJointFrameTransformStep):
    """Map LeRobot training targets into the checkpoint's joint frame.

    Dataset state and action are both recorded in the current LeRobot arm frame.
    ISAAC proprio and action quantile statistics are in the legacy/model frame, so
    the two inputs to the training stream must receive the same forward transform::

        q_model = signs * q_lerobot + offsets

    The step is intentionally part of the preprocessor. During serving there is no
    action in the input transition and it is a no-op; during training it executes
    immediately before the mharmony pack step normalizes action supervision.
    """


@ProcessorStepRegistry.register(name="perceptron_isaac_action_frame_transform")
@dataclass
class PerceptronIsaacActionFrameTransformStep(_PerceptronIsaacJointFrameTransformStep):
    """Map predicted joint targets from the checkpoint's (old) frame back to current-LeRobot.

    Runs AFTER Isaac action unnormalization, immediately before the robot action is sent:
        q_lerobot = signs * (q_model - offsets)
    Exact inverse of PerceptronIsaacStateFrameTransformStep for signs in {+1, -1}. No-op when None.
    """

    _inverse: ClassVar[bool] = True


def make_perceptron_isaac_pre_post_processors(
    config: PerceptronIsaacConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the native eval/serving pre/postprocessor pair."""
    _adopt_dataset_state_feature_names(config, dataset_meta)
    # The LIBERO env (envs/libero.py) emits the wrist camera as `observation.images.image2`, but a checkpoint
    # trained on a dataset that names it `wrist_image` (e.g. the IPEC no-noops 20fps set) expects that key.
    # Map the env's image2 onto the policy's 2nd camera (camera_order[1]) so eval obs match the trained layout.
    # Guarded + no-op when the batch lacks image2 (training batches already carry the right key) or when the
    # 2nd camera is itself "image2" (a checkpoint trained on image2 needs no rename).
    rename_map: dict[str, str] = {}
    cams = config.camera_order
    if len(cams) >= 2 and cams[1] != "image2":
        rename_map["observation.images.image2"] = f"observation.images.{cams[1]}"
    render_step_class: type[PerceptronIsaacRenderProcessorStep] = PerceptronIsaacMharmonyPackProcessorStep
    render_step_kwargs = {
        "native_render_metadata_path": config.native_render_metadata_path,
        "native_stats_path": config.native_stats_path,
        "fast_processor_path": config.fast_processor_path,
        "fast_processor_tree_sha256": config.fast_processor_tree_sha256,
        "mharmony_version": config.mharmony_version,
        "action_dim": int(config.action_dim),
        "proprio_dim": int(config.proprio_dim),
        "vector_max_states": int(config.vector_max_states),
        "dataset_name": str(config.dataset_name),
        "robot_type": str(config.robot_type),
        "target_fps": config.target_fps,
        "action_conditioning": bool(config.action_conditioning),
        "action_conditioning_role": str(config.action_conditioning_role),
        "mistake_conditioning": bool(config.mistake_conditioning),
        "normalization_profile_id": config.normalization_profile_id,
        "normalization_profile_scope": config.normalization_profile_scope,
        "normalization_validation_status": config.normalization_validation_status,
        "render_metadata": (
            IsaacMharmonyRenderMetadata.from_json_file(config.native_render_metadata_path).to_json_dict()
            if config.native_render_metadata_path and Path(config.native_render_metadata_path).is_file()
            else None
        ),
        "train_clip_normalized_actions": bool(config.train_clip_normalized_actions),
        "clip_normalized_max": float(config.clip_normalized_max),
        "fast_clip_normalized_max": float(config.fast_clip_normalized_max),
        "train_skip_outlier_threshold": float(config.train_skip_outlier_threshold),
        "train_max_sequence_length": int(config.train_max_sequence_length),
        "normalize_gripper": bool(config.normalize_gripper),
        "action_feature_names": list(config.action_feature_names or []) or None,
        "state_feature_names": list(config.state_feature_names or []) or None,
    }
    render_enabled = True
    render_step = render_step_class(
        enabled=render_enabled,
        inference_backend=str(config.inference_backend),
        config_toml_path=config.config_toml_path,
        stats_path=config.stats_path,
        camera_order=list(config.camera_order),
        image_size=tuple(config.image_size),
        image_preprocessing=str(config.image_preprocessing),
        image_keys=[f"observation.images.{cam}" for cam in config.camera_order],
        allow_image_key_fallback=bool(config.allow_image_key_fallback),
        normalize_language=bool(config.normalize_task_text),
        n_obs_steps=int(config.n_obs_steps),
        chunk_size=int(config.chunk_size),
        objective=str(config.objective),
        patch_size=int(config.render_patch_size),
        max_num_patches=config.render_max_num_patches,
        min_num_patches=config.render_min_num_patches,
        pixel_shuffle_scale=int(config.render_pixel_shuffle_scale),
        temporal_patch_size=int(config.render_temporal_patch_size),
        device=str(config.device),
        dtype="bfloat16",
        suite_stats_path=config.suite_stats_path,
        suite_by_task_index_path=config.suite_by_task_index_path,
        **render_step_kwargs,
    )
    declared_fps = getattr(dataset_meta, "fps", None)
    if declared_fps is not None:
        # Hand the anchor derivation the training dataset's declared clock so a genuine
        # rate mismatch with the checkpoint's target_fps fails loudly instead of silently
        # scaling training model time.
        render_step._dataset_declared_fps = float(declared_fps)
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map=rename_map),
        AddBatchDimensionProcessorStep(),
        # Joint-frame adapter (obs): current-LeRobot -> checkpoint (old) convention, BEFORE the render
        # step normalizes proprio and before the policy's online history captures the state. No-op when
        # joint_signs/offsets are None.
        PerceptronIsaacStateFrameTransformStep(
            joint_signs=config.joint_signs, joint_offsets=config.joint_offsets
        ),
        # Training targets come from the same current-LeRobot frame as observation.state.
        # Convert them to the checkpoint frame before the pack step applies ISAAC stats.
        # Serving inputs contain no action, so this is a no-op online.
        PerceptronIsaacTrainingActionFrameTransformStep(
            joint_signs=config.joint_signs, joint_offsets=config.joint_offsets
        ),
        render_step,
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = []
    output_steps.append(
        PerceptronIsaacActionUnnormalizeProcessorStep(
            enabled=True,
            stats_path=config.native_stats_path or config.stats_path,
            clip_normalized_max=float(config.clip_normalized_max),
            clip_action_pose=bool(config.clip_action_pose),
            gripper_binary_to_signed=bool(config.gripper_binary_to_signed),
            normalize_gripper=bool(config.normalize_gripper),
            action_feature_names=list(config.action_feature_names or []) or None,
            state_feature_names=list(config.state_feature_names or []) or None,
            normalization_profile_id=config.normalization_profile_id,
            normalization_profile_scope=config.normalization_profile_scope,
            normalization_validation_status=config.normalization_validation_status,
        )
    )
    # Joint-frame adapter (action): checkpoint (old) -> current-LeRobot convention, AFTER unnormalize
    # and before the robot action leaves the pipeline. No-op when joint_signs/offsets are None.
    output_steps.append(
        PerceptronIsaacActionFrameTransformStep(
            joint_signs=config.joint_signs, joint_offsets=config.joint_offsets
        )
    )
    output_steps.append(DeviceProcessorStep(device="cpu"))
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    if dataset_stats is not None:
        _install_dataset_stats(
            preprocessor,
            postprocessor,
            _dataset_stats_to_isaac_stats(
                dataset_stats, config, dataset_fps=getattr(dataset_meta, "fps", None)
            ),
        )
    _record_config_stats_export(config, preprocessor)
    return preprocessor, postprocessor


# The shared trainer always supplies normalizer/unnormalizer overrides built from dataset
# statistics, but ISAAC owns normalization itself: its steps carry the checkpoint's quantile
# stats and ``normalization_mapping`` is IDENTITY throughout, so the pipeline has no such step
# and ``from_pretrained`` would reject the unused override keys outright.
_INAPPLICABLE_OVERRIDE_KEYS = frozenset({"normalizer_processor", "unnormalizer_processor"})


def _drop_inapplicable_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Filter out overrides for steps the ISAAC pipelines deliberately do not contain."""
    if not overrides:
        return {}
    return {key: value for key, value in overrides.items() if key not in _INAPPLICABLE_OVERRIDE_KEYS}


def _require_restored_stats_match_config_paths(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    config: PerceptronIsaacConfig,
) -> None:
    """Fail loudly at load when serialized pipeline stats and config-path stats diverge.

    The pipeline safetensors hold the stats the checkpoint actually trained with; the
    policy-owned consumers (online render step, settle branch) path-load
    ``config.native_stats_path``. If the two disagree, proprio would be normalized in a
    different frame than the model was finetuned on -- the serving split-brain this
    module now guards against. Values are compared, never labels: base and finetune
    stats are known to share a ``profile_id`` while carrying different quantiles.
    """
    restored: list[tuple[str, Any]] = []
    for step in preprocessor.steps:
        if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep) and step._stats is not None:
            restored.append(("preprocessor pack step", step))
    for step in postprocessor.steps:
        if isinstance(step, PerceptronIsaacActionUnnormalizeProcessorStep) and step._stats is not None:
            restored.append(("postprocessor unnormalize step", step))
    if not restored:
        return

    for (name_a, step_a), (name_b, step_b) in zip(restored, restored[1:], strict=False):
        diffs = isaac_stats_value_differences(step_a._stats, step_b._stats)
        if diffs:
            raise RuntimeError(
                f"Perceptron Isaac package normalization is internally inconsistent: the "
                f"{name_a} and {name_b} serialized different stats values: {'; '.join(diffs)}."
            )

    stats_path = config.native_stats_path or config.stats_path
    if not stats_path:
        return
    path = Path(str(stats_path))
    if not path.is_file():
        # Saved packages are deliberately portable without their source stats file
        # (the stats ride in the pipeline safetensors), and the serialized stats are
        # handed to the policy at serving. Divergence fails loudly below; absence
        # only means the path-load fallback is unavailable.
        logging.warning(
            "Perceptron Isaac config native_stats_path %r does not exist; serving "
            "will rely exclusively on the pipeline's serialized stats.",
            str(stats_path),
        )
        return

    for name, step in restored:
        path_stats = _maybe_passthrough_grippers(step, load_isaac_stats(path))
        diffs = isaac_stats_value_differences(step._stats, path_stats)
        if diffs:
            raise RuntimeError(
                "Perceptron Isaac package normalization is split-brained: the "
                f"{name}'s serialized stats (the training-time ground truth) disagree "
                f"with config.native_stats_path={str(path)!r}: {'; '.join(diffs)}. "
                "Serving this package would normalize proprio with the wrong quantiles. "
                "Fix the package: export the serialized stats next to the checkpoint "
                "(lerobot.policies.perceptron_isaac.isaac_stats.save_isaac_stats) and "
                "point config.json's native_stats_path at that file."
            )


def _reconcile_normalization_recipe(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    config: PerceptronIsaacConfig,
) -> None:
    """Apply the config's normalization/clip recipe to pipelines loaded from a package.

    `PolicyProcessorPipeline.from_pretrained` restores every step field from the base
    package's JSON and applies only the caller's device-style overrides. Without this,
    a finetune's `--policy.clip_normalized_max` (and siblings) are recorded in
    train_config.json but never reach the processors, so the run silently trains and
    serves on the base's values while its own config claims otherwise.

    Safe on both paths: at eval the config is loaded *from* the checkpoint, so the values
    already agree and this is a no-op. Only an explicit finetune override changes anything,
    and every change is logged so a silent divergence cannot recur.
    """
    fields: list[tuple[Any, str, Any]] = []
    for step in preprocessor.steps:
        if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep):
            fields += [
                (step, "clip_normalized_max", float(config.clip_normalized_max)),
                (step, "fast_clip_normalized_max", float(config.fast_clip_normalized_max)),
                (step, "train_clip_normalized_actions", bool(config.train_clip_normalized_actions)),
                (step, "train_skip_outlier_threshold", float(config.train_skip_outlier_threshold)),
                (step, "train_max_sequence_length", int(config.train_max_sequence_length)),
                (step, "normalize_gripper", bool(config.normalize_gripper)),
                (step, "action_feature_names", list(config.action_feature_names or []) or None),
                (step, "state_feature_names", list(config.state_feature_names or []) or None),
            ]
    for step in postprocessor.steps:
        if isinstance(step, PerceptronIsaacActionUnnormalizeProcessorStep):
            fields += [
                (step, "clip_normalized_max", float(config.clip_normalized_max)),
                (step, "normalize_gripper", bool(config.normalize_gripper)),
                (step, "action_feature_names", list(config.action_feature_names or []) or None),
                (step, "state_feature_names", list(config.state_feature_names or []) or None),
            ]

    changed = []
    for step, name, value in fields:
        current = getattr(step, name, None)
        if current != value:
            setattr(step, name, value)
            changed.append(f"{type(step).__name__}.{name}: {current} -> {value}")
    if changed:
        logging.info(
            "Perceptron Isaac normalization recipe overridden from policy config:\n  %s",
            "\n  ".join(changed),
        )


def _reconcile_training_action_frame_step(
    preprocessor: PolicyProcessorPipeline,
    config: PerceptronIsaacConfig,
) -> None:
    """Install/validate the v2 training action transform in loaded packages.

    Imported 65k bases were serialized before the training-side bug was found, so
    their preprocessor JSON cannot contain this step. Adding it in memory lets a new
    fine-tune start safely from that immutable base. Newly saved checkpoints retain
    the step and take the validation branch on reload.
    """

    existing = [
        step
        for step in preprocessor.steps
        if isinstance(step, PerceptronIsaacTrainingActionFrameTransformStep)
    ]
    if len(existing) > 1:
        raise ValueError("ISAAC preprocessor contains duplicate training action-frame transforms.")
    if existing:
        step = existing[0]
        if step.joint_signs != config.joint_signs or step.joint_offsets != config.joint_offsets:
            raise ValueError(
                "Saved ISAAC training action-frame transform does not match policy joint_signs/joint_offsets."
            )
        return

    pack_indices = [
        index
        for index, step in enumerate(preprocessor.steps)
        if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    ]
    if len(pack_indices) != 1:
        raise ValueError(
            "ISAAC preprocessor must contain exactly one mharmony pack step before the "
            "training action-frame transform can be reconciled."
        )
    preprocessor.steps.insert(
        pack_indices[0],
        PerceptronIsaacTrainingActionFrameTransformStep(
            joint_signs=config.joint_signs,
            joint_offsets=config.joint_offsets,
        ),
    )
    logging.info("Installed ISAAC v2 training action-frame transform before mharmony normalization.")


def make_perceptron_isaac_pre_post_processors_from_pretrained(
    config: PerceptronIsaacConfig,
    pretrained_path: str,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load the paired ISAAC pipelines and reconnect package-local assets.

    Imported packages inline normalization into processor safetensors and
    render metadata into the pack-step config. Relative paths are retained as
    a compatibility fallback for older serialized pipelines only.
    """
    _adopt_dataset_state_feature_names(config, dataset_meta)
    # Mirror the fresh-construction path, where the pack step's __post_init__ raises for
    # a half-configured pair; without this the loaded-package path silently disables
    # per-suite routing instead.
    if bool(config.suite_stats_path) != bool(config.suite_by_task_index_path):
        raise ValueError(
            "Perceptron Isaac per-suite normalization needs suite_stats_path and "
            "suite_by_task_index_path together; got "
            f"suite_stats_path={config.suite_stats_path!r}, "
            f"suite_by_task_index_path={config.suite_by_task_index_path!r}."
        )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=preprocessor_config_filename,
        overrides=_drop_inapplicable_overrides(preprocessor_overrides),
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        revision=revision,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        config_filename=postprocessor_config_filename,
        overrides=_drop_inapplicable_overrides(postprocessor_overrides),
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        revision=revision,
    )
    _reconcile_training_action_frame_step(preprocessor, config)
    _reconcile_normalization_recipe(preprocessor, postprocessor, config)
    package_root = Path(pretrained_path)
    if package_root.is_dir():
        for step in (*preprocessor.steps, *postprocessor.steps):
            for attribute in (
                "native_render_metadata_path",
                "native_stats_path",
                "stats_path",
                "fast_processor_path",
            ):
                raw = getattr(step, attribute, None)
                if not raw or Path(str(raw)).is_absolute():
                    continue
                candidate = package_root / str(raw)
                if candidate.exists():
                    setattr(step, attribute, str(candidate))
        normalization_path = package_root / "policy_normalization.json"
        for step in preprocessor.steps:
            if (
                not isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
                or step.render_metadata is None
                or step.render_metadata.get("training_proprio_contract") is not None
            ):
                continue
            if not normalization_path.is_file():
                raise FileNotFoundError(
                    "Existing ISAAC package needs policy_normalization.json to hydrate its "
                    "Genesis-compatible training prompt contract."
                )
            metadata = hydrate_isaac_training_proprio_contract(
                IsaacMharmonyRenderMetadata(**step.render_metadata),
                policy_normalization_path=normalization_path,
                policy_state_dataset=config.policy_state_dataset,
            )
            step.render_metadata = metadata.to_json_dict()
            step._stream_builder = None
    declared_fps = getattr(dataset_meta, "fps", None)
    for step in preprocessor.steps:
        if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep):
            step.device = str(config.device)
            if declared_fps is not None:
                # Hand the anchor derivation the training dataset's declared clock so a
                # genuine rate mismatch with the checkpoint's target_fps fails loudly
                # instead of silently scaling training model time.
                step._dataset_declared_fps = float(declared_fps)
            # This is a policy/run choice, like per-suite stats, rather than an
            # immutable property of the imported package. Without reconciliation,
            # --policy.normalize_task_text updates config.json while the loaded pack
            # step silently keeps the base package's old value.
            step.normalize_language = bool(config.normalize_task_text)
            # Geometry contract follows config.json the same way: a letterbox finetune of
            # a package whose serialized step predates the field must not silently fall
            # back to the step default ("stretch").
            step.image_preprocessing = str(config.image_preprocessing)
            # Per-suite normalization is a run-level choice, so it arrives on the policy config
            # (--policy.suite_stats_path) rather than in the package's saved step config. Without
            # re-applying it here the loaded step keeps suite_stats_path=None, _ensure_suite_stats
            # returns False, and every sample is silently normalized with the single set of
            # quantiles baked into the package -- one suite's frame applied to all of them. That
            # trains a model whose only valid eval normalization is that one suite's, which looks
            # exactly like a policy that learned one suite and failed the rest.
            if config.suite_stats_path and config.suite_by_task_index_path:
                step.suite_stats_path = str(config.suite_stats_path)
                step.suite_by_task_index_path = str(config.suite_by_task_index_path)
                step._suite_stats = None  # force reload with the new paths
                step._suite_by_task_index = None
                logging.info(
                    "Perceptron Isaac per-suite normalization enabled from policy config: %s",
                    config.suite_stats_path,
                )
    if dataset_stats is not None:
        _install_dataset_stats(
            preprocessor,
            postprocessor,
            _dataset_stats_to_isaac_stats(
                dataset_stats, config, dataset_fps=getattr(dataset_meta, "fps", None)
            ),
        )
    elif dataset_meta is None:
        # Serving/eval load: the package must not be split-brained between its
        # serialized pipeline stats and the config's path-loaded stats.
        #
        # Training RESUME must be exempt: the trainer passes dataset_meta but omits
        # dataset_stats on resume, the serialized pipeline stats are authoritative
        # there, and the resumed train_config.json still records the LAUNCH-time
        # native_stats_path (the base package's, for every pre-export finetune) --
        # running the check would fail every healthy finetune resume.
        _require_restored_stats_match_config_paths(preprocessor, postprocessor, config)
    _record_config_stats_export(config, preprocessor)
    return preprocessor, postprocessor
