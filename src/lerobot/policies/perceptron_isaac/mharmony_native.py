"""Native mharmony boundary for Perceptron Isaac eval.

This module is the native eval/serving boundary: LeRobot code may call the
Genesis mharmony adapter layer here, but it must not directly depend on Genesis
runtime config objects, inference stream builders, or Genesis normalization/loss
runtime.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.policies.processor_utils import coerce_rgb_uint8

from .configuration_perceptron_isaac import PerceptronIsaacConfig
from .fast_processor import encode_fast_action_tokens
from .isaac_stats import (
    IsaacNormalizationStats,
    load_isaac_stats,
    normalize_isaac_actions,
    normalize_isaac_proprio,
)
from .mharmony_adapter import (
    load_mharmony_encoding_cache,
    load_mharmony_stream_adapter,
    load_mharmony_types,
    load_qwen35_image_processor,
    to_local_tensor_stream,
)

DEFAULT_LIBERO_ACTION_LAYOUT: tuple[str, ...] = ()
DEFAULT_LIBERO_CAMERA_VIEWS = ("primary", "wrist")
DEFAULT_MHARMONY_RESERVED_TOKEN_GROUPS = (
    {
        "tokenizer": "physical-intelligence/fast",
        "size": 2048,
    },
)
GENESIS_TEXT_TYPE_TAG = "genesis_text_type"
GENESIS_TEXT_TYPE_TIMESTAMP = "timestamp"
GENESIS_TEXT_TYPE_ACTION = "action"
GENESIS_TEXT_TYPE_ACTION_C = "action_c"
FLOW_ACTION_TOKEN_GROUP_NAME = "flow_action"  # nosec B105 - stream label, not a credential
FAST_ACTION_TOKEN_GROUP_NAME = "fast_action"  # nosec B105 - stream label, not a credential
FAST_ACTION_TOKENIZER_NAME = "physical-intelligence/fast"
FLOW_ACTION_MARKER_TOKEN_ID = 0
ACTION_MESSAGE_STOP_TOKEN = "return"  # nosec B105 - harmony control-token name, not a credential
DEFAULT_INFERENCE_FPS = 20.0
DEFAULT_ANCHOR_TIMESTAMP_SECONDS = 30.0
QWEN_TIMESTAMP_PRECISION = 2


class IsaacActionOutlierError(ValueError):
    """A normalized action exceeds the checkpoint's training skip threshold."""


@dataclass(frozen=True)
class IsaacMharmonyRenderMetadata:
    """Checkpoint-local metadata needed by the native mharmony renderer."""

    schema: str = "perceptron_isaac_mharmony_render_v1"
    camera_order: list[str] = field(default_factory=lambda: ["image", "wrist_image"])
    image_size: list[int] = field(default_factory=lambda: [256, 256])
    n_obs_steps: int = 3
    action_horizon: int = 30
    action_dim: int = 7
    proprio_dim: int = 8
    vector_max_states: int = 128
    target_fps: float | None = 20.0
    dataset_name: str | None = None
    robot_type: str | None = "generic"
    objective_form: str = "Flow"
    action_representation: str = "absolute"
    control_mode: str | None = "ee"
    action_layout: list[str] = field(default_factory=lambda: list(DEFAULT_LIBERO_ACTION_LAYOUT))
    camera_views: list[str] = field(default_factory=lambda: list(DEFAULT_LIBERO_CAMERA_VIEWS))
    include_scene_description: bool = False
    rtc_prefix_length: int = 0
    mharmony_encoding_name: str = "QWEN35_HARMONY"
    mharmony_reserved_token_groups: list[dict[str, Any]] = field(
        default_factory=lambda: [dict(group) for group in DEFAULT_MHARMONY_RESERVED_TOKEN_GROUPS]
    )
    timestamp_precision: int = QWEN_TIMESTAMP_PRECISION
    message_stop_token: str = ACTION_MESSAGE_STOP_TOKEN
    patch_size: int = 16
    pixel_shuffle_scale: int = 2
    temporal_patch_size: int = 2
    max_num_patches: int | None = 576
    min_num_patches: int | None = None
    training_proprio_contract: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls, config: PerceptronIsaacConfig, *, stats_fps: float | None = None
    ) -> IsaacMharmonyRenderMetadata:
        return cls(
            camera_order=list(config.camera_order),
            image_size=[int(config.image_size[0]), int(config.image_size[1])],
            n_obs_steps=int(config.n_obs_steps),
            action_horizon=int(config.chunk_size),
            action_dim=int(config.action_dim),
            proprio_dim=int(config.proprio_dim),
            vector_max_states=int(config.vector_max_states),
            target_fps=float(stats_fps) if stats_fps is not None else DEFAULT_INFERENCE_FPS,
            dataset_name=str(config.dataset_name),
            robot_type=str(config.robot_type),
            include_scene_description=bool(config.include_scene_description),
            rtc_prefix_length=int(config.rtc_prefix_length),
            patch_size=int(config.render_patch_size),
            pixel_shuffle_scale=int(config.render_pixel_shuffle_scale),
            temporal_patch_size=int(config.render_temporal_patch_size),
            max_num_patches=config.render_max_num_patches
            if config.render_max_num_patches is not None
            else 576,
            min_num_patches=config.render_min_num_patches,
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> IsaacMharmonyRenderMetadata:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Isaac mharmony metadata must be a JSON object, got {type(raw).__name__}.")
        return cls(**raw)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_for_config(self, config: PerceptronIsaacConfig) -> None:
        if self.schema != "perceptron_isaac_mharmony_render_v1":
            raise ValueError(f"Unsupported Isaac mharmony metadata schema {self.schema!r}.")
        if tuple(self.camera_order) != tuple(config.camera_order):
            raise ValueError(
                f"Isaac mharmony camera_order {tuple(self.camera_order)} does not match config {tuple(config.camera_order)}."
            )
        if tuple(int(x) for x in self.image_size) != tuple(int(x) for x in config.image_size):
            raise ValueError(
                f"Isaac mharmony image_size {tuple(self.image_size)} does not match config {tuple(config.image_size)}."
            )
        if int(self.n_obs_steps) != int(config.n_obs_steps):
            raise ValueError(f"Isaac mharmony n_obs_steps {self.n_obs_steps} != config {config.n_obs_steps}.")
        if int(self.action_horizon) != int(config.chunk_size):
            raise ValueError(
                f"Isaac mharmony action_horizon {self.action_horizon} != chunk_size {config.chunk_size}."
            )
        if int(self.action_dim) != int(config.action_dim):
            raise ValueError(f"Isaac mharmony action_dim {self.action_dim} != config {config.action_dim}.")
        if int(self.proprio_dim) != int(config.proprio_dim):
            raise ValueError(f"Isaac mharmony proprio_dim {self.proprio_dim} != config {config.proprio_dim}.")
        if not str(self.robot_type or "").strip():
            raise ValueError(
                "Isaac mharmony metadata requires robot_type for the robotics configuration block."
            )
        if str(self.robot_type) != str(config.robot_type):
            raise ValueError(
                f"Isaac mharmony robot_type {self.robot_type!r} does not match config {config.robot_type!r}."
            )
        if str(self.objective_form) != "Flow":
            raise ValueError(
                f"Native Isaac mharmony only supports objective_form='Flow', got {self.objective_form!r}."
            )
        if str(self.action_representation) != "absolute":
            raise ValueError(
                "Native Isaac mharmony only supports absolute flow actions for LIBERO, "
                f"got action_representation={self.action_representation!r}."
            )
        if int(self.timestamp_precision) != QWEN_TIMESTAMP_PRECISION:
            raise ValueError(
                f"Native Isaac mharmony expects Qwen timestamp precision {QWEN_TIMESTAMP_PRECISION}, "
                f"got {self.timestamp_precision}."
            )
        if not self.camera_views:
            raise ValueError("Isaac mharmony metadata requires a non-empty camera_views list.")
        if self.target_fps is not None and float(self.target_fps) <= 0:
            raise ValueError(f"Isaac mharmony target_fps must be positive when set, got {self.target_fps!r}.")
        if config.target_fps is not None and (
            self.target_fps is None or abs(float(self.target_fps) - float(config.target_fps)) > 1e-9
        ):
            raise ValueError(
                f"Isaac mharmony target_fps {self.target_fps} does not match config {config.target_fps}."
            )
        if self.dataset_name is not None and str(self.dataset_name) != str(config.dataset_name):
            raise ValueError(
                f"Isaac mharmony dataset_name {self.dataset_name!r} does not match config {config.dataset_name!r}."
            )
        if bool(self.include_scene_description):
            raise ValueError(
                "Native Isaac mharmony metadata must use include_scene_description=false for no-scene LIBERO."
            )
        if int(self.rtc_prefix_length) != 0:
            raise ValueError(
                "Native Isaac mharmony metadata must use rtc_prefix_length=0 for MolmoAct LIBERO eval."
            )
        if self.training_proprio_contract is not None:
            _validate_training_proprio_contract(
                self.training_proprio_contract,
                dataset_name=self.dataset_name,
                proprio_dim=self.proprio_dim,
            )


def build_isaac_training_proprio_contract(
    *,
    robot_type: str,
    dataset_name: str,
    proprio_dim: int,
    contract_version: int,
    contract_hash: str,
) -> dict[str, Any]:
    """Compile the supported Genesis policy-state ABI into movable render metadata."""
    profile_components: dict[str, tuple[int, list[tuple[str, int, int]]]] = {
        "bi_yam": (
            14,
            [
                ("left_arm_joints", 0, 6),
                ("left_gripper", 6, 7),
                ("right_arm_joints", 7, 13),
                ("right_gripper", 13, 14),
            ],
        ),
        "so100_so101": (
            6,
            [
                ("arm_joints", 0, 5),
                ("gripper", 5, 6),
            ],
        ),
        "libero": (
            8,
            [
                ("end_effector_position", 0, 3),
                ("end_effector_orientation", 3, 7),
                ("gripper", 7, 8),
            ],
        ),
    }
    if robot_type not in profile_components:
        raise ValueError(f"Unsupported ISAAC training proprio robot_type {robot_type!r}.")
    expected_dim, component_ranges = profile_components[robot_type]
    if int(proprio_dim) != expected_dim:
        raise ValueError(f"ISAAC {robot_type} training proprio dim {proprio_dim} != expected {expected_dim}.")
    contract = {
        "schema": "perceptron_isaac_training_proprio_contract_v1",
        "source": "robotics_configurations.yml",
        "dataset_name": str(dataset_name),
        "output_name": "state",
        "sources": [{"key": "state", "dim": expected_dim}],
        "components": [
            {
                "name": name,
                "source": "state",
                "range": [start, end],
                "channel": "proprio",
            }
            for name, start, end in component_ranges
        ],
        "excluded": [],
        "target_dim": expected_dim,
        "contract_version": int(contract_version),
        "contract_hash": str(contract_hash),
    }
    _validate_training_proprio_contract(
        contract,
        dataset_name=dataset_name,
        proprio_dim=proprio_dim,
    )
    return contract


def hydrate_isaac_training_proprio_contract(
    metadata: IsaacMharmonyRenderMetadata,
    *,
    policy_normalization_path: str | Path,
    policy_state_dataset: str | None = None,
) -> IsaacMharmonyRenderMetadata:
    """Hydrate packages imported before training prompt provenance was inlined."""
    if metadata.training_proprio_contract is not None:
        return metadata
    normalization = json.loads(Path(policy_normalization_path).read_text())
    dataset_identity = str(policy_state_dataset or metadata.dataset_name)
    candidates = [
        entry
        for entry in normalization.get("entries", [])
        if entry.get("policy_state_dataset") == dataset_identity and entry.get("objective") == "Flow"
    ]
    identities = {
        (int(entry["policy_state_contract_version"]), str(entry["policy_state_contract_hash"]))
        for entry in candidates
    }
    if len(identities) != 1:
        raise ValueError(
            "ISAAC package does not resolve exactly one Flow policy-state contract for "
            f"dataset {dataset_identity!r}: {sorted(identities)}."
        )
    contract_version, contract_hash = identities.pop()
    return replace(
        metadata,
        training_proprio_contract=build_isaac_training_proprio_contract(
            robot_type=str(metadata.robot_type),
            dataset_name=str(metadata.dataset_name),
            proprio_dim=int(metadata.proprio_dim),
            contract_version=contract_version,
            contract_hash=contract_hash,
        ),
    )


def _validate_training_proprio_contract(
    contract: dict[str, Any], *, dataset_name: str | None, proprio_dim: int
) -> None:
    required = {
        "schema",
        "source",
        "dataset_name",
        "output_name",
        "sources",
        "components",
        "excluded",
        "target_dim",
        "contract_version",
        "contract_hash",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError(f"ISAAC training proprio contract must contain exactly {sorted(required)}.")
    if contract["schema"] != "perceptron_isaac_training_proprio_contract_v1":
        raise ValueError(f"Unsupported ISAAC training proprio schema {contract['schema']!r}.")
    if contract["source"] != "robotics_configurations.yml" or contract["output_name"] != "state":
        raise ValueError("ISAAC training proprio contract has an unsupported source or output name.")
    if dataset_name is not None and str(contract["dataset_name"]) != str(dataset_name):
        raise ValueError("ISAAC training proprio dataset does not match render metadata.")
    if contract["target_dim"] != int(proprio_dim) or contract["contract_version"] != 1:
        raise ValueError("ISAAC training proprio target dim or contract version is invalid.")
    contract_hash = contract["contract_hash"]
    if (
        not isinstance(contract_hash, str)
        or len(contract_hash) != 64
        or any(char not in "0123456789abcdef" for char in contract_hash)
    ):
        raise ValueError("ISAAC training proprio contract_hash must be lowercase SHA-256 hex.")
    sources = contract["sources"]
    components = contract["components"]
    if sources != [{"key": "state", "dim": int(proprio_dim)}] or not isinstance(components, list):
        raise ValueError("ISAAC training proprio sources/components are invalid.")
    covered: list[int] = []
    for component in components:
        if not isinstance(component, dict) or set(component) != {"name", "source", "range", "channel"}:
            raise ValueError("ISAAC training proprio component is malformed.")
        bounds = component["range"]
        if (
            not isinstance(component["name"], str)
            or component["source"] != "state"
            or component["channel"] != "proprio"
            or not isinstance(bounds, list)
            or len(bounds) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)
            or bounds[0] < 0
            or bounds[1] <= bounds[0]
            or bounds[1] > int(proprio_dim)
        ):
            raise ValueError("ISAAC training proprio component range is invalid.")
        covered.extend(range(bounds[0], bounds[1]))
    if covered != list(range(int(proprio_dim))) or contract["excluded"] != []:
        raise ValueError("ISAAC training proprio components must cover the state exactly once.")


def load_native_render_metadata(
    config: PerceptronIsaacConfig, *, stats_fps: float | None = None
) -> IsaacMharmonyRenderMetadata:
    if config.native_render_metadata_path:
        metadata = IsaacMharmonyRenderMetadata.from_json_file(config.native_render_metadata_path)
    else:
        metadata = IsaacMharmonyRenderMetadata.from_config(config, stats_fps=stats_fps)
    if stats_fps is not None and metadata.target_fps is None:
        metadata = replace(metadata, target_fps=float(stats_fps))
    metadata.validate_for_config(config)
    return metadata


@dataclass(frozen=True)
class IsaacMharmonyContentPlan:
    """Serializable Isaac prompt plan before mharmony objects are constructed."""

    preamble: str
    anchor_timestamp_seconds: float
    observation_timestamps_seconds: list[float]
    user_content: list[dict[str, Any]]
    assistant_content: list[dict[str, Any]]


def load_native_isaac_stats(config: PerceptronIsaacConfig) -> IsaacNormalizationStats:
    stats_path = config.native_stats_path or config.stats_path
    if not stats_path:
        raise RuntimeError(
            "Perceptron Isaac inference_backend='native_mharmony' requires native_stats_path "
            "(or stats_path as a fallback)."
        )
    return load_isaac_stats(stats_path)


def build_isaac_mharmony_content_plan(
    *,
    observation_window: list[dict[str, Any]],
    prompt: str,
    metadata: IsaacMharmonyRenderMetadata,
    stats: IsaacNormalizationStats,
    anchor_timestamp_seconds: float | None = None,
    normalized_action_target: np.ndarray | None = None,
    fast_action_tokens: list[int] | None = None,
    action_is_pad: list[bool] | None = None,
    terminal: bool = False,
) -> IsaacMharmonyContentPlan:
    """Build the native LIBERO no-scene mharmony content plan.

    This mirrors the Genesis ``trajectory_flow_matching_pipeline`` inference
    contract for the supported native eval surface: Flow actions, explicit deltas,
    timestamps, RGB images, last-state-only proprio, and no scene description.
    """
    _validate_native_inference_contract(metadata, stats)
    history = _normalize_observation_history(observation_window, metadata)
    fps = _metadata_fps(metadata, stats)
    dt = 1.0 / fps if fps > 0 else 1.0 / DEFAULT_INFERENCE_FPS
    anchor_ts = (
        float(anchor_timestamp_seconds)
        if anchor_timestamp_seconds is not None
        else DEFAULT_ANCHOR_TIMESTAMP_SECONDS
    )
    if not np.isfinite(anchor_ts) or anchor_ts < 0.0:
        raise ValueError(
            f"Isaac mharmony anchor timestamp must be finite and nonnegative, got {anchor_ts!r}."
        )
    timestamps = [
        max(0.0, float(anchor_ts - (metadata.n_obs_steps - 1 - idx) * dt))
        for idx in range(metadata.n_obs_steps)
    ]

    preamble = _build_preamble(prompt=prompt, metadata=metadata, fps=fps)
    user_content: list[dict[str, Any]] = [
        {
            "kind": "text",
            "text": preamble,
            "metadata": {
                "trajectory_header": bool(prompt),
                "robotics_configuration": True,
                "robot_type": metadata.robot_type,
                "objective_form": metadata.objective_form,
            },
        }
    ]

    for step_idx, (frame, timestamp_seconds) in enumerate(zip(history, timestamps, strict=True)):
        user_content.append(_timestamp_content_item(step_idx, timestamp_seconds, metadata))
        for camera_idx, camera in enumerate(metadata.camera_order):
            user_content.append(
                {
                    "kind": "image",
                    "image": frame["images"][camera],
                    "camera": camera,
                    "media_id": f"image_{step_idx * len(metadata.camera_order) + camera_idx}",
                    "metadata": {
                        "trajectory_step": step_idx,
                        "trajectory_key": camera,
                        "timestamp_seconds": timestamp_seconds,
                    },
                }
            )
        if step_idx == metadata.n_obs_steps - 1:
            user_content.append(
                _vector_content_item(frame["proprio"], step_idx, timestamp_seconds, metadata, stats)
            )

    flow_tags = _flow_action_tags(
        metadata,
        stats,
        normalized_action_target=normalized_action_target,
        action_is_pad=action_is_pad,
        terminal=terminal,
    )
    training_prompt = normalized_action_target is not None or fast_action_tokens is not None
    if training_prompt:
        flow_tags = _project_genesis_training_action_tags(flow_tags)
    assistant_content = [
        {
            "kind": "tokens",
            "tokens": [FLOW_ACTION_MARKER_TOKEN_ID],
            "tags": flow_tags,
        }
    ]
    if fast_action_tokens is not None:
        assistant_content.append(
            {
                "kind": "tokens",
                "tokens": fast_action_tokens,
                "tags": _fast_action_tags(flow_tags),
            }
        )
    return IsaacMharmonyContentPlan(
        preamble=preamble,
        anchor_timestamp_seconds=anchor_ts,
        observation_timestamps_seconds=timestamps,
        user_content=user_content,
        assistant_content=assistant_content,
    )


class IsaacNativeMharmonyRenderer:
    """Native LeRobot renderer for Isaac LIBERO eval streams."""

    def __init__(self, *, metadata: IsaacMharmonyRenderMetadata, stats: IsaacNormalizationStats):
        _validate_native_inference_contract(metadata, stats)
        self.metadata = metadata
        self.stats = stats
        self._image_processor = None
        self._image_processor_key: tuple[Any, ...] | None = None

    def build(
        self,
        *,
        observation_window: list[dict[str, Any]],
        prompt: str,
        device: Any = "cuda",
        dtype: Any = None,
        patch_size: int = 16,
        max_num_patches: int | None = 576,
        min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 2,
        temporal_patch_size: int = 1,
        anchor_timestamp_seconds: float | None = None,
        scene_events: Any = None,
    ):
        if scene_events:
            raise ValueError("native_mharmony supports no-scene LIBERO inference only.")
        plan = build_isaac_mharmony_content_plan(
            observation_window=observation_window,
            prompt=prompt,
            metadata=self.metadata,
            stats=self.stats,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
        )
        return self._render_plan_to_tensor_stream(
            plan,
            device=device,
            dtype=dtype,
            patch_size=patch_size,
            max_num_patches=max_num_patches,
            min_num_patches=min_num_patches,
            pixel_shuffle_scale=pixel_shuffle_scale,
            temporal_patch_size=temporal_patch_size,
        )

    def build_training(
        self,
        *,
        observation_window: list[dict[str, Any]],
        prompt: str,
        action_chunk: Any,
        fast_processor: Any,
        action_is_pad: list[bool] | None = None,
        clip_normalized_actions: bool = True,
        clip_normalized_max: float = 10.0,
        fast_clip_normalized_max: float | None = None,
        skip_outlier_threshold: float = 20.0,
        max_sequence_length: int = 4096,
        device: Any = "cpu",
        dtype: Any = None,
        patch_size: int = 16,
        max_num_patches: int | None = 576,
        min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 2,
        temporal_patch_size: int = 1,
        anchor_timestamp_seconds: float | None = None,
        terminal: bool = False,
        stats: IsaacNormalizationStats | None = None,
    ):
        """Render one joint FAST+Flow training sample without Genesis transforms.

        ``terminal`` mirrors Genesis's ``_action_completion_metadata``: a chunk whose last
        action step is the trajectory's terminal step keeps the ``<|end|>`` footer, while every
        non-terminal chunk is supervised with ``<|return|>``. Genesis's LeRobot-native training
        builder emits a synthetic window that never reaches the terminal step, so non-terminal
        is the correct default here.

        ``stats`` overrides the builder's normalization for this one sample, which is what
        multi-suite LIBERO training needs: Genesis normalizes each LIBERO suite by its own
        quantiles, and the suites are not interchangeable -- end-effector height spans
        [0.916, 1.286] in libero_spatial but [0.010, 0.337] in libero_object, disjoint ranges
        that would push cross-normalized proprio far outside [-1, 1]. Defaults to the builder's
        own stats, so single-suite callers are unaffected.
        """
        stats = stats if stats is not None else self.stats
        action = np.asarray(action_chunk, dtype=np.float32)
        expected_shape = (int(self.metadata.action_horizon), int(self.metadata.action_dim))
        if action.shape != expected_shape:
            raise ValueError(
                f"ISAAC training action_chunk must have shape {expected_shape}, got {action.shape}."
            )
        normalized = normalize_isaac_actions(action, stats.action).astype(np.float32, copy=False)
        max_abs = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        if skip_outlier_threshold > 0.0 and max_abs > skip_outlier_threshold:
            raise IsaacActionOutlierError(
                f"ISAAC normalized action max {max_abs:.6g} exceeds skip threshold "
                f"{skip_outlier_threshold:.6g}."
            )
        # Genesis clips the two objectives independently: the FAST tokenizer transform and the
        # flow-target transform each carry their own cap. Sharing one clipped array would feed
        # the tokenizer an action distribution Genesis never trains it on.
        if clip_normalized_actions:
            fast_cap = clip_normalized_max if fast_clip_normalized_max is None else fast_clip_normalized_max
            fast_normalized = np.clip(normalized, -fast_cap, fast_cap)
            normalized = np.clip(normalized, -clip_normalized_max, clip_normalized_max)
        else:
            fast_normalized = normalized
        if action_is_pad is None:
            action_is_pad = [False] * expected_shape[0]
        if len(action_is_pad) != expected_shape[0] or any(
            not isinstance(value, bool) for value in action_is_pad
        ):
            raise ValueError("ISAAC action_is_pad must be a boolean list matching the action horizon.")
        group_sizes = [
            int(group["size"])
            for group in self.metadata.mharmony_reserved_token_groups
            if group.get("tokenizer") in (None, FAST_ACTION_TOKENIZER_NAME)
            and group.get("name") in (None, FAST_ACTION_TOKEN_GROUP_NAME)
        ]
        if len(group_sizes) != 1:
            raise ValueError(
                "ISAAC mharmony metadata must expose exactly one reserved group for the FAST action tokenizer."
            )
        fast_tokens = encode_fast_action_tokens(
            fast_processor,
            fast_normalized[None, ...],
            reserved_pool_size=group_sizes[0],
        )
        plan = build_isaac_mharmony_content_plan(
            observation_window=observation_window,
            prompt=prompt,
            metadata=self.metadata,
            stats=stats,
            anchor_timestamp_seconds=anchor_timestamp_seconds,
            normalized_action_target=normalized,
            fast_action_tokens=fast_tokens,
            action_is_pad=action_is_pad,
            terminal=bool(terminal),
        )
        stream = self._render_plan_to_tensor_stream(
            plan,
            device=device,
            dtype=dtype,
            patch_size=patch_size,
            max_num_patches=max_num_patches,
            min_num_patches=min_num_patches,
            pixel_shuffle_scale=pixel_shuffle_scale,
            temporal_patch_size=temporal_patch_size,
            # Genesis sets message_stop_token="return" for any chunk whose last action step is
            # not the trajectory terminal step (core/datasets/augment/trajectory.py), and
            # render_document_to_stream swaps the rendered <|end|> for <|return|> accordingly.
            # Terminal chunks keep <|end|>.
            rewrite_final_assistant_footer=not bool(terminal),
        )
        if stream.shape[1] > max_sequence_length:
            raise ValueError(
                f"ISAAC joint training stream uses {stream.shape[1]} tokens, exceeding "
                f"max_sequence_length={max_sequence_length}."
            )
        return stream

    @staticmethod
    def collate(streams: list[Any]):
        """Right-pad per-sample streams into one local TensorStream batch."""
        import torch

        from .tensor_stream import Event, TensorStream, TextType, create_stream

        if not streams:
            raise ValueError("ISAAC training collate received no streams.")
        samples = [stream.streams[0] if hasattr(stream, "streams") else stream for stream in streams]
        lengths = [sum(event.num_tokens() for event in sample.events) for sample in samples]
        max_length = max(lengths)
        padded = []
        for sample, length in zip(samples, lengths, strict=True):
            events = list(sample.events)
            if length < max_length:
                last_time = float(events[-1].time[1]) if events else 0.0
                events.append(
                    Event.from_text_tokens(
                        torch.zeros(max_length - length, dtype=torch.long),
                        time=(last_time + 1.0, last_time + 1.0),
                        type=TextType.padding,
                        role=None,
                    )
                )
            padded.append(create_stream(events, sample.priority, schedule=False))
        return TensorStream(padded)

    def _render_plan_to_tensor_stream(
        self,
        plan: IsaacMharmonyContentPlan,
        *,
        device: Any,
        dtype: Any,
        patch_size: int,
        max_num_patches: int | None,
        min_num_patches: int | None,
        pixel_shuffle_scale: int,
        temporal_patch_size: int,
        rewrite_final_assistant_footer: bool = True,
    ):
        types = load_mharmony_types()
        encoding_cache = load_mharmony_encoding_cache()
        stream_adapter = load_mharmony_stream_adapter()

        encoding = encoding_cache.get_harmony_encoding(
            self.metadata.mharmony_encoding_name,
            reserved_token_groups=self.metadata.mharmony_reserved_token_groups or None,
        )
        conversation = types.MHConversation(
            messages=[
                types.MHMessage(
                    role=types.HarmonyRole("user"),
                    content=[
                        _plan_item_to_mh_content(types, item, encoding=encoding, metadata=self.metadata)
                        for item in plan.user_content
                    ],
                ),
                types.MHMessage(
                    role=types.HarmonyRole("assistant"),
                    content=[
                        _plan_item_to_mh_content(types, item, encoding=encoding, metadata=self.metadata)
                        for item in plan.assistant_content
                    ],
                    channel="final",
                ),
            ]
        )
        harmony_conv = conversation.to_harmony()
        effective_max_num_patches = max_num_patches
        if effective_max_num_patches is None:
            effective_max_num_patches = self.metadata.max_num_patches
        effective_min_num_patches = min_num_patches
        if effective_min_num_patches is None:
            effective_min_num_patches = self.metadata.min_num_patches
        preprocess_config = {
            "patch_size": patch_size or self.metadata.patch_size,
            "max_num_patches": effective_max_num_patches,
            "min_num_patches": effective_min_num_patches,
            "pixel_shuffle_scale": pixel_shuffle_scale or self.metadata.pixel_shuffle_scale,
            "temporal_patch_size": temporal_patch_size or self.metadata.temporal_patch_size,
        }
        image_processor = self._ensure_image_processor(
            patch_size=preprocess_config["patch_size"],
            max_num_patches=preprocess_config["max_num_patches"],
            min_num_patches=preprocess_config["min_num_patches"],
            pixel_shuffle_scale=preprocess_config["pixel_shuffle_scale"],
            temporal_patch_size=preprocess_config["temporal_patch_size"],
        )
        if not hasattr(encoding, "render_conversation_multimodal_with_processors"):
            raise RuntimeError(
                "native_mharmony requires render_conversation_multimodal_with_processors support."
            )
        rendered = encoding.render_conversation_multimodal_with_processors(
            harmony_conv,
            preprocess_config=preprocess_config,
            image_processor=image_processor,
        )
        if rewrite_final_assistant_footer:
            rendered = _maybe_swap_final_assistant_footer(
                rendered,
                harmony_conv=harmony_conv,
                stop_token=self.metadata.message_stop_token,
                encoding=encoding,
            )
        stream = stream_adapter.rendered_stream_to_genesis_stream(
            rendered,
            encoding_name=self.metadata.mharmony_encoding_name,
        )
        return to_local_tensor_stream(stream, device=device, dtype=dtype)

    def _ensure_image_processor(
        self,
        *,
        patch_size: int,
        max_num_patches: int | None,
        min_num_patches: int | None,
        pixel_shuffle_scale: int,
        temporal_patch_size: int,
    ):
        key = (patch_size, max_num_patches, min_num_patches, pixel_shuffle_scale, temporal_patch_size)
        if self._image_processor is None or self._image_processor_key != key:
            qwen35_image_processor = load_qwen35_image_processor()
            self._image_processor = qwen35_image_processor.create_qwen35_mharmony_image_processor(
                patch_size=patch_size,
                max_num_patches=max_num_patches,
                min_num_patches=min_num_patches,
                pixel_shuffle_scale=pixel_shuffle_scale,
                temporal_patch_size=temporal_patch_size,
            )
            self._image_processor_key = key
        return self._image_processor


def _validate_native_inference_contract(
    metadata: IsaacMharmonyRenderMetadata, stats: IsaacNormalizationStats
) -> None:
    if int(metadata.n_obs_steps) not in (1, 3):
        raise ValueError(
            "native_mharmony supports checkpoint-owned n_obs_steps=1 or n_obs_steps=3; "
            f"got {metadata.n_obs_steps}."
        )
    if bool(metadata.include_scene_description):
        raise ValueError("native_mharmony supports include_scene_description=false only.")
    if int(metadata.rtc_prefix_length) != 0:
        raise ValueError("native_mharmony supports rtc_prefix_length=0 only.")
    if str(metadata.objective_form) != "Flow":
        raise ValueError(f"native_mharmony supports objective_form='Flow', got {metadata.objective_form!r}.")
    if str(metadata.action_representation) != "absolute":
        raise ValueError("native_mharmony supports absolute action representation only.")
    if int(metadata.action_horizon) != int(stats.action_horizon):
        raise ValueError(
            f"metadata action_horizon {metadata.action_horizon} != stats {stats.action_horizon}."
        )
    if int(metadata.action_dim) != int(stats.action_dim):
        raise ValueError(f"metadata action_dim {metadata.action_dim} != stats {stats.action_dim}.")
    if int(metadata.proprio_dim) != int(stats.proprio_dim):
        raise ValueError(f"metadata proprio_dim {metadata.proprio_dim} != stats {stats.proprio_dim}.")
    if metadata.training_proprio_contract is not None:
        _validate_training_proprio_contract(
            metadata.training_proprio_contract,
            dataset_name=metadata.dataset_name,
            proprio_dim=metadata.proprio_dim,
        )
    if len(metadata.camera_order) == 0:
        raise ValueError("native_mharmony requires at least one camera.")


def _metadata_fps(metadata: IsaacMharmonyRenderMetadata, stats: IsaacNormalizationStats) -> float:
    fps = metadata.target_fps if metadata.target_fps is not None else stats.target_fps
    return float(fps if fps is not None else DEFAULT_INFERENCE_FPS)


def _normalize_observation_history(
    observation_window: list[dict[str, Any]],
    metadata: IsaacMharmonyRenderMetadata,
) -> list[dict[str, Any]]:
    if not observation_window:
        raise ValueError("observation_window must contain at least one frame.")
    history = list(observation_window)[-int(metadata.n_obs_steps) :]
    if len(history) < int(metadata.n_obs_steps):
        history = [history[0]] * (int(metadata.n_obs_steps) - len(history)) + history

    normalized: list[dict[str, Any]] = []
    for idx, frame in enumerate(history):
        images = frame.get("images")
        if not isinstance(images, dict):
            raise ValueError(f"observation_window[{idx}] missing images dict.")
        missing = [camera for camera in metadata.camera_order if camera not in images]
        if missing:
            raise KeyError(f"observation_window[{idx}] missing cameras {missing}; have {list(images)}.")
        proprio = np.asarray(frame.get("proprio"), dtype=np.float32).reshape(-1)
        if proprio.shape != (int(metadata.proprio_dim),):
            raise ValueError(
                f"observation_window[{idx}] proprio must have shape ({metadata.proprio_dim},), got {proprio.shape}."
            )
        normalized.append(
            {"images": {camera: images[camera] for camera in metadata.camera_order}, "proprio": proprio}
        )
    return normalized


def _build_preamble(*, prompt: str, metadata: IsaacMharmonyRenderMetadata, fps: float) -> str:
    sections: list[str] = []
    if prompt:
        sections.append(str(prompt))
    config_text = _format_robotics_configuration_text(metadata, fps=fps)
    if config_text:
        sections.append(config_text)
    return "\n".join(sections)


def _format_robotics_configuration_text(metadata: IsaacMharmonyRenderMetadata, *, fps: float) -> str:
    lines = ["Configuration:"]
    if str(metadata.robot_type or "") != "generic":
        lines.append(f"robot_type: {metadata.robot_type}")
    if metadata.objective_form:
        lines.append(f"form: {metadata.objective_form}")
    lines.append(f"fps: {fps:.6g}")
    if metadata.action_representation == "relative":
        lines.append("action_representation: relative")
    if metadata.control_mode:
        lines.append(f"control_mode: {metadata.control_mode}")
    if metadata.action_layout:
        lines.append(f"action_layout: [{', '.join(str(x) for x in metadata.action_layout)}]")
    if metadata.camera_views:
        lines.append(f"camera_views: [{', '.join(str(x) for x in metadata.camera_views)}]")
    return "\n".join(lines)


def _normalize_time_qwen(timestamp: float, *, precision: int) -> float:
    return round(float(timestamp), int(precision))


def _format_timestamp_qwen(timestamp: float, *, precision: int) -> str:
    return f"<{timestamp:0.{int(precision)}f} seconds>"


def _timestamp_content_item(
    step_idx: int,
    timestamp_seconds: float,
    metadata: IsaacMharmonyRenderMetadata,
) -> dict[str, Any]:
    precision = int(metadata.timestamp_precision)
    normalized_seconds = _normalize_time_qwen(timestamp_seconds, precision=precision)
    return {
        "kind": "timestamp",
        "tokens": [],
        "text": _format_timestamp_qwen(normalized_seconds, precision=precision),
        "normalized_seconds": normalized_seconds,
        "precision": precision,
        "tags": {
            GENESIS_TEXT_TYPE_TAG: GENESIS_TEXT_TYPE_TIMESTAMP,
            "timestamp_text": _format_timestamp_qwen(normalized_seconds, precision=precision),
            "timestamp_seconds": float(timestamp_seconds),
            "timestamp_normalized_seconds": normalized_seconds,
        },
        "metadata": {
            "trajectory_timestamp": True,
            "trajectory_step": step_idx,
            "timestamp_seconds": float(timestamp_seconds),
            "timestamp_normalized_seconds": normalized_seconds,
            "timestamp_format": "qwen_seconds",
            "timestamp_precision": precision,
        },
    }


def _vector_content_item(
    proprio: np.ndarray,
    step_idx: int,
    timestamp_seconds: float,
    metadata: IsaacMharmonyRenderMetadata,
    stats: IsaacNormalizationStats,
) -> dict[str, Any]:
    normalized = normalize_isaac_proprio(np.asarray(proprio, dtype=np.float32).reshape(-1), stats.proprio)
    fallback_dims = _effective_bounds_fallback_dim_count(
        stats.proprio.q01, stats.proprio.q99, stats.proprio.minimum, stats.proprio.maximum
    )
    vector_metadata: dict[str, Any] = {
        "trajectory_step": step_idx,
        "trajectory_key": "state",
        "timestamp_seconds": float(timestamp_seconds),
        "source_state_keys": ["state"],
        "proprio_collapsed": False,
        "proprio_schema_source": "robotics_configurations.yml",
        "proprio_schema_dataset": metadata.dataset_name,
        "proprio_normalized": True,
        "proprio_normalization": "quantile_q01_q99",
        "proprio_stats_key": "proprio",
        "proprio_stats_dim": int(stats.proprio.dim),
        "proprio_normalized_dim": int(stats.proprio.dim),
        "proprio_normalization_eps": 1e-6,
        "proprio_effective_bounds_fallback_dims": int(fallback_dims),
    }
    contract = metadata.training_proprio_contract
    if contract is not None:
        components = contract["components"]
        source_slices = [list(component["range"]) for component in components]
        source_dims = [int(bounds[1] - bounds[0]) for bounds in source_slices]
        target_dim = int(contract["target_dim"])
        original_dim = int(np.asarray(proprio).reshape(-1).shape[0])
        vector_metadata.update(
            {
                "source_state_keys": [str(component["source"]) for component in components],
                "source_state_dims": source_dims,
                "source_state_slices": source_slices,
                "source_state_raw_layout": [dict(source) for source in contract["sources"]],
                "proprio_collapsed": len(components) > 1
                or any(bounds != [0, original_dim] for bounds in source_slices),
                "proprio_schema_source": contract["source"],
                "proprio_schema_dataset": contract["dataset_name"],
                "policy_state_components": [dict(component) for component in components],
                "policy_state_excluded": [dict(component) for component in contract["excluded"]],
                "policy_state_contract_version": int(contract["contract_version"]),
                "policy_state_contract_hash": contract["contract_hash"],
                "proprio_original_dim": original_dim,
                "proprio_target_dim": target_dim,
                "proprio_padding_applied": original_dim < target_dim,
                "proprio_truncated": original_dim > target_dim,
                "proprio_stats_key": "state",
            }
        )
    return {
        "kind": "vector",
        "values": normalized.astype(np.float32, copy=False).reshape(-1).tolist(),
        "shape": [int(normalized.reshape(-1).shape[0])],
        "dtype": "float32",
        "name": "state",
        "metadata": vector_metadata,
    }


def _effective_bounds_fallback_dim_count(
    q01: np.ndarray,
    q99: np.ndarray,
    minimum: np.ndarray | None,
    maximum: np.ndarray | None,
    *,
    eps: float = 1e-6,
) -> int:
    if minimum is None or maximum is None:
        return 0
    collapsed = (np.asarray(q99, dtype=np.float32) - np.asarray(q01, dtype=np.float32)) <= eps
    has_range = (np.asarray(maximum, dtype=np.float32) - np.asarray(minimum, dtype=np.float32)) > eps
    return int(np.count_nonzero(collapsed & has_range))


def _flow_action_tags(
    metadata: IsaacMharmonyRenderMetadata,
    stats: IsaacNormalizationStats,
    *,
    normalized_action_target: np.ndarray | None = None,
    action_is_pad: list[bool] | None = None,
    terminal: bool = False,
) -> dict[str, Any]:
    k = int(metadata.n_obs_steps)
    horizon = int(metadata.action_horizon)
    action_dim = int(metadata.action_dim)
    anchor_idx = k - 1
    emitted_action_indices = list(range(anchor_idx, anchor_idx + horizon))
    if normalized_action_target is None:
        action_target = normalize_isaac_actions(
            np.zeros((horizon, action_dim), dtype=np.float32),
            stats.action,
        ).astype(np.float32, copy=False)
    else:
        action_target = np.asarray(normalized_action_target, dtype=np.float32)
        if action_target.shape != (horizon, action_dim):
            raise ValueError(
                f"normalized_action_target must have shape {(horizon, action_dim)}, got {action_target.shape}."
            )
    if action_is_pad is None:
        action_is_pad = [False] * horizon
    return {
        "token_group": FLOW_ACTION_TOKEN_GROUP_NAME,
        GENESIS_TEXT_TYPE_TAG: GENESIS_TEXT_TYPE_ACTION_C,
        "trajectory_action": anchor_idx,
        "action_index": anchor_idx,
        "obs_window": [0, anchor_idx],
        "action_window": [anchor_idx, emitted_action_indices[-1]],
        # Genesis drops message_stop_token from terminal chunks, which is what makes their
        # footer <|end|> rather than <|return|>.
        "terminal": bool(terminal),
        "message_stop_token": None if terminal else metadata.message_stop_token,
        "observation_delta_indices": list(range(-(k - 1), 1)),
        "observation_is_pad": [False] * k,
        "requested_action_delta_indices": list(range(horizon)),
        "action_delta_indices": list(range(horizon)),
        "action_is_pad": list(action_is_pad),
        "action_indices": emitted_action_indices,
        "objective_form": metadata.objective_form,
        "action_target": action_target.tolist(),
        "action_horizon": horizon,
        "action_dim": action_dim,
        "stats_key": "action",
        "normalized": True,
        "flow_matching_target": True,
    }


def _fast_action_tags(flow_tags: dict[str, Any]) -> dict[str, Any]:
    tags = dict(flow_tags)
    tags.update(
        {
            "token_group": FAST_ACTION_TOKEN_GROUP_NAME,
            "tokenizer": FAST_ACTION_TOKENIZER_NAME,
            GENESIS_TEXT_TYPE_TAG: GENESIS_TEXT_TYPE_ACTION,
            "objective_form": "FAST",
        }
    )
    tags.pop("action_target", None)
    tags.pop("flow_matching_target", None)
    return tags


_GENESIS_TRAINING_ACTION_TAG_KEYS = (
    "token_group",
    GENESIS_TEXT_TYPE_TAG,
    "action_index",
    "trajectory_action",
    "objective_form",
    "obs_window",
    "action_window",
    "action_indices",
    "action_delta_indices",
    "action_is_pad",
    "action_target",
    "action_horizon",
    "action_dim",
    "normalized",
)


def _project_genesis_training_action_tags(tags: dict[str, Any]) -> dict[str, Any]:
    """Match Genesis conversation._collect_token_tags at the final TensorStream boundary."""
    return {key: tags[key] for key in _GENESIS_TRAINING_ACTION_TAG_KEYS if key in tags}


def _plan_item_to_mh_content(
    types,
    item: dict[str, Any],
    *,
    encoding,
    metadata: IsaacMharmonyRenderMetadata,
):
    kind = str(item.get("kind") or "text")
    if kind == "text":
        return types.MHText(str(item.get("text") or ""))
    if kind == "timestamp":
        normalized_seconds = float(item["normalized_seconds"])
        precision = int(item.get("precision", metadata.timestamp_precision))
        tokens = encoding.encode_timestamps_qwen([normalized_seconds], precision=precision)[0]
        return types.MHTokens(tokens=[int(token) for token in tokens], tags=item.get("tags"))
    if kind == "tokens":
        return types.MHTokens(tokens=[int(token) for token in item.get("tokens", [])], tags=item.get("tags"))
    if kind == "image":
        media_ref = types.MediaRef(
            id=str(item["media_id"]),
            mime="image/png",
            bytes_b64=_image_to_png_b64(item["image"]),
            metadata=item.get("metadata"),
        )
        return types.MHImage(media_ref=media_ref, metadata=item.get("metadata"))
    if kind == "vector":
        values = np.asarray(item.get("values"), dtype=np.float32).reshape(-1).tolist()
        return types.MHVector(
            values=values,
            shape=[int(x) for x in item.get("shape", [len(values)])],
            dtype=str(item.get("dtype") or "float32"),
            metadata=item.get("metadata"),
            name=item.get("name"),
        )
    raise ValueError(f"Unsupported Isaac mharmony content kind {kind!r}.")


def _image_to_png_b64(image: Any) -> str:
    from PIL import Image as PILImage

    if isinstance(image, bytes | bytearray):
        with PILImage.open(io.BytesIO(bytes(image))) as pil:
            rgb = pil.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
    arr = coerce_rgb_uint8(
        image,
        contiguous=True,
        shape_error="Unsupported Isaac image shape for native mharmony: {shape}.",
    )
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _maybe_swap_final_assistant_footer(
    rendered: dict[str, Any],
    *,
    harmony_conv,
    stop_token: str | None,
    encoding,
) -> dict[str, Any]:
    if not stop_token:
        return rendered
    messages = getattr(harmony_conv, "messages", None)
    if not messages:
        return rendered
    last_message = messages[-1]
    last_role = getattr(getattr(last_message, "author", None), "role", None)
    if getattr(last_role, "value", last_role) != "assistant":
        return rendered
    if getattr(last_message, "channel", None) != "final":
        return rendered

    events = rendered.get("stream", {}).get("events", [])
    if not isinstance(events, list):
        return rendered
    end_token = encoding.encode("<|end|>", allowed_special={"<|end|>"})[0]
    replacement = encoding.encode(f"<|{stop_token}|>", allowed_special={f"<|{stop_token}|>"})[0]
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        if event.get("role") != "assistant" or event.get("channel") != "final":
            continue
        if event.get("modality") not in {"special_token", "role_marker"}:
            continue
        data = event.get("data")
        if isinstance(data, dict) and data.get("Tokens") == [end_token]:
            data["Tokens"] = [replacement]
            break
    return rendered
