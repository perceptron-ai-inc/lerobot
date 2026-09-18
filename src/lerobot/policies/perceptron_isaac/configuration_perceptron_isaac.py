"""Configuration for the native Perceptron ISAAC VLA policy.

Processors own mharmony packing, FAST target construction, stats
normalization, and action unnormalization. The policy owns inference state and
the joint text/FAST/flow training loss. Genesis bridge rendering and loss paths
are intentionally unsupported.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from ..processor_utils import flatten_feature_names
from ..utils import ensure_vla_feature_contract
from .checkpoint_integrity import is_lowercase_sha256
from .hardware_features import so100_hardware_feature_aliases
from .isaac_stats import require_closed_loop_safe_isaac_profile
from .mharmony_contract import SUPPORTED_MHARMONY_VERSION, normalize_mharmony_version_marker

# Checkpoint-local stats sidecar written next to config.json at save time. Relative on
# purpose: PEFT loaders resolve it against the adapter checkpoint root first (see
# PerceptronIsaacPolicy.resolve_checkpoint_config_paths), keeping the package
# self-contained and relocatable.
NATIVE_STATS_EXPORT_FILENAME = "isaac_stats.json"

# The inference recipe a Genesis export ships at its root. Its rendering block is the
# checkpoint's own declaration of the mHarmony reserved-token layout it was trained with.
NATIVE_RECIPE_EXPORT_FILENAME = "policy_inference_recipe.json"


def is_portable_isaac05_repository(model_dir: Path) -> bool:
    """True when ``model_dir`` is the root of a raw Isaac-0.5 export.

    Such an export nests the LeRobot policy package in ``lerobot_policy/`` and keeps
    its documented assets (``fast_processor_pinned/``, ``isaac_stats.json``,
    ``policy_normalization.json``) at this root, so anything resolving paths for a
    package loaded from ``<root>/lerobot_policy`` has to recognise the root.
    """
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and config.get("model_type") == "isaac_0_5"


# Per-suite routing tables, copied next to config.json at save time for the same reason and
# resolved by the same hook. Training-time inputs rather than serving sidecars, but a config
# that ships absolute training-host paths is stale the moment the package moves.
SUITE_STATS_EXPORT_FILENAME = "isaac_suite_stats.json"
SUITE_BY_TASK_INDEX_EXPORT_FILENAME = "isaac_suite_by_task_index.json"


@PreTrainedConfig.register_subclass("perceptron_isaac")
@dataclass
class PerceptronIsaacConfig(PreTrainedConfig):
    """Config for the Perceptron Isaac flow-matching VLA policy.

    The policy shell remains expert-agnostic. Checkpoint identity selects either
    the existing Qwen3.5 VLA or the MK1 Qwen3.6 null-MoE backbone; both attach
    the vector encoder and MolmoAct action expert behind the same policy API.
    """

    # --- observation / action geometry ---
    # Released ISAAC checkpoints use either one observation (post3279) or a
    # three-frame history (nobs3 experiments). The importer writes the exact
    # checkpoint-owned value into the packaged policy config.
    n_obs_steps: int = 3
    chunk_size: int = 30
    n_action_steps: int = 30
    action_dim: int = 7  # real LIBERO action width; model output is padded to max_action_dim
    max_action_dim: int = 64
    max_action_horizon: int = 64
    proprio_dim: int = 8  # real LIBERO proprio width
    max_state_dim: int = 128

    # --- expert / inference ---
    # Native eval/serving backend. No Genesis bridge backend is supported.
    inference_backend: str = "native_mharmony"
    action_expert_type: str = "molmoact"
    num_inference_steps: int = 10
    # Flow-sample averaging: sample the flow ODE `num_flow_samples` times (fresh noise each) and
    # mean the normalized chunks before unnormalize. 1 = single sample (no averaging).
    num_flow_samples: int = 4
    flow_seed_base: int | None = None
    # Flow-target clip. This is the checkpoint-owned value: the inference recipe carries it on
    # the ``flow_action`` wire, and the importer copies it from there.
    clip_normalized_max: float = 10.0
    # FAST-target clip, applied only to the tokenizer input. Genesis lowers the two objectives
    # through separate transforms with independent caps, so a single shared cap trains FAST on
    # a different action distribution than Genesis does.
    #
    # This default is NOT universal -- it must match the base recipe's
    # ``dataset.transforms.strategies.overrides.action.clip_normalized_max``, which each
    # pretraining run sets independently of the ``fast_tokenizer`` library default of 1.0:
    #   LIBERO/Genesis lineage (hf-genesis-ft10, hf-genesis-ft40k):        FAST 1.0,  Flow 10.0
    #   bi-YAM lineage (hf-midtrain-step80000, hf-r4-step-200000):         FAST 10.0, Flow 10.0
    # Check the base's config.toml before finetuning, and set this deliberately rather than
    # inheriting the default. The caps are not interchangeable: q01/q99 normalization puts real
    # values well outside [-1, 1] (up to 3.68 on bi-yam-stack-cups, on all 14 dims), so 1.0
    # against a bi-YAM base saturates targets the base saw unclipped. That is a legitimate
    # choice -- it is what MolmoAct2 does on every objective -- but it changes the target
    # distribution the base was pretrained on, so make it on purpose.
    fast_clip_normalized_max: float = 1.0
    # LIBERO eval clips OSC pose deltas (all dims but the last/gripper) to [-1,1] before
    # the sim consumes them. Set False only for ablations.
    clip_action_pose: bool = True
    # Settle: emit `num_settle_steps` idle actions (hold pose, gripper=`settle_gripper`) at the
    # start of each episode before the policy controls, so the sim stabilizes from the init state.
    # 0 disables.
    #
    # This is NOT the right lever for matching Genesis's LIBERO settling. Genesis waits
    # NUM_STEPS_WAIT=50 inside its adapter's reset, stepping the simulator directly, so that
    # wait is free and the policy still gets the full 280-step episode budget. These idle
    # actions instead come out of `select_action` and are consumed from that same budget.
    # LeRobot's LiberoEnv has the equivalent free wait (`num_steps_wait`, applied in reset);
    # set that to match a reference, and leave this at 0 for LIBERO.
    # Settle frames never advance the episode clock -- see PerceptronIsaacPolicy.select_action.
    num_settle_steps: int = 0
    settle_gripper: float = -1.0
    # RTC (real-time chunking): NOT SUPPORTED -- __post_init__ rejects any non-zero value, so
    # this must stay 0. Two independent gates keep it off: (1) the vendored MolmoAct expert
    # (pre-genesis-#3402) collapses per-token tau in its SinusoidalTimeEmbedding
    # (`timesteps.view(B,-1)[:,0]`), so per-row prefix timesteps are ignored -- genesis HEAD's
    # DiT expert has since fixed this, so that argument is vendored-head-specific; and (2) the
    # checkpoint contract declares its trained RTC budget (`rtc_max_delay_steps`, 0 for every
    # deployed package), which MolmoActExpertHead.sample enforces against any requested
    # action prefix. Reproducing the Genesis flow-eval regime (exec horizon 12 with a 4-row
    # pinned prefix) would require an RTC-trained checkpoint plus enabling the prefix branch
    # in select_action.
    rtc_prefix_length: int = 0

    # --- vision / cameras ---
    image_size: tuple[int, int] = (256, 256)
    # Geometry applied before the ISAAC renderer. Imported and pre-existing packages default
    # to ``stretch`` for exact backward compatibility; new SO100 fine-tunes set ``letterbox``
    # and deployment reads this saved field to reproduce training preprocessing.
    image_preprocessing: str = "stretch"
    camera_order: tuple[str, ...] = ("image", "wrist_image")  # agentview, wrist
    # Physical camera role -> checkpoint-internal visual slot. Imported SO100
    # checkpoints historically call their slots ``top``/``side`` even when a
    # local fine-tune supplies the fixed side camera first and wrist camera
    # second. Saving this mapping prevents those slot labels from being
    # misread as mount semantics at deployment time.
    serving_camera_roles: dict[str, str] | None = None
    allow_image_key_fallback: bool = True  # compatibility only; authenticated imports set false
    # False passes gripper dimensions through in physical units instead of q01/q99-normalizing
    # them, matching MolmoAct2's `normalize_gripper: false` and its saved per-dim stat mask.
    # Genesis normalizes every dimension, so True is what the ISAAC bases were pretrained with;
    # setting False changes the representation the base saw and is an opt-in experiment.
    # The gripper dimensions are found by name from action_feature_names/state_feature_names,
    # as MolmoAct2's _default_feature_mask does, and the resulting mask is serialized into the
    # checkpoint stats so eval inherits it rather than re-deriving it from a serving config.
    normalize_gripper: bool = True
    # Remap a {0,1} gripper (1=open, 0=close; e.g. IPEC no-noops) to the robosuite/env [-1,+1] convention
    # (+1=close, -1=open) at serving: g_env = 1 - 2*g. Off for datasets already using signed gripper.
    gripper_binary_to_signed: bool = False

    # --- joint calibration convention adapter (old-convention datasets on current-LeRobot arms) ---
    # SO100/SO101 checkpoints trained on pre-LeRobot-0.5.0 community data expect the OLD joint
    # convention, while a fresh `lerobot-calibrate` produces the CURRENT convention. The Isaac stats
    # and weights are old-convention, and this policy does NO implicit remap, so on a current-LeRobot
    # arm the proprio would be out-of-distribution and absolute targets sent in the wrong frame.
    # This bidirectional adapter reconciles them as explicit processor steps (do NOT "calibrate into
    # the old convention" — calibrate normally and transform here):
    #   state  (obs, before render/normalize):    q_model = signs * q_lerobot + offsets
    #   action (training, before normalize):      q_model = signs * q_lerobot + offsets
    #   action (after unnormalize, before robot): q_lerobot = signs * (q_model - offsets)
    # For SO100/SO101 use signs=[1,-1,1,1,1,1], offsets=[0,90,90,0,0,0] (same as the sibling LeRobot
    # MolmoAct2-SO100_101 checkpoint; see docs/source/backwardcomp.mdx). Both None => no-op (sim/LIBERO
    # or already-current-convention data).
    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None
    # Version 2 closes the training-side half of the joint-frame contract: action
    # supervision is converted into the same model frame as proprio before ISAAC
    # quantile normalization. Raw imported bases predate this field but acquire the
    # current default when loaded; saved fine-tunes persist it so unsafe v1 resumes
    # can be rejected without guessing from directory names.
    training_action_frame_contract_version: int = 2

    # --- model / checkpoint paths (filled by the user / launcher) ---
    hf_model_path: str | None = None  # checkpoint-local Qwen3.5 or MK1 composite HF directory
    # Zero-step packages exercise checkpoint/model plumbing but are never deployable.
    artifact_kind: str = "trained_policy"
    trained_steps: int = 1
    # Apply the converted Qwen3.5 checkpoint RMSNorm correction at load. MK1 is already
    # zero-centered and its importer writes false into the packaged policy config.
    apply_offset_norm: bool = True
    # Deprecated compatibility fields accepted by older configs. They are not
    # used by the native eval/serving path.
    config_toml_path: str | None = None
    stats_path: str | None = None
    per_suite_stats_dir: str | None = None
    include_scene_description: bool = False  # scene-free serving by default (reproducible)
    vector_max_states: int = 128  # proprio is padded to this width before the encoder
    normalize_task_text: bool = True
    action_conditioning: bool = False
    action_conditioning_role: str = "user"
    mistake_conditioning: bool = False

    # LeRobot-native mharmony processor geometry. Eval and direct serving share
    # the same processor-owned packing and the policy consumes pre-rendered streams.
    render_processor_enabled: bool = False
    render_patch_size: int = 16
    render_pixel_shuffle_scale: int = 2
    render_temporal_patch_size: int = 2
    render_max_num_patches: int | None = None  # None => metadata/default max patch budget
    render_min_num_patches: int | None = None

    # Native serving metadata used by the native_mharmony eval processors.
    dataset_name: str = "libero"
    policy_state_dataset: str | None = None
    robot_type: str = "generic"
    control_mode: str = "joint"
    target_fps: float | None = None
    # Importer-owned deployment geometry. Generic rollout consumes these
    # fields without branching on policy type and fails before connect when
    # the robot cannot satisfy an authenticated package contract.
    action_feature_names: list[str] | None = None
    state_feature_names: list[str] | None = None
    strict_hardware_feature_contract: bool = False
    strict_environment_feature_contract: bool = False
    normalization_profile_id: str | None = None
    normalization_profile_scope: str | None = None
    normalization_validation_status: str | None = None
    deployment_adapter_sha256: str | None = None
    qwen35_trained_package_manifest_sha256: str | None = None
    mk1_model_import_sha256: str | None = None
    mk1_source_model_import_sha256: str | None = None
    mk1_trained_package_manifest_sha256: str | None = None
    native_render_metadata_path: str | None = None
    native_stats_path: str | None = None
    # Per-suite training normalization. Multi-suite LIBERO training (all four suites in one
    # dataset, as Genesis's cloud/libero:1.0 does) must normalize each sample by its own suite:
    # end-effector height spans [0.916, 1.286] in libero_spatial but [0.010, 0.337] in
    # libero_object, so cross-normalizing pushes proprio far outside [-1, 1]. Set both together;
    # inference is single-scope by contract and leaves them unset.
    suite_stats_path: str | None = None
    suite_by_task_index_path: str | None = None
    # Checkpoint-local, immutable copy of physical-intelligence/fast. Importing
    # a checkpoint records the complete tree digest before remote code is used.
    fast_processor_path: str | None = None
    fast_processor_tree_sha256: str | None = None
    # Exact standalone runtime covered by the ISAAC render-parity contract.
    mharmony_version: str = SUPPORTED_MHARMONY_VERSION
    # True = native eval contract: preprocessor owns mharmony packing, policy returns
    # normalized actions, postprocessor owns unnormalization/clipping.
    native_require_processor_stream: bool = True

    # --- LeRobot-native finetuning ---
    objective: str = "mixed"
    loss_plan: str = "text_ntp_fast_flow_action"
    train_expert_only: bool = False  # True freezes the VLM + vision, trains only model.action_expert.*
    flow_matching_detach_vlm_activations: bool = False  # True = knowledge-insulation (no VLM grad from flow)
    exclude_non_agent_roles: bool = True
    train_samples_per_chunk: int = 8
    train_clip_normalized_actions: bool = True
    train_skip_outlier_threshold: float = 20.0
    train_max_sequence_length: int = 4096
    flow_rtc_max_delay_steps: int = 0
    flow_rtc_delay_sampling: str = "uniform"
    flow_dual_timestep_ratio: float = 0.0
    flow_mask_padded_action_rows: bool = True  # MolmoAct expert masks padded terminal action rows
    softmax_auxiliary_loss_scale: float = 0.0001
    freeze_input_embeddings: bool = True
    # Accelerator reads this field and opens the matching autocast context. ISAAC training keeps
    # numerically sensitive loss reductions in FP32 internally, independently of this compute dtype.
    dtype: str = "bfloat16"
    # Store every *trainable* parameter and its Adam moments in FP32 while computing under BF16
    # autocast. In expert-only/LoRA training the frozen VLM stays BF16; in dense training every
    # parameter is trainable and therefore remains FP32. Optimizing the expert directly in BF16
    # rounds many AdamW updates at the recipe's learning rates.
    train_storage_fp32: bool = True

    # Native Isaac processors own normalization, so bypass LeRobot's stock normalizer.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # --- optimizer / scheduler presets ---
    optimizer_lr: float = 1e-5
    optimizer_vit_lr: float = 5e-6
    optimizer_action_expert_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    optimizer_warmup_steps: int = 200
    max_train_steps: int = 30000

    def __post_init__(self):
        super().__post_init__()
        self.mharmony_version = normalize_mharmony_version_marker(self.mharmony_version)
        if self.n_action_steps <= 0:
            raise ValueError("Perceptron Isaac n_action_steps must be positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
        if self.max_action_dim < 1 or self.action_dim < 1 or self.action_dim > self.max_action_dim:
            raise ValueError(
                "Perceptron Isaac action_dim must be positive and no greater than max_action_dim; "
                f"got action_dim={self.action_dim}, max_action_dim={self.max_action_dim}."
            )
        if self.proprio_dim < 1 or self.proprio_dim > self.vector_max_states:
            raise ValueError(
                "Perceptron Isaac proprio_dim must be positive and no greater than vector_max_states; "
                f"got proprio_dim={self.proprio_dim}, vector_max_states={self.vector_max_states}."
            )
        if self.max_action_horizon < 1 or self.chunk_size < 1 or self.chunk_size > self.max_action_horizon:
            raise ValueError(
                "Perceptron Isaac chunk_size must be positive and no greater than "
                f"max_action_horizon; got chunk_size={self.chunk_size}, "
                f"max_action_horizon={self.max_action_horizon}."
            )
        allowed_backends = {"native_mharmony"}
        if self.inference_backend not in allowed_backends:
            raise ValueError(
                f"Unsupported Perceptron Isaac inference_backend={self.inference_backend!r}; "
                f"expected one of {sorted(allowed_backends)}."
            )
        if self.artifact_kind not in {"trained_policy", "neutral_debug"}:
            raise ValueError(
                "Perceptron Isaac artifact_kind must be 'trained_policy' or 'neutral_debug', "
                f"got {self.artifact_kind!r}."
            )
        if not isinstance(self.trained_steps, int) or isinstance(self.trained_steps, bool):
            raise ValueError("Perceptron Isaac trained_steps must be an integer.")
        if self.artifact_kind == "neutral_debug" and self.trained_steps != 0:
            raise ValueError("A neutral_debug Perceptron Isaac package must declare trained_steps=0.")
        if self.artifact_kind == "trained_policy" and self.trained_steps <= 0:
            raise ValueError("A trained_policy Perceptron Isaac package must declare positive trained_steps.")
        # Both names resolve to the same vendored expert head: 'dit' is genesis's
        # post-#3402 name for it (see build_action_expert_head's alias).
        if self.action_expert_type not in {"molmoact", "dit"}:
            raise ValueError(
                "Perceptron Isaac action_expert_type must be 'molmoact' or 'dit'; "
                f"got {self.action_expert_type!r}."
            )
        if self.n_obs_steps not in (1, 3):
            raise ValueError(
                "Perceptron Isaac checkpoints support n_obs_steps=1 or n_obs_steps=3; "
                f"got {self.n_obs_steps}."
            )
        if len(self.image_size) != 2 or any(int(size) <= 0 for size in self.image_size):
            raise ValueError(
                "Perceptron Isaac image_size must contain positive (height, width) values; "
                f"got {self.image_size}."
            )
        if self.image_preprocessing not in {"stretch", "letterbox"}:
            raise ValueError(
                "Perceptron Isaac image_preprocessing must be 'stretch' or 'letterbox'; "
                f"got {self.image_preprocessing!r}."
            )
        if self.serving_camera_roles is not None:
            expected_roles = {"side", "wrist"}
            if set(self.serving_camera_roles) != expected_roles:
                raise ValueError(
                    "Perceptron Isaac serving_camera_roles must define exactly side and wrist; "
                    f"got {sorted(self.serving_camera_roles)}."
                )
            slots = list(self.serving_camera_roles.values())
            if len(set(slots)) != 2 or not set(slots).issubset(set(self.camera_order)):
                raise ValueError(
                    "Perceptron Isaac serving_camera_roles must map side/wrist to two distinct "
                    f"camera_order slots; got {self.serving_camera_roles} for {self.camera_order}."
                )
        if self.include_scene_description:
            raise ValueError(
                "Perceptron Isaac no-scene LIBERO checkpoints require include_scene_description=false."
            )
        if self.action_conditioning_role != "user":
            raise ValueError(
                "Native Perceptron Isaac action conditioning supports only action_conditioning_role='user'."
            )
        if (self.joint_signs is None) != (self.joint_offsets is None):
            raise ValueError("joint_signs and joint_offsets must both be set or both be None.")
        if self.joint_signs is not None and len(self.joint_signs) != len(self.joint_offsets):
            raise ValueError("joint_signs and joint_offsets must have the same length.")
        if self.joint_signs is not None:
            if len(self.joint_signs) > min(self.action_dim, self.proprio_dim):
                raise ValueError("Joint-frame transform exceeds action/state dimensions.")
            if any(float(sign) not in {-1.0, 1.0} for sign in self.joint_signs):
                raise ValueError("joint_signs must contain only +1 or -1 for an exact inverse.")
            if any(not math.isfinite(float(offset)) for offset in self.joint_offsets):
                raise ValueError("joint_offsets must be finite.")
        if self.training_action_frame_contract_version != 2:
            raise ValueError(
                "Perceptron Isaac supports only training_action_frame_contract_version=2; "
                f"got {self.training_action_frame_contract_version}."
            )
        # Older bridge checkpoints may still contain these fields. Do not use or
        # re-serialize the deprecated TOML/per-suite knobs in the native eval path.
        if self.native_stats_path is None and self.stats_path:
            self.native_stats_path = self.stats_path
        self.config_toml_path = None
        self.per_suite_stats_dir = None
        if self.rtc_prefix_length != 0:
            raise ValueError("Perceptron Isaac MolmoAct no-scene LIBERO eval requires rtc_prefix_length=0.")
        if self.train_samples_per_chunk < 1:
            raise ValueError("train_samples_per_chunk must be >= 1.")
        if self.train_skip_outlier_threshold < 0.0:
            raise ValueError("train_skip_outlier_threshold must be nonnegative.")
        if self.train_max_sequence_length < 1:
            raise ValueError("train_max_sequence_length must be >= 1.")
        if (self.fast_processor_path is None) != (self.fast_processor_tree_sha256 is None):
            raise ValueError(
                "fast_processor_path and fast_processor_tree_sha256 must both be set or both be None."
            )
        if self.fast_processor_tree_sha256 is not None and not is_lowercase_sha256(
            self.fast_processor_tree_sha256
        ):
            raise ValueError("fast_processor_tree_sha256 must be a lowercase SHA-256 digest.")
        if self.flow_rtc_max_delay_steps != 0:
            raise ValueError(
                "Native ISAAC training does not yet implement RTC prefix sampling; "
                "flow_rtc_max_delay_steps must be 0."
            )
        if self.flow_dual_timestep_ratio != 0.0:
            raise ValueError(
                "Native ISAAC training does not yet implement dual-timestep sampling; "
                "flow_dual_timestep_ratio must be 0."
            )
        if self.softmax_auxiliary_loss_scale < 0.0:
            raise ValueError("softmax_auxiliary_loss_scale must be nonnegative.")
        if self.dtype != "bfloat16":
            raise ValueError(
                "Perceptron Isaac training and inference require dtype='bfloat16' compute; "
                f"got {self.dtype!r}."
            )
        if not str(self.robot_type).strip():
            raise ValueError("Perceptron Isaac native metadata requires a non-empty robot_type.")
        if not str(self.control_mode).strip():
            raise ValueError("Perceptron Isaac native metadata requires a non-empty control_mode.")
        if self.policy_state_dataset is not None and not str(self.policy_state_dataset).strip():
            raise ValueError("policy_state_dataset must be a non-empty string when set.")
        for field_name in (
            "deployment_adapter_sha256",
            "qwen35_trained_package_manifest_sha256",
            "mk1_model_import_sha256",
            "mk1_source_model_import_sha256",
            "mk1_trained_package_manifest_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None and not is_lowercase_sha256(value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
        if self.target_fps is not None and float(self.target_fps) <= 0.0:
            raise ValueError("Perceptron Isaac target_fps must be positive when set.")
        for label, names, width in (
            ("action_feature_names", self.action_feature_names, self.action_dim),
            ("state_feature_names", self.state_feature_names, self.proprio_dim),
        ):
            if names is not None and (
                len(names) != width
                or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name.strip() for name in names)
            ):
                raise ValueError(f"{label} must contain {width} unique, non-empty names.")
        if self.strict_hardware_feature_contract and (
            self.action_feature_names is None or self.state_feature_names is None
        ):
            raise ValueError(
                "strict_hardware_feature_contract requires action_feature_names and state_feature_names."
            )
        self._validate_declared_feature_widths()
        require_closed_loop_safe_isaac_profile(
            profile_id=self.normalization_profile_id,
            profile_scope=self.normalization_profile_scope,
            validation_status=self.normalization_validation_status,
        )
        if self.strict_hardware_feature_contract or self.strict_environment_feature_contract:
            self._validate_known_deployment_contract()

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        """Validate strict hardware joint order before positional values or stats are consumed."""
        if not self.strict_hardware_feature_contract:
            return
        aliases = so100_hardware_feature_aliases() if self.robot_type.lower() == "so100_so101" else {}
        for key, expected in ((ACTION, self.action_feature_names), (OBS_STATE, self.state_feature_names)):
            assert expected is not None, "Strict hardware configs must declare state and action names."
            names = flatten_feature_names(features.get(key, {}).get("names"), empty_as_none=True)
            canonical_names = [aliases.get(name, name) for name in names] if names is not None else None
            if canonical_names != expected:
                raise ValueError(
                    f"ISAAC dataset feature order for {key} must match the package hardware layout: "
                    f"expected={expected}, got={names}. Only reviewed semantic aliases are accepted; "
                    "different layouts require an explicit dataset transform."
                )

    def checkpoint_asset_roots(self) -> tuple[Path, ...]:
        """Directories that can hold this policy's checkpoint-local sidecars.

        ``pretrained_path`` is the policy package, which for a raw Isaac-0.5 export is
        ``<root>/lerobot_policy`` while the documented sidecars sit at ``<root>``.
        """
        if self.pretrained_path is None:
            return ()
        package_root = Path(self.pretrained_path)
        if not package_root.is_dir():
            return ()
        export_root = package_root.parent
        if is_portable_isaac05_repository(export_root):
            return (package_root, export_root)
        return (package_root,)

    def resolve_native_stats_path(self) -> str | None:
        """Return the normalization stats path this config can actually load.

        The declared fields win, so an importer-written package keeps pointing at its
        own sidecar. A raw export declares neither, so fall back to the conventional
        ``isaac_stats.json`` beside the package it was loaded from: the same filename
        ``_save_pretrained`` writes, carrying the same quantiles the packaged
        processor state file holds.
        """
        declared = self.native_stats_path or self.stats_path
        if declared:
            return str(declared)
        for root in self.checkpoint_asset_roots():
            candidate = root / NATIVE_STATS_EXPORT_FILENAME
            if candidate.is_file():
                return str(candidate)
        return None

    def resolve_native_recipe_path(self) -> str | None:
        """Return the inference recipe shipped beside the package this config was loaded from.

        A raw export keeps ``policy_inference_recipe.json`` at its root, next to the
        ``lerobot_policy/`` package, exactly like ``isaac_stats.json``. An importer-written
        package ships ``native_render_metadata.json`` instead, and a config with no
        ``pretrained_path`` has no checkpoint to read, so both return ``None`` here.
        """
        for root in self.checkpoint_asset_roots():
            candidate = root / NATIVE_RECIPE_EXPORT_FILENAME
            if candidate.is_file():
                return str(candidate)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write config.json, exporting checkpoint-local sidecars when known.

        Two independent exports, either of which may fire alone: the normalization stats
        below, and the per-suite routing tables in ``_export_suite_tables``. Together they
        are what makes every path-typed field in a saved config checkpoint-local.

        The ISAAC processor factories record the live pipeline stats on this config
        (``_export_native_stats``, a plain attribute -- draccus serializes dataclass
        fields only). When present, the stats are written to a checkpoint-local
        ``isaac_stats.json`` (masks embedded) and the saved config points at it with
        a RELATIVE path, so a finetune package never references the base package's
        stats file. Without this, every finetune config.json inherits the base's
        absolute ``native_stats_path`` and serving-side path-loads normalize proprio
        with the wrong quantiles.
        """
        overrides: dict[str, object] = {}
        for name in (
            "apply_offset_norm",
            "mk1_model_import_sha256",
            "mk1_source_model_import_sha256",
        ):
            override = getattr(self, f"_export_{name}", None)
            if override is not None or hasattr(self, f"_export_{name}"):
                overrides[name] = override
        for name, record in getattr(self, "_checkpoint_relative_paths", {}).items():
            if (
                isinstance(record, dict)
                and str(getattr(self, name, "")) == record.get("absolute")
                and isinstance(record.get("relative"), str)
            ):
                overrides[name] = record["relative"]

        stats = getattr(self, "_export_native_stats", None)
        if stats is not None:
            from .isaac_stats import save_isaac_stats

            save_isaac_stats(stats, Path(save_directory) / NATIVE_STATS_EXPORT_FILENAME)
            overrides["native_stats_path"] = NATIVE_STATS_EXPORT_FILENAME
            overrides["stats_path"] = None
            # Labels follow the exported stats so the saved config stops advertising the
            # base lineage while carrying finetune-dataset values.
            overrides["normalization_profile_id"] = stats.profile_id
            overrides["normalization_profile_scope"] = stats.profile_scope
            overrides["normalization_validation_status"] = stats.validation_status

        overrides.update(self._export_suite_tables(Path(save_directory)))

        if not overrides:
            super()._save_pretrained(save_directory)
            return

        original = {name: getattr(self, name) for name in overrides}
        for name, value in overrides.items():
            setattr(self, name, value)
        try:
            super()._save_pretrained(save_directory)
        finally:
            for name, value in original.items():
                setattr(self, name, value)

    def _export_suite_tables(self, save_directory: Path) -> dict[str, str | None]:
        """Copy the per-suite routing tables into the checkpoint, returning relative overrides.

        ``suite_stats_path`` / ``suite_by_task_index_path`` arrive as absolute
        training-host paths (``--policy.suite_stats_path=...``), so without this every
        multi-suite finetune's config.json ships two paths that are stale the moment the
        package leaves the machine that trained it. Copying them checkpoint-local and
        writing RELATIVE values keeps the package self-contained, exactly as the stats
        export does, and ``_resolve_checkpoint_local_paths`` resolves both back at load.

        Never fatal: a save that raises would lose a training checkpoint over a routing
        table that only the training render path reads. A source that has gone missing
        is left declared as-is and still fails loudly at the first render.
        """
        import logging
        import shutil

        pairs = (
            ("suite_stats_path", SUITE_STATS_EXPORT_FILENAME),
            ("suite_by_task_index_path", SUITE_BY_TASK_INDEX_EXPORT_FILENAME),
        )
        # Both or neither: the processor rejects a lone path, so a half-copied pair would
        # turn a recoverable stale path into an outright invalid config.
        declared = {name: getattr(self, name, None) for name, _ in pairs}
        if not all(str(value or "").strip() for value in declared.values()):
            return {}
        sources = {name: Path(str(value)) for name, value in declared.items()}
        missing = [str(path) for path in sources.values() if not path.is_file()]
        if missing:
            logging.getLogger(__name__).warning(
                "Perceptron Isaac per-suite table(s) %s are not readable at save time; the saved "
                "config keeps the declared path(s). Serving is unaffected (per-suite routing is "
                "training-only), but a resume elsewhere will fail on them.",
                missing,
            )
            return {}

        exported: dict[str, str | None] = {}
        for name, filename in pairs:
            destination = save_directory / filename
            source = sources[name]
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
            exported[name] = filename
        return exported

    def _validate_known_deployment_contract(self) -> None:
        robot_type = self.robot_type.lower()
        if robot_type == "bi_yam":
            expected_layout = [
                *(f"left_joint_{index}.pos" for index in range(6)),
                "left_gripper.pos",
                *(f"right_joint_{index}.pos" for index in range(6)),
                "right_gripper.pos",
            ]
            expected = {
                "dataset_name": "molmoact2_bimanualyam",
                "control_mode": "joint",
                "action_dim": 14,
                "proprio_dim": 14,
                "chunk_size": 30,
                "n_action_steps": 30,
                "num_inference_steps": 10,
                "num_flow_samples": 1,
                "image_size": (360, 640),
                "camera_order": ("top", "left", "right"),
                "clip_action_pose": False,
                "gripper_binary_to_signed": False,
                "num_settle_steps": 0,
                "normalize_task_text": False,
                "apply_offset_norm": False,
                "target_fps": 30.0,
            }
            self._require_platform_values("BimanualYAM", expected)
            if self.joint_signs is not None or self.joint_offsets is not None:
                raise ValueError("Unsafe BimanualYAM config: joint calibration adapters must be disabled.")
            if self.action_feature_names != expected_layout or self.state_feature_names != expected_layout:
                raise ValueError("Unsafe BimanualYAM config: feature layout does not match joint_gripper_14.")
        elif robot_type == "so100_so101":
            expected_layout = [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ]
            expected = {
                "dataset_name": "molmoact2_so100_101",
                "control_mode": "joint",
                "action_dim": 6,
                "proprio_dim": 6,
                "chunk_size": 30,
                "n_action_steps": 30,
                "num_inference_steps": 10,
                "num_flow_samples": 1,
                "image_size": (256, 256),
                "camera_order": ("top", "side"),
                "clip_action_pose": False,
                "gripper_binary_to_signed": False,
                "num_settle_steps": 0,
                "apply_offset_norm": False,
                "target_fps": 30.0,
                "joint_signs": [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
                "joint_offsets": [0.0, 90.0, 90.0, 0.0, 0.0, 0.0],
            }
            self._require_platform_values("SO100/SO101", expected)
            # The imported base preserves its reviewed native prompt behavior
            # (False), while derived fine-tunes may normalize both dataset tasks
            # and serving prompts (True).  This is safe only because the value is
            # serialized with the policy and reconciled into the pack processor.
            if self.action_feature_names != expected_layout or self.state_feature_names != expected_layout:
                raise ValueError("Unsafe SO100/SO101 config: feature layout does not match joint_gripper_6.")
        elif robot_type == "libero":
            expected = {
                "dataset_name": "libero",
                "control_mode": "ee",
                "action_dim": 7,
                "proprio_dim": 8,
                "image_size": (256, 256),
                "camera_order": ("image", "wrist_image"),
                "clip_action_pose": True,
                "gripper_binary_to_signed": False,
                # num_settle_steps is deliberately NOT pinned: it is package-owned (read from the
                # deployment adapter) and differs across package generations (10 on the parent
                # line, 40 on the libero-era line), and the field's own docs recommend 0 -- the
                # env's free in-reset `num_steps_wait` is the right settling lever, not idle
                # frames charged to the episode budget.
                "normalize_task_text": True,
                "apply_offset_norm": False,
                "joint_signs": None,
                "joint_offsets": None,
                "target_fps": 20.0,
            }
            self._require_platform_values("LIBERO", expected)
        else:
            raise ValueError(
                "Strict Perceptron deployment requires a supported robot_type: "
                "bi_yam, so100_so101, or libero."
            )

    def _require_platform_values(self, platform: str, expected: dict[str, object]) -> None:
        actual = {
            **{name: getattr(self, name) for name in expected},
            "image_size": tuple(int(size) for size in self.image_size),
            "camera_order": tuple(self.camera_order),
        }
        mismatches = [
            f"{name}={actual[name]!r} (expected {expected_value!r})"
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        ]
        if mismatches:
            raise ValueError(f"Unsafe {platform} Perceptron config: " + "; ".join(mismatches))

    def validate_features(self) -> None:
        """Ensure a visual input exists and pin the (real-width) state/action features.

        Unlike groot we do not pad observation.state to max_state_dim here; the
        native mharmony packer pads/normalizes the proprio vector for the model.
        """
        self._validate_declared_feature_widths()
        ensure_vla_feature_contract(
            self.input_features,
            self.output_features,
            state_dim=self.proprio_dim,
            action_dim=self.action_dim,
            visual_feature_error=(
                "Perceptron Isaac requires at least one visual input feature "
                "(no FeatureType.VISUAL found in input_features)."
            ),
        )

    def _validate_declared_feature_widths(self) -> None:
        for label, feature, width_name, width in (
            (OBS_STATE, (self.input_features or {}).get(OBS_STATE), "proprio_dim", self.proprio_dim),
            (ACTION, (self.output_features or {}).get(ACTION), "action_dim", self.action_dim),
        ):
            if feature is not None and tuple(feature.shape) != (width,):
                raise ValueError(
                    f"Perceptron Isaac {label} feature width must match {width_name}={width}; "
                    f"got shape={tuple(feature.shape)}."
                )

    @property
    def max_eval_batch_size(self) -> int:
        """ISAAC online history and action queues are per-policy-instance state."""
        return 1

    @property
    def max_eval_parallel_tasks(self) -> int:
        """Concurrent tasks cannot share the online history, RNG clock or action queue."""
        return 1

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.optimizer_warmup_steps,
            num_decay_steps=self.max_train_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        # Checkpoint-owned history window, oldest through the current frame.
        return list(range(-(self.n_obs_steps - 1), 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
