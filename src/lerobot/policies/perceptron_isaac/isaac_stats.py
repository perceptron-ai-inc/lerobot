"""Stats helpers for Perceptron Isaac native eval.

The JSON schema matches the flow-matching stats sidecars used by Isaac
checkpoints. Keeping the loader and quantile math local lets policy/processor
code normalize and unnormalize actions without external runtime helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

# The native quantile math is hardcoded to this epsilon, matching Genesis's
# quantile_normalization default. Checkpoints declaring a different value cannot be served
# faithfully, so the importer rejects them rather than silently substituting this one.
NATIVE_NORMALIZATION_EPS = 1e-6


def _as_vector(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector, got shape {arr.shape}.")
    return arr


def _effective_quantile_bounds(
    q01: np.ndarray,
    q99: np.ndarray,
    *,
    minimum: np.ndarray | None,
    maximum: np.ndarray | None,
    eps: float = NATIVE_NORMALIZATION_EPS,
    error_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(q01, dtype=np.float32)
    upper = np.asarray(q99, dtype=np.float32)
    if lower.shape != upper.shape:
        raise ValueError(f"{error_prefix}: lower shape {lower.shape} != upper shape {upper.shape}.")
    # Match Genesis _coerce_quantile_stats: corrupt stats must fail loudly rather than
    # silently sign-flipping or NaN-ing a normalized dimension.
    if lower.size == 0:
        raise ValueError(f"{error_prefix}: stats must be non-empty.")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError(f"{error_prefix}: stats must be finite.")
    if bool(np.any(upper < lower)):
        raise ValueError(f"{error_prefix}: stats must satisfy q99 >= q01.")
    # Genesis has no identity/collapsed special case: it normalizes a collapsed dim through
    # the eps denominator, so a raw value equal to q01 maps to -1.0 rather than 0.0.
    collapsed = upper - lower <= eps
    if minimum is None or maximum is None:
        return lower, upper

    min_arr = np.asarray(minimum, dtype=np.float32)
    max_arr = np.asarray(maximum, dtype=np.float32)
    if min_arr.shape != lower.shape or max_arr.shape != lower.shape:
        raise ValueError(f"{error_prefix}: min/max shapes must match q01/q99.")
    if not np.isfinite(min_arr).all() or not np.isfinite(max_arr).all():
        raise ValueError(f"{error_prefix}: min/max stats must be finite.")
    if bool(np.any(max_arr < min_arr)):
        raise ValueError(f"{error_prefix}: min/max stats must satisfy max >= min.")

    use_minmax = collapsed & (max_arr - min_arr > eps)
    if bool(use_minmax.any()):
        lower = lower.copy()
        upper = upper.copy()
        lower[use_minmax] = min_arr[use_minmax]
        upper[use_minmax] = max_arr[use_minmax]
    return lower, upper


def _normalize_quantile(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    eps: float = NATIVE_NORMALIZATION_EPS,
    error_prefix: str,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[-1] != lower.shape[0]:
        raise ValueError(f"{error_prefix}: last dim {arr.shape[-1]} != stats dim {lower.shape[0]}.")
    return ((arr - lower) / (upper - lower + eps)) * 2.0 - 1.0


def _unnormalize_quantile(
    x_norm: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    eps: float = NATIVE_NORMALIZATION_EPS,
    error_prefix: str,
) -> np.ndarray:
    arr = np.asarray(x_norm, dtype=np.float32)
    if arr.shape[-1] != lower.shape[0]:
        raise ValueError(f"{error_prefix}: last dim {arr.shape[-1]} != stats dim {lower.shape[0]}.")
    # Include +eps so this is the exact inverse of _normalize_quantile, matching Genesis
    # unnormalize_quantile.
    return lower + (arr + 1.0) * 0.5 * (upper - lower + eps)


def _apply_normalize_mask(transformed: np.ndarray, raw: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Keep masked-out dimensions at their raw values, as MolmoAct2's np.where(mask, ...) does."""
    if mask is None:
        return transformed
    return np.where(mask, transformed, np.asarray(raw, dtype=np.float32)).astype(np.float32, copy=False)


def gripper_normalize_mask(feature_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """True per dimension except where the feature name marks a gripper.

    Matches MolmoAct2's ``_default_feature_mask``: a substring test on the feature name,
    so it follows the checkpoint's declared layout instead of a hardcoded index.
    """
    names = [str(name) for name in feature_names]
    if not names:
        raise ValueError("gripper_normalize_mask requires a non-empty feature name list.")
    mask = np.array(["gripper" not in name.lower() for name in names], dtype=bool)
    if bool(mask.all()):
        raise ValueError(
            f"No gripper dimension found in feature names {names}; refusing to build a "
            "no-op passthrough mask, which would silently normalize the grippers."
        )
    return mask


@dataclass(frozen=True, slots=True)
class IsaacStatsBlock:
    """Quantile stats for one vector, with an optional per-dimension normalize mask.

    ``mask`` mirrors MolmoAct2's stat key of the same name: True normalizes the dimension,
    False passes the raw physical value straight through. None (the default) normalizes
    every dimension, which is Genesis's own behaviour and what every checkpoint imported
    before the mask existed encodes.
    """

    q01: np.ndarray
    q99: np.ndarray
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    mask: np.ndarray | None = None
    _effective_lower: np.ndarray = field(init=False, repr=False, compare=False)
    _effective_upper: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "q01", _as_vector(self.q01, name="q01"))
        object.__setattr__(self, "q99", _as_vector(self.q99, name="q99"))
        if self.q01.shape != self.q99.shape:
            raise ValueError(f"q01 shape {self.q01.shape} != q99 shape {self.q99.shape}.")
        if self.minimum is not None:
            object.__setattr__(self, "minimum", _as_vector(self.minimum, name="min"))
        if self.maximum is not None:
            object.__setattr__(self, "maximum", _as_vector(self.maximum, name="max"))
        if (self.minimum is None) != (self.maximum is None):
            raise ValueError("minimum and maximum stats must be provided together.")
        if self.minimum is not None:
            assert self.maximum is not None
            if self.minimum.shape != self.q01.shape or self.maximum.shape != self.q01.shape:
                raise ValueError("min/max stats shape must match q01/q99.")
        if self.mask is not None:
            mask = np.asarray(self.mask).reshape(-1).astype(bool)
            if mask.shape != self.q01.shape:
                raise ValueError(f"mask shape {mask.shape} must match q01/q99 {self.q01.shape}.")
            if bool(mask.all()):
                # An all-true mask is the same as no mask; normalize it away so serialized
                # state and equality comparisons do not distinguish the two spellings.
                mask = None
            object.__setattr__(self, "mask", mask)
        lower, upper = _effective_quantile_bounds(
            self.q01,
            self.q99,
            minimum=self.minimum,
            maximum=self.maximum,
            error_prefix="Isaac stats",
        )
        object.__setattr__(self, "_effective_lower", lower)
        object.__setattr__(self, "_effective_upper", upper)

    @property
    def dim(self) -> int:
        return int(self.q01.shape[0])

    def effective_bounds(self, *, eps: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        if eps is None:
            return self._effective_lower, self._effective_upper
        lower, upper = _effective_quantile_bounds(
            self.q01,
            self.q99,
            minimum=self.minimum,
            maximum=self.maximum,
            eps=eps,
            error_prefix="Isaac action stats",
        )
        return lower, upper

    def effective_action_bounds(self, *, eps: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self.effective_bounds(eps=eps)


@dataclass(frozen=True, slots=True)
class IsaacNormalizationStats:
    action: IsaacStatsBlock
    proprio: IsaacStatsBlock
    target_fps: float | None
    action_horizon: int
    schema: str
    profile_id: str | None = None
    profile_scope: str | None = None
    validation_status: str | None = None

    @property
    def action_dim(self) -> int:
        return self.action.dim

    @property
    def proprio_dim(self) -> int:
        return self.proprio.dim


def apply_gripper_passthrough(
    stats: IsaacNormalizationStats,
    *,
    action_feature_names: list[str] | tuple[str, ...],
    state_feature_names: list[str] | tuple[str, ...],
) -> IsaacNormalizationStats:
    """Return `stats` with gripper dimensions marked passthrough on both blocks.

    Genesis normalizes every dimension including the grippers; MolmoAct2 masks its gripper
    dimensions out and feeds them in physical units. This is the ISAAC-side switch for the
    latter. It is a change to the representation the base checkpoint was pretrained on, so
    callers opt in explicitly rather than getting it by default.
    """
    action_mask = gripper_normalize_mask(action_feature_names)
    state_mask = gripper_normalize_mask(state_feature_names)
    if action_mask.shape[0] != stats.action.dim:
        raise ValueError(
            f"action feature names cover {action_mask.shape[0]} dims but action stats have "
            f"{stats.action.dim}."
        )
    if state_mask.shape[0] != stats.proprio.dim:
        raise ValueError(
            f"state feature names cover {state_mask.shape[0]} dims but proprio stats have "
            f"{stats.proprio.dim}."
        )

    return replace(
        stats,
        action=replace(stats.action, mask=action_mask),
        proprio=replace(stats.proprio, mask=state_mask),
    )


def _stats_block_from_json(raw: dict[str, Any]) -> IsaacStatsBlock:
    return IsaacStatsBlock(
        q01=raw["q01"],
        q99=raw["q99"],
        minimum=raw.get("min"),
        maximum=raw.get("max"),
        mask=raw.get("mask"),
    )


def load_isaac_stats(path: str | Path) -> IsaacNormalizationStats:
    return isaac_stats_from_dict(json.loads(Path(path).read_text()))


def load_isaac_suite_stats(path: str | Path) -> dict[str, IsaacNormalizationStats]:
    """Load one normalization block per dataset scope, keyed by scope name.

    Multi-suite LIBERO training needs this because Genesis normalizes every LIBERO suite by its
    own quantiles and the suites are genuinely not interchangeable -- end-effector height spans
    [0.916, 1.286] in ``libero_spatial`` but [0.010, 0.337] in ``libero_object``. Those ranges do
    not overlap, so normalizing one suite with another's stats drives proprio far outside
    [-1, 1] instead of filling it.

    Expects ``{"<scope>": {<same shape as load_isaac_stats>}, ...}``.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Isaac suite stats at {path} must be a non-empty JSON object of scope -> stats.")
    scopes: dict[str, IsaacNormalizationStats] = {}
    for scope, block in raw.items():
        if not isinstance(block, dict):
            raise ValueError(f"Isaac suite stats scope {scope!r} must map to an object.")
        scopes[str(scope)] = isaac_stats_from_dict(block)
    horizons = {s.action_horizon for s in scopes.values()}
    dims = {s.action_dim for s in scopes.values()}
    fps = {s.target_fps for s in scopes.values()}
    if len(horizons) != 1 or len(dims) != 1 or len(fps) != 1:
        raise ValueError(
            "Isaac suite stats must agree on action_horizon/action_dim/target_fps across scopes; "
            f"got horizons={sorted(horizons)}, dims={sorted(dims)}, target_fps={sorted(fps, key=str)}."
        )
    return scopes


def isaac_stats_from_dict(raw: dict[str, Any]) -> IsaacNormalizationStats:
    action = _stats_block_from_json(raw["action"])
    proprio = _stats_block_from_json(raw["proprio"])
    action_dim = int(raw["action_dim"])
    if action_dim != action.dim:
        raise ValueError(f"action_dim {action_dim} != action stats dim {action.dim}.")
    raw_proprio_dim = raw.get("proprio_dim")
    if raw_proprio_dim is not None and int(raw_proprio_dim) != proprio.dim:
        raise ValueError(f"proprio_dim {raw_proprio_dim} != proprio stats dim {proprio.dim}.")
    return IsaacNormalizationStats(
        action=action,
        proprio=proprio,
        target_fps=float(raw["target_fps"]) if raw.get("target_fps") is not None else None,
        action_horizon=int(raw["action_horizon"]),
        schema=str(raw["schema"]),
        profile_id=str(raw["profile_id"]) if raw.get("profile_id") is not None else None,
        profile_scope=str(raw["profile_scope"]) if raw.get("profile_scope") is not None else None,
        validation_status=(
            str(raw["validation_status"]) if raw.get("validation_status") is not None else None
        ),
    )


def _stats_block_to_json(block: IsaacStatsBlock) -> dict[str, Any]:
    return {
        "q01": block.q01.astype(np.float64).tolist(),
        "q99": block.q99.astype(np.float64).tolist(),
        "min": None if block.minimum is None else block.minimum.astype(np.float64).tolist(),
        "max": None if block.maximum is None else block.maximum.astype(np.float64).tolist(),
        "mask": None if block.mask is None else block.mask.astype(bool).tolist(),
    }


def isaac_stats_to_dict(stats: IsaacNormalizationStats) -> dict[str, Any]:
    """Serialize normalization stats to the JSON schema `load_isaac_stats` reads.

    The per-dimension normalize masks are embedded: a path-loaded copy must carry
    the gripper passthrough itself, because the loading step's `normalize_gripper`
    flag defaults to True and would otherwise silently drop the mask.
    """
    return {
        "action": _stats_block_to_json(stats.action),
        "proprio": _stats_block_to_json(stats.proprio),
        "action_dim": int(stats.action_dim),
        "proprio_dim": int(stats.proprio.dim),
        "action_horizon": int(stats.action_horizon),
        "target_fps": stats.target_fps,
        "schema": stats.schema,
        "profile_id": stats.profile_id,
        "profile_scope": stats.profile_scope,
        "validation_status": stats.validation_status,
    }


def save_isaac_stats(stats: IsaacNormalizationStats, path: str | Path) -> None:
    """Write checkpoint-local stats JSON that round-trips through `load_isaac_stats`."""
    require_closed_loop_safe_isaac_stats(stats)
    Path(path).write_text(json.dumps(isaac_stats_to_dict(stats), indent=2) + "\n")


def _block_value_differences(name: str, a: IsaacStatsBlock, b: IsaacStatsBlock) -> list[str]:
    problems: list[str] = []
    if a.dim != b.dim:
        return [f"{name}: dim {a.dim} != {b.dim}"]
    for field_name in ("q01", "q99", "minimum", "maximum"):
        left = getattr(a, field_name)
        right = getattr(b, field_name)
        if (left is None) != (right is None):
            problems.append(f"{name}.{field_name}: present on one side only")
        elif left is not None and not np.array_equal(left, right):
            delta = float(np.abs(np.asarray(left) - np.asarray(right)).max())
            problems.append(f"{name}.{field_name}: values differ (max abs delta {delta:.6g})")
    if (a.mask is None) != (b.mask is None) or (a.mask is not None and not np.array_equal(a.mask, b.mask)):
        problems.append(f"{name}.mask: {a.mask} != {b.mask}")
    return problems


def isaac_stats_value_differences(a: IsaacNormalizationStats, b: IsaacNormalizationStats) -> list[str]:
    """Compare two stats bundles by VALUES, never by labels.

    `profile_id`/`profile_scope`/`validation_status` are deliberately excluded: distinct
    stats bundles are known to share a `profile_id` (base vs finetune-dataset stats), so
    any identity check keyed on labels would falsely equate them. An empty return means
    the bundles are numerically interchangeable.
    """
    problems = _block_value_differences("action", a.action, b.action)
    problems += _block_value_differences("proprio", a.proprio, b.proprio)
    if (a.target_fps is None) != (b.target_fps is None) or (
        a.target_fps is not None and float(a.target_fps) != float(b.target_fps)
    ):
        problems.append(f"target_fps: {a.target_fps} != {b.target_fps}")
    if int(a.action_horizon) != int(b.action_horizon):
        problems.append(f"action_horizon: {a.action_horizon} != {b.action_horizon}")
    return problems


def _encode_label(value: str) -> torch.Tensor:
    data = str(value).encode("utf-8")
    # torch.frombuffer rejects zero-length buffers; an empty label must still serialize,
    # or the checkpoint save crashes at the end of the run.
    if not data:
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def _decode_label(state: dict[str, torch.Tensor], key: str) -> str | None:
    tensor = state.get(key)
    if tensor is None:
        return None
    return bytes(tensor.detach().cpu().to(torch.uint8).numpy().tolist()).decode("utf-8")


def isaac_stats_to_state_dict(stats: IsaacNormalizationStats) -> dict[str, torch.Tensor]:
    """Serialize normalization into processor-owned safetensors state."""
    state: dict[str, torch.Tensor] = {
        "action.q01": torch.from_numpy(stats.action.q01.copy()),
        "action.q99": torch.from_numpy(stats.action.q99.copy()),
        "proprio.q01": torch.from_numpy(stats.proprio.q01.copy()),
        "proprio.q99": torch.from_numpy(stats.proprio.q99.copy()),
        "metadata.target_fps": torch.tensor(
            float("nan") if stats.target_fps is None else stats.target_fps,
            dtype=torch.float64,
        ),
        "metadata.action_horizon": torch.tensor(stats.action_horizon, dtype=torch.int64),
        "metadata.action_dim": torch.tensor(stats.action_dim, dtype=torch.int64),
    }
    for prefix, block in (("action", stats.action), ("proprio", stats.proprio)):
        if block.minimum is not None:
            state[f"{prefix}.min"] = torch.from_numpy(block.minimum.copy())
        if block.maximum is not None:
            state[f"{prefix}.max"] = torch.from_numpy(block.maximum.copy())
        if block.mask is not None:
            # Serialized so eval inherits the passthrough from the checkpoint instead of
            # depending on the serving config agreeing with the training config.
            state[f"{prefix}.mask"] = torch.from_numpy(block.mask.copy())
    # Carry the closed-loop-safety labels through the round trip. Dropping them would make
    # require_closed_loop_safe_isaac_stats vacuously pass on any reconstructed bundle, so an
    # ablation-labeled profile could drive a robot. safetensors stores tensors only, so encode
    # the strings as UTF-8 bytes.
    state["metadata.schema"] = _encode_label(stats.schema)
    for key, value in (
        ("profile_id", stats.profile_id),
        ("profile_scope", stats.profile_scope),
        ("validation_status", stats.validation_status),
    ):
        if value is not None:
            state[f"metadata.{key}"] = _encode_label(value)
    return state


def isaac_stats_from_state_dict(state: dict[str, torch.Tensor]) -> IsaacNormalizationStats:
    """Reconstruct processor-owned normalization from safetensors state."""
    required = {
        "action.q01",
        "action.q99",
        "proprio.q01",
        "proprio.q99",
        "metadata.target_fps",
        "metadata.action_horizon",
        "metadata.action_dim",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(f"ISAAC processor stats state is missing keys: {sorted(missing)}.")

    def _array(key: str) -> np.ndarray:
        return state[key].detach().cpu().float().numpy().copy()

    def _mask(key: str) -> np.ndarray | None:
        if key not in state:
            return None
        return state[key].detach().cpu().bool().numpy().copy()

    action = IsaacStatsBlock(
        q01=_array("action.q01"),
        q99=_array("action.q99"),
        minimum=_array("action.min") if "action.min" in state else None,
        maximum=_array("action.max") if "action.max" in state else None,
        mask=_mask("action.mask"),
    )
    proprio = IsaacStatsBlock(
        q01=_array("proprio.q01"),
        q99=_array("proprio.q99"),
        minimum=_array("proprio.min") if "proprio.min" in state else None,
        maximum=_array("proprio.max") if "proprio.max" in state else None,
        mask=_mask("proprio.mask"),
    )
    target_fps = float(state["metadata.target_fps"].item())
    if np.isnan(target_fps):
        target_fps = None
    action_dim = int(state["metadata.action_dim"].item())
    if action_dim != action.dim:
        raise ValueError(f"ISAAC processor action_dim {action_dim} != stats dim {action.dim}.")
    return IsaacNormalizationStats(
        action=action,
        proprio=proprio,
        target_fps=target_fps,
        action_horizon=int(state["metadata.action_horizon"].item()),
        schema=_decode_label(state, "metadata.schema") or "flow_matching_stats_v1",
        profile_id=_decode_label(state, "metadata.profile_id"),
        profile_scope=_decode_label(state, "metadata.profile_scope"),
        validation_status=_decode_label(state, "metadata.validation_status"),
    )


def require_closed_loop_safe_isaac_profile(
    *,
    profile_id: str | None,
    profile_scope: str | None,
    validation_status: str | None,
) -> None:
    """Reject normalization metadata explicitly labeled as a closed-loop ablation."""
    unsafe = (
        validation_status == "unsafe_closed_loop_ablation"
        or profile_scope == "output_denormalization_ablation"
        or bool(profile_id and profile_id.startswith("ablation/"))
    )
    if unsafe:
        raise ValueError(
            "ISAAC normalization profile is explicitly unsafe for closed-loop deployment: "
            f"profile_id={profile_id!r}, profile_scope={profile_scope!r}, "
            f"validation_status={validation_status!r}."
        )


def require_closed_loop_safe_isaac_stats(stats: IsaacNormalizationStats) -> None:
    """Reject normalization bundles explicitly labeled as closed-loop ablations."""
    require_closed_loop_safe_isaac_profile(
        profile_id=stats.profile_id,
        profile_scope=stats.profile_scope,
        validation_status=stats.validation_status,
    )


def normalize_isaac_proprio(
    x: np.ndarray, stats: IsaacStatsBlock, *, eps: float = NATIVE_NORMALIZATION_EPS
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape != stats.q01.shape:
        raise ValueError(f"proprio shape {arr.shape} does not match stats {stats.q01.shape}.")
    lower, upper = stats.effective_bounds(eps=eps)
    normalized = _normalize_quantile(
        arr,
        lower,
        upper,
        eps=eps,
        error_prefix="normalize_isaac_proprio",
    )
    # MolmoAct2 masks against the original raw input here (its clip writes to a separate
    # name), so a passthrough dimension reaches the model in physical units.
    return _apply_normalize_mask(normalized, arr, stats.mask)


def normalize_isaac_actions(
    x: np.ndarray, stats: IsaacStatsBlock, *, eps: float = NATIVE_NORMALIZATION_EPS
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != stats.dim:
        raise ValueError(f"action prefix must have shape [P, {stats.dim}], got {arr.shape}.")
    lower, upper = stats.effective_action_bounds()
    normalized = _normalize_quantile(
        arr,
        lower,
        upper,
        eps=eps,
        error_prefix="normalize_isaac_actions",
    )
    return _apply_normalize_mask(normalized, arr, stats.mask)


def unnormalize_isaac_actions(
    x_norm: np.ndarray,
    stats: IsaacStatsBlock,
    *,
    clip: bool = False,
    clip_normalized_max: float = 1.0,
) -> np.ndarray:
    lower, upper = stats.effective_action_bounds()
    arr = np.asarray(x_norm, dtype=np.float32)
    if clip:
        bound = float(clip_normalized_max)
        if bound <= 0.0:
            raise ValueError(f"clip_normalized_max must be > 0, got {clip_normalized_max!r}.")
        arr = np.clip(arr, -bound, bound)
    unnormalized = _unnormalize_quantile(
        arr,
        lower,
        upper,
        error_prefix="unnormalize_isaac_actions",
    )
    # MolmoAct2 reassigns its working array to the clipped values before masking, so the
    # passthrough carries the clipped model output rather than the raw one. Mirror that:
    # for a gripper already living in physical [0, 1] the clip is inert, but the two
    # stacks stay bit-comparable on out-of-range predictions.
    return _apply_normalize_mask(unnormalized, arr, stats.mask)
