"""Source-compatible DiT metadata and refusal of retired RTC execution requests.

RTC recipe fields describe source provenance only. They never enable prefix execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

DIT_ACTION_EXPERT_CONFIG_SCHEMA_VERSION = 1
DIT_ACTION_EXPERT_CONFIG_V1_FIELDS = (
    "action_dim",
    "action_horizon",
    "num_layers",
    "hidden_dim",
    "num_heads",
    "mlp_ratio",
    "num_inference_steps",
    "timestep_sampling_alpha",
    "timestep_sampling_beta",
    "timestep_sampling_scale",
    "timestep_sampling_offset",
    "train_samples_per_chunk",
    "timestep_embed_dim",
    "rtc_max_delay_steps",
    "rtc_probability",
    "rtc_delay_sampling",
    "rtc_poisson_mean",
    "mask_padded_action_rows",
    "drop_action_dim_overflow",
    "ffn_multiple_of",
    "qk_norm",
    "qk_norm_eps",
    "rope",
    "context_layer_norm",
    "causal_attn",
    "k_batched_cross_attn",
    "k_batched_cross_attn_backend",
)

DIT_CONTEXT_CONDITIONING_FIELDS = ("context_source", "context_layers", "context_hidden_norm")


@dataclass(frozen=True)
class SharedFinalContext:
    """Schema 1 uses one already-normalized final hidden tensor for every block."""


@dataclass(frozen=True)
class LayerHiddenContext:
    """Schema 2 maps ordered post-residual decoder outputs to expert blocks."""

    context_layers: tuple[int, ...]
    context_hidden_norm: Literal["none", "final_norm"]


ActionExpertConditioning = SharedFinalContext | LayerHiddenContext


def validate_action_context(
    action: Mapping[str, Any],
    *,
    schema_version: object,
    backbone_num_layers: int | None = None,
) -> ActionExpertConditioning:
    """Validate versioned context metadata without importing Torch or reading weights."""
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in (1, 2)
    ):
        raise ValueError(f"action_expert schema_version must be 1 or 2; got {schema_version!r}.")
    present = set(DIT_CONTEXT_CONDITIONING_FIELDS) & action.keys()
    if schema_version == 1:
        if present:
            raise ValueError(
                f"action_expert schema_version 1 has unexpected context fields: {sorted(present)}."
            )
        return SharedFinalContext()
    missing = set(DIT_CONTEXT_CONDITIONING_FIELDS) - action.keys()
    if missing:
        raise ValueError(f"action_expert schema_version 2 is missing required fields: {sorted(missing)}.")
    if action["context_source"] != "layer_hidden":
        raise ValueError("action_expert context_source must be 'layer_hidden' for schema_version 2.")
    norm = action["context_hidden_norm"]
    if norm not in ("none", "final_norm"):
        raise ValueError("action_expert context_hidden_norm must be 'none' or 'final_norm'.")
    layers = action["context_layers"]
    count = action.get("num_layers")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("action_expert num_layers must be a positive integer.")
    if not isinstance(layers, list) or len(layers) != count:
        raise ValueError("action_expert context_layers must be a JSON list with one index per expert block.")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in layers):
        raise ValueError("action_expert context_layers entries must be non-boolean, nonnegative integers.")
    if any(left >= right for left, right in zip(layers, layers[1:], strict=False)):
        raise ValueError("action_expert context_layers must be unique and strictly increasing.")
    if backbone_num_layers is not None:
        if backbone_num_layers < 1 or layers[-1] >= backbone_num_layers:
            raise ValueError("action_expert context_layers index is outside the backbone decoder bounds.")
        if norm == "none" and layers[-1] == backbone_num_layers - 1:
            raise ValueError(
                "action_expert context_hidden_norm='none' cannot use the unexposed raw last layer."
            )
    return LayerHiddenContext(tuple(layers), norm)


def validate_dit_action_metadata(
    action: Mapping[str, Any], *, backbone_num_layers: int | None = None
) -> ActionExpertConditioning:
    """Require the exact schema-1 payload or schema-2 payload plus complete conditioning triple."""
    schema = action.get("schema_version")
    conditioning = validate_action_context(
        action, schema_version=schema, backbone_num_layers=backbone_num_layers
    )
    if action.get("type") != "dit":
        raise ValueError("only the 'dit' action expert is supported.")
    required = set(DIT_ACTION_EXPERT_CONFIG_V1_FIELDS) | {"schema_version", "type"}
    if isinstance(conditioning, LayerHiddenContext):
        required.update(DIT_CONTEXT_CONDITIONING_FIELDS)
    missing = sorted(required - action.keys())
    unexpected = sorted(action.keys() - required)
    if missing or unexpected:
        raise ValueError(
            f"action_expert metadata key mismatch: missing={missing}, unexpected fields={unexpected}."
        )
    return conditioning


@dataclass(eq=False)
class ActionExpertStepModulation:
    """Precomputed AdaLN values for one ordinary action-expert step."""

    conditioning: torch.Tensor
    block_modulations: Sequence[tuple[torch.Tensor, ...]]
    final_modulation: tuple[torch.Tensor, torch.Tensor]


def reject_rtc_prefix(
    action_prefix: torch.Tensor | None,
    prefix_length: int | Sequence[int] | torch.Tensor | None,
) -> None:
    """Accept legacy RTC-off arguments, but fail before execution on a prefix request."""
    import torch

    if action_prefix is not None:
        raise ValueError("RTC action prefixes are unsupported.")
    if prefix_length is None:
        return
    if isinstance(prefix_length, torch.Tensor):
        enabled = bool(prefix_length.ne(0).any())
    elif isinstance(prefix_length, int):
        enabled = prefix_length != 0
    else:
        enabled = any(value != 0 for value in prefix_length)
    if enabled:
        raise ValueError("RTC prefix_length must be zero or None.")
