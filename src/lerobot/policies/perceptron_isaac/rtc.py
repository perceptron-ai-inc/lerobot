"""Source-compatible DiT metadata and refusal of retired RTC execution requests.

RTC recipe fields describe source provenance only. They never enable prefix execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
