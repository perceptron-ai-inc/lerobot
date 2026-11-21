"""Import-light DiT metadata and real-time chunking (RTC) inference contracts.

Torch is imported lazily so metadata validation stays import-light. RTC prefix
execution is validated against the RTC support declared by the checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


def rtc_is_enabled(*, max_delay_steps: int, probability: float | None) -> bool:
    """Return whether training can produce a non-empty RTC prefix."""
    return int(max_delay_steps) > 0 and probability != 0.0


def effective_rtc_max_prefix_steps(
    *,
    max_delay_steps: int,
    probability: float | None,
    action_horizon: int,
) -> int:
    """Return the largest prefix length in the RTC training support."""
    max_delay_steps = int(max_delay_steps)
    action_horizon = int(action_horizon)
    if max_delay_steps < 0:
        raise ValueError(f"max_delay_steps must be >= 0; got {max_delay_steps}.")
    if probability is not None and not 0.0 <= float(probability) <= 1.0:
        raise ValueError(f"probability must be None or in [0, 1]; got {probability}.")
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be >= 1; got {action_horizon}.")
    if not rtc_is_enabled(max_delay_steps=max_delay_steps, probability=probability):
        return 0
    return min(max_delay_steps, action_horizon - 1)


@dataclass(frozen=True)
class RtcCapability:
    """A checkpoint's trained-RTC inference budget and the stage that last trained its weights.

    ``prefix_budget`` is the declared inference prefix budget: the longest prefix the
    sampler admits, 0 when the checkpoint declares none. ``declared_max_delay_steps`` and
    ``declared_probability`` repeat the action-expert recipe, which training stages carry
    forward unchanged. ``final_stage_max_delay_steps`` is what the final training stage
    itself sampled: None when the declaring recipe is that stage, 0 when a later stage
    trained without RTC prefixes.
    """

    prefix_budget: int
    declared_max_delay_steps: int
    declared_probability: float | None
    final_stage_max_delay_steps: int | None

    @property
    def inherited(self) -> bool:
        """Whether the budget is an ancestor's declaration rather than this stage's training.

        An inherited budget still admits prefixes. It records provenance only: nothing has
        validated prefix inference on the weights the final stage produced.
        """
        return self.prefix_budget > 0 and self.final_stage_max_delay_steps == 0


def resolve_rtc_capability(
    action: Mapping[str, Any] | None,
    *,
    prediction_steps: int,
    final_stage_max_delay_steps: int | None,
) -> RtcCapability:
    """Combine an action-expert recipe with the final training stage's RTC record.

    ``action`` is the checkpoint's ``action_expert`` metadata. A missing recipe, or a legacy
    block without RTC fields, declares no budget, so trained prefixes stay rejected.
    """
    max_delay = 0 if action is None else action.get("rtc_max_delay_steps", 0)
    probability = None if action is None else action.get("rtc_probability")
    if isinstance(max_delay, bool) or not isinstance(max_delay, int):
        raise ValueError(f"action_expert rtc_max_delay_steps must be an integer, got {max_delay!r}.")
    if probability is not None and (
        isinstance(probability, bool) or not isinstance(probability, int | float)
    ):
        raise ValueError(f"action_expert rtc_probability must be a number or null, got {probability!r}.")
    return RtcCapability(
        prefix_budget=effective_rtc_max_prefix_steps(
            max_delay_steps=max_delay,
            probability=probability,
            action_horizon=prediction_steps,
        ),
        declared_max_delay_steps=max_delay,
        declared_probability=None if probability is None else float(probability),
        final_stage_max_delay_steps=final_stage_max_delay_steps,
    )


@dataclass(frozen=True, eq=False)
class ResolvedRTCActionPrefix:
    """Validated RTC prefix geometry shared by native and HF sampling."""

    source_dim: int | None
    output_dim: int
    lengths: torch.Tensor | None
    max_length: int


@dataclass(eq=False)
class ActionExpertStepModulation:
    """Precomputed AdaLN values for one ordinary action-expert step."""

    conditioning: torch.Tensor
    block_modulations: Sequence[tuple[torch.Tensor, ...]]
    final_modulation: tuple[torch.Tensor, torch.Tensor]


def prepare_rtc_conditioning(
    base_timesteps: torch.Tensor,
    prefix_mask: torch.Tensor,
    *,
    time_conditioning: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compact suffix/prefix conditioning for checkpointed DiT blocks.

    RTC has a sampled suffix timestep per chunk and the fixed clean timestep 1
    for prefix rows. Blocks project these two values inside their activation-
    checkpointed forward, then select them per row.
    """
    import torch

    if base_timesteps.dim() != 1:
        raise ValueError(f"base_timesteps must have shape [B]; got {tuple(base_timesteps.shape)}.")
    if prefix_mask.dim() != 2 or prefix_mask.shape[0] != base_timesteps.shape[0]:
        raise ValueError(
            f"prefix_mask must have shape [B,H] with B={base_timesteps.shape[0]}; got {tuple(prefix_mask.shape)}."
        )
    suffix_conditioning = time_conditioning(base_timesteps)
    # The clean RTC prefix always uses flow time 1, independent of the chunk or
    # flow draw. Compute it once; block projections broadcast this single row.
    prefix_conditioning = time_conditioning(
        torch.ones(1, device=base_timesteps.device, dtype=base_timesteps.dtype)
    )
    row_mask = prefix_mask.to(device=base_timesteps.device, dtype=torch.bool).unsqueeze(-1)
    return suffix_conditioning, prefix_conditioning, row_mask


def project_rtc_modulation(
    suffix_conditioning: torch.Tensor,
    prefix_conditioning: torch.Tensor,
    prefix_mask: torch.Tensor,
    *,
    modulation: Callable[[torch.Tensor], torch.Tensor],
    chunks: int,
) -> tuple[torch.Tensor, ...]:
    """Project two compact RTC values and select the result per action row."""
    import torch

    suffix = modulation(suffix_conditioning)
    prefix = modulation(prefix_conditioning)
    selected = torch.where(prefix_mask, prefix.unsqueeze(1), suffix.unsqueeze(1))
    return tuple(selected.chunk(chunks, dim=-1))


def validate_rtc_prefix_capability(
    *,
    prefix_length: int,
    max_delay_steps: int,
    probability: float | None,
    action_horizon: int,
    allow_ood: bool = False,
) -> int:
    """Validate a requested prefix against the RTC training support."""
    prefix_length = int(prefix_length)
    effective_max = effective_rtc_max_prefix_steps(
        max_delay_steps=max_delay_steps,
        probability=probability,
        action_horizon=action_horizon,
    )
    if prefix_length < 0:
        raise ValueError(f"prefix_length must be >= 0; got {prefix_length}.")
    if prefix_length > effective_max and not allow_ood:
        raise ValueError(
            f"prefix_length={prefix_length} exceeds the maximum supported RTC prefix {effective_max} "
            f"(configured max_delay_steps={int(max_delay_steps)}, probability={probability}, "
            f"action_horizon={int(action_horizon)}). Pass allow_ood_rtc_prefix=True only for an explicit "
            "out-of-distribution research experiment."
        )
    return effective_max


def resolve_rtc_action_prefix(
    *,
    action_prefix: torch.Tensor | None,
    prefix_length: int | Sequence[int] | torch.Tensor | None,
    action_dim: int | None,
    batch_size: int,
    action_horizon: int,
    expert_action_dim: int,
    rtc_max_delay_steps: int,
    rtc_probability: float | None,
    device: torch.device,
    allow_ood: bool = False,
) -> ResolvedRTCActionPrefix:
    """Validate and normalize an RTC action-prefix request."""
    import torch

    action_horizon = int(action_horizon)
    expert_action_dim = int(expert_action_dim)
    output_dim = expert_action_dim if action_dim is None else int(action_dim)
    if output_dim < 1 or output_dim > expert_action_dim:
        raise ValueError(f"action_dim must be in [1, {expert_action_dim}]; got {output_dim}.")

    if action_prefix is None:
        if isinstance(prefix_length, torch.Tensor):
            has_prefix_length = bool(prefix_length.detach().to(device="cpu").any().item())
        elif prefix_length is None:
            has_prefix_length = False
        elif isinstance(prefix_length, int):
            has_prefix_length = prefix_length != 0
        else:
            has_prefix_length = any(int(value) != 0 for value in prefix_length)
        if has_prefix_length:
            raise ValueError("action_prefix is required when prefix_length is non-zero.")
        return ResolvedRTCActionPrefix(None, output_dim, None, 0)

    if action_prefix.dim() != 3 or action_prefix.shape[0] != batch_size:
        raise ValueError(
            f"action_prefix must have shape [B, P, D] with B={batch_size}; got {tuple(action_prefix.shape)}."
        )
    source_dim = int(action_prefix.shape[2])
    if source_dim < 1 or source_dim > expert_action_dim:
        raise ValueError(f"action_prefix last dim must be in [1, {expert_action_dim}]; got {source_dim}.")

    if prefix_length is None:
        lengths = torch.full((batch_size,), action_prefix.shape[1], device=device, dtype=torch.long)
    elif isinstance(prefix_length, torch.Tensor):
        lengths = prefix_length.to(device=device, dtype=torch.long)
        if lengths.dim() == 0:
            lengths = lengths.expand(batch_size)
        elif tuple(lengths.shape) != (batch_size,):
            raise ValueError(
                f"prefix_length tensor must have shape [] or [{batch_size}]; got {tuple(lengths.shape)}."
            )
    elif isinstance(prefix_length, int):
        lengths = torch.full((batch_size,), prefix_length, device=device, dtype=torch.long)
    else:
        lengths = torch.as_tensor(list(prefix_length), device=device, dtype=torch.long)
        if tuple(lengths.shape) != (batch_size,):
            raise ValueError(
                f"prefix_length sequence must have length {batch_size}; got {tuple(lengths.shape)}."
            )

    max_allowed = action_horizon - 1
    # One host transfer for every check below: this runs per /predict and per chunk in the
    # inference-MSE eval, so each extra `.item()` on a device tensor is a blocking sync.
    lengths_cpu = lengths.detach().to("cpu")
    if bool(((lengths_cpu < 0) | (lengths_cpu > max_allowed)).any().item()):
        raise ValueError(
            f"prefix_length values must be in [0, {max_allowed}] so RTC leaves at least one model-generated "
            f"suffix row; got {lengths_cpu.tolist()}."
        )
    max_length = int(lengths_cpu.max().item()) if lengths_cpu.numel() else 0
    validate_rtc_prefix_capability(
        prefix_length=max_length,
        max_delay_steps=rtc_max_delay_steps,
        probability=rtc_probability,
        action_horizon=action_horizon,
        allow_ood=allow_ood,
    )
    if action_prefix.shape[1] < max_length:
        raise ValueError(
            f"action_prefix has only {action_prefix.shape[1]} rows but max prefix_length={max_length}."
        )
    if max_length > 0 and action_dim is None and source_dim != expert_action_dim:
        raise ValueError(
            "action_dim is required when action_prefix is narrower than the expert action width; "
            f"got prefix dim {source_dim} and expert width {expert_action_dim}."
        )
    if max_length > 0 and source_dim not in (output_dim, expert_action_dim):
        raise ValueError(
            "action_prefix last dim must match action_dim or the expert action width; "
            f"got prefix dim {source_dim}, action_dim {output_dim}, expert width {expert_action_dim}."
        )
    return ResolvedRTCActionPrefix(source_dim, output_dim, lengths, max_length)


def materialize_rtc_action_prefix(
    resolved: ResolvedRTCActionPrefix,
    action_prefix: torch.Tensor | None,
    *,
    batch_size: int,
    action_horizon: int,
    expert_action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    dim_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Materialize the fixed prefix values and their row mask."""
    import torch

    if action_prefix is None or resolved.max_length == 0:
        return None, None
    assert resolved.lengths is not None
    assert resolved.source_dim is not None
    prefix_tensor = torch.zeros(batch_size, action_horizon, expert_action_dim, dtype=dtype, device=device)
    prefix_tensor[:, : resolved.max_length, : resolved.source_dim] = action_prefix[
        :, : resolved.max_length
    ].to(
        device=device,
        dtype=dtype,
    )
    if dim_mask is not None:
        prefix_tensor = prefix_tensor * dim_mask
    prefix_mask = torch.arange(action_horizon, device=device).view(
        1, action_horizon, 1
    ) < resolved.lengths.view(
        batch_size,
        1,
        1,
    )
    return prefix_tensor, prefix_mask


def integrate_rtc_euler(
    initial_state: torch.Tensor,
    *,
    num_steps: int,
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    prefix_tensor: torch.Tensor | None,
    prefix_mask: torch.Tensor | None,
    dim_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Integrate a flow velocity while pinning an optional RTC prefix."""
    import torch

    def apply_masks(state: torch.Tensor) -> torch.Tensor:
        if dim_mask is not None:
            state = state * dim_mask
        if prefix_mask is not None and prefix_tensor is not None:
            state = torch.where(prefix_mask, prefix_tensor, state)
        return state

    state = apply_masks(initial_state)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        flow_time = step / num_steps
        velocity = velocity_fn(state, flow_time)
        if dim_mask is not None:
            velocity = velocity * dim_mask
        if prefix_mask is not None:
            velocity = torch.where(prefix_mask, torch.zeros_like(velocity), velocity)
        state = apply_masks(state + dt * velocity)
    return state
