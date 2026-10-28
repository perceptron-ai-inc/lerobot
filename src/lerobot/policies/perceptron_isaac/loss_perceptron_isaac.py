"""LeRobot-native joint text/FAST and flow-matching loss for ISAAC.

The implementation intentionally operates on the policy's local TensorStream
types and has no Genesis loss imports.  Flow randomness can be supplied as
explicit tensors so Genesis-oracle fixtures replay across implementations
without depending on RNG consumption order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch import Tensor

from .tensor_stream import FLOW_ACTION_TEXT_TYPE, SyntheticType, TensorStream, TextType
from .tensor_stream_utils import (
    ROLE_TO_IDX,
    action_event_provenance_key,
    build_action_context_mask,
    resolve_tensor_stream_metadata,
    set_data,
)

ISAAC_FLOW_TIMESTEPS_KEY = "perceptron_isaac_flow_timesteps"
ISAAC_FLOW_NOISE_KEY = "perceptron_isaac_flow_noise"
ISAAC_FLOW_LOSS_MASK_KEY = "perceptron_isaac_flow_loss_mask"

_TEXT_LOSS_TYPES = (
    TextType.text,
    TextType.action,
    TextType.control,
    TextType.timestamp,
    SyntheticType.audio_transcript,
    SyntheticType.caption,
    SyntheticType.segmentation,
)


@dataclass(frozen=True)
class IsaacLossOutput:
    loss: Tensor
    per_sample_loss: Tensor
    metrics: dict[str, Tensor]


@dataclass(frozen=True)
class _FlowBatch:
    targets: Tensor
    full_mask: Tensor
    horizon_mask: Tensor
    batch_indices: list[int]
    action_start_indices: list[int]
    provenance_keys: list[tuple[object, ...] | None]


def _valid_next_token_labels(tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Match Genesis valid_n_offset_tensor(..., n=1) without importing it."""
    batch_size, seq_len = tokens.shape
    prefix = torch.cumsum(valid, dim=1)
    targets = prefix + 1
    indices = torch.searchsorted(prefix, targets, right=False)
    clamped = indices.clamp(max=seq_len - 1)
    matches = (indices < seq_len) & (prefix.gather(1, clamped) == targets) & valid
    labels = torch.zeros_like(tokens)
    batch_idx, source_idx = torch.nonzero(matches, as_tuple=True)
    labels[batch_idx, source_idx] = tokens[batch_idx, indices[batch_idx, source_idx]]
    return labels, matches


def _text_loss(
    output: Any,
    stream: TensorStream,
    *,
    exclude_non_agent_roles: bool,
    softmax_auxiliary_loss_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    activations = output.final_activations
    metadata = resolve_tensor_stream_metadata(stream, None)
    roles = ("agent", "assistant", None) if exclude_non_agent_roles else tuple(ROLE_TO_IDX)
    tokens, valid_tokens = set_data(stream, _TEXT_LOSS_TYPES, roles=roles, metadata=metadata)
    labels, valid_labels = _valid_next_token_labels(tokens, valid_tokens)
    labels = labels[:, : activations.shape[1]]
    valid_labels = valid_labels[:, : activations.shape[1]]
    classifier = output.final_embedding.weight

    batch_size = activations.shape[0]
    numerators = activations.new_zeros(batch_size, dtype=torch.float32)
    counts = activations.new_zeros(batch_size, dtype=torch.float32)
    for batch_idx in range(0 if stream.collective_only else batch_size):
        mask = valid_labels[batch_idx]
        count = mask.sum()
        if not bool(count):
            continue
        logits = activations[batch_idx, mask] @ classifier.t()
        logits_fp32 = logits.float()
        token_loss = functional.cross_entropy(logits_fp32, labels[batch_idx, mask], reduction="none")
        if softmax_auxiliary_loss_scale:
            token_loss = (
                token_loss + softmax_auxiliary_loss_scale * torch.logsumexp(logits_fp32, dim=-1).square()
            )
        numerators[batch_idx] = token_loss.sum()
        counts[batch_idx] = count
    if not bool(counts.sum()):
        # Empty local text still connects the backbone and classifier to DDP, without
        # reducing the entire vocabulary weight tensor on ordinary nonempty batches.
        numerators = (
            numerators + activations.float().sum(dim=(1, 2)) * 0 + classifier.reshape(-1)[0].float() * 0
        )
    per_sample = numerators / counts.clamp_min(1.0)
    return numerators, counts, per_sample


def _text_label_count(
    stream: TensorStream,
    *,
    exclude_non_agent_roles: bool,
) -> Tensor:
    if stream.collective_only:
        return torch.zeros((), dtype=torch.int64)
    metadata = resolve_tensor_stream_metadata(stream, None)
    roles = ("agent", "assistant", None) if exclude_non_agent_roles else tuple(ROLE_TO_IDX)
    tokens, valid_tokens = set_data(stream, _TEXT_LOSS_TYPES, roles=roles, metadata=metadata)
    _, valid_labels = _valid_next_token_labels(tokens, valid_tokens)
    return valid_labels[:, : stream.shape[1] - 1].sum(dtype=torch.int64)


def _collect_flow_batch(
    stream: TensorStream,
    *,
    action_dim: int,
    action_horizon: int,
    mask_padded_action_rows: bool,
    device: torch.device,
    exclude_non_agent_roles: bool = True,
) -> _FlowBatch:
    # Mirror Genesis's _allowed_role_indices: honor the flag for the flow objective too, so
    # exclude_non_agent_roles=False admits every role rather than silently dropping markers.
    allowed_roles = (
        {ROLE_TO_IDX["agent"], ROLE_TO_IDX["assistant"], ROLE_TO_IDX[None]}
        if exclude_non_agent_roles
        else set(ROLE_TO_IDX.values())
    )
    records: list[tuple[int, int, Any]] = []
    for batch_idx, sample in enumerate(stream.streams):
        offset = 0
        for event in sample.events:
            event_start = offset
            offset += event.num_tokens()
            if event.type != FLOW_ACTION_TEXT_TYPE or ROLE_TO_IDX.get(event.role, -1) not in allowed_roles:
                continue
            raw_target = (event.tags or {}).get("action_target")
            if raw_target is None:
                raise ValueError("ISAAC flow marker is missing tags['action_target'].")
            records.append((batch_idx, event_start, event))
    if not records:
        if stream.collective_only:
            targets = torch.zeros(0, action_horizon, action_dim, device=device, dtype=torch.float32)
            return _FlowBatch(targets, targets.clone(), targets.any(dim=2), [], [], [])
        raise ValueError("ISAAC training batch contains no flow action markers.")
    if len(records) != len(stream.streams) or {record[0] for record in records} != set(
        range(len(stream.streams))
    ):
        raise ValueError("ISAAC native training requires exactly one flow action marker per sample.")

    targets = torch.zeros(len(records), action_horizon, action_dim, device=device, dtype=torch.float32)
    full_mask = torch.zeros_like(targets)
    batch_indices: list[int] = []
    action_start_indices: list[int] = []
    provenance_keys: list[tuple[object, ...] | None] = []
    for chunk_idx, (batch_idx, event_start, event) in enumerate(records):
        target = torch.as_tensor(event.tags["action_target"], device=device, dtype=torch.float32)
        if not bool(torch.isfinite(target).all()):
            raise ValueError("ISAAC action target must contain only finite values.")
        if target.ndim != 2:
            raise ValueError(f"ISAAC action target must be [H,D], got {tuple(target.shape)}.")
        horizon, real_dim = target.shape
        if horizon < 1 or horizon > action_horizon or real_dim < 1 or real_dim > action_dim:
            raise ValueError(
                f"ISAAC action target shape {tuple(target.shape)} exceeds expert {(action_horizon, action_dim)}."
            )
        row_mask = torch.ones(horizon, device=device, dtype=torch.bool)
        if mask_padded_action_rows and event.tags.get("action_is_pad") is not None:
            raw_pad = event.tags["action_is_pad"]
            if (
                not isinstance(raw_pad, list | tuple)
                or len(raw_pad) != horizon
                or any(not isinstance(value, bool) for value in raw_pad)
            ):
                raise ValueError("ISAAC action_is_pad must be a boolean list matching the action horizon.")
            row_mask = ~torch.as_tensor(raw_pad, device=device, dtype=torch.bool)
        if not bool(row_mask.any()):
            raise ValueError("ISAAC flow target has no unpadded action rows.")
        targets[chunk_idx, :horizon, :real_dim] = target
        full_mask[chunk_idx, :horizon, :real_dim] = row_mask[:, None]
        batch_indices.append(batch_idx)
        action_start_indices.append(event_start)
        provenance_keys.append(action_event_provenance_key(event))
    targets = targets * full_mask
    horizon_mask = full_mask.any(dim=2)
    return _FlowBatch(
        targets=targets,
        full_mask=full_mask,
        horizon_mask=horizon_mask,
        batch_indices=batch_indices,
        action_start_indices=action_start_indices,
        provenance_keys=provenance_keys,
    )


def _coerce_flow_timesteps(
    *, flow: _FlowBatch, sample_count: int, timesteps: Tensor
) -> tuple[Tensor, Tensor]:
    """Validate flow timestep geometry and values without drawing randomness or changing inputs."""
    chunk_count, horizon, _ = flow.targets.shape
    device = flow.targets.device
    timesteps = timesteps.to(device=device, dtype=torch.float32)
    if sample_count == 1 and timesteps.shape == (chunk_count,):
        timesteps = timesteps.unsqueeze(0)
    expected_scalar = (sample_count, chunk_count)
    expected_grid = (sample_count, chunk_count, horizon, 1)
    if tuple(timesteps.shape) == expected_scalar:
        expert_timesteps = timesteps
        timestep_grid = timesteps[:, :, None, None].expand(-1, -1, horizon, 1)
    elif tuple(timesteps.shape) == expected_grid:
        timestep_grid = timesteps
        valid_rows = flow.horizon_mask[None, :, :, None].expand(sample_count, -1, -1, -1)
        first_valid = flow.horizon_mask.to(torch.int64).argmax(dim=1)
        expert_timesteps = timestep_grid[:, torch.arange(chunk_count, device=device), first_valid, 0]
        if not torch.allclose(
            timestep_grid.masked_select(valid_rows),
            expert_timesteps[:, :, None, None].expand(-1, -1, horizon, 1).masked_select(valid_rows),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "ISAAC flow timestep grids must be constant over valid horizon rows "
                "while RTC and dual-timestep training are disabled."
            )
    else:
        raise ValueError(
            "ISAAC flow timesteps must have shape "
            f"{expected_scalar} or {expected_grid}, got {tuple(timesteps.shape)}."
        )
    if not bool(torch.isfinite(timestep_grid).all()) or not bool(
        ((timestep_grid >= 0.0) & (timestep_grid <= 1.0)).all()
    ):
        raise ValueError("ISAAC flow timesteps must be finite and lie in [0, 1].")
    return timestep_grid, expert_timesteps


def _coerce_flow_noise(*, flow: _FlowBatch, sample_count: int, noise: Tensor) -> Tensor:
    """Validate flow noise geometry and values without drawing randomness or changing inputs."""
    noise = noise.to(device=flow.targets.device, dtype=torch.float32)
    if sample_count == 1 and noise.shape == flow.targets.shape:
        noise = noise.unsqueeze(0)
    expected_noise = (sample_count, *flow.targets.shape)
    if tuple(noise.shape) != expected_noise:
        raise ValueError(f"ISAAC flow noise must have shape {expected_noise}, got {tuple(noise.shape)}.")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("ISAAC flow noise must contain only finite values.")
    return noise


def _coerce_flow_randomness(
    *,
    expert: Any,
    flow: _FlowBatch,
    sample_count: int,
    timesteps: Tensor | None,
    noise: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    chunk_count, horizon, action_dim = flow.targets.shape
    device = flow.targets.device
    if timesteps is None:
        sampled = [
            expert.sample_timesteps(chunk_count, device=device, dtype=torch.float32)
            for _ in range(sample_count)
        ]
        timesteps = torch.stack(sampled, dim=0)
    timestep_grid, expert_timesteps = _coerce_flow_timesteps(
        flow=flow, sample_count=sample_count, timesteps=timesteps
    )
    if noise is None:
        noise = torch.randn(
            sample_count,
            chunk_count,
            horizon,
            action_dim,
            device=device,
            dtype=torch.float32,
        )
    noise = _coerce_flow_noise(flow=flow, sample_count=sample_count, noise=noise)
    return timestep_grid, expert_timesteps, noise * flow.full_mask.unsqueeze(0)


def _coerce_flow_loss_mask(
    *,
    flow: _FlowBatch,
    sample_count: int,
    loss_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    base_mask = flow.full_mask.unsqueeze(0).expand(sample_count, -1, -1, -1)
    if loss_mask is None:
        mask = base_mask
    else:
        mask = loss_mask.to(device=flow.targets.device, dtype=torch.float32)
        if sample_count == 1 and tuple(mask.shape) == tuple(flow.targets.shape):
            mask = mask.unsqueeze(0)
        expected = (sample_count, *flow.targets.shape)
        if tuple(mask.shape) != expected:
            raise ValueError(f"ISAAC flow loss mask must have shape {expected}, got {tuple(mask.shape)}.")
        if not bool(torch.isfinite(mask).all()) or not bool(((mask >= 0.0) & (mask <= 1.0)).all()):
            raise ValueError("ISAAC flow loss mask must contain finite values in [0, 1].")
        if bool((mask * (1.0 - base_mask)).ne(0).any()):
            raise ValueError("ISAAC flow loss mask cannot enable padded horizon rows or action dimensions.")
    denominators = mask.sum(dim=(2, 3))
    if bool((denominators <= 0).any()):
        raise ValueError("ISAAC flow loss mask must select at least one element per draw and action chunk.")
    return mask, denominators


def perceptron_isaac_objective_counts(
    stream: TensorStream,
    *,
    loss_plan: str,
    action_dim: int,
    action_horizon: int,
    train_samples_per_chunk: int,
    flow_mask_padded_action_rows: bool,
    exclude_non_agent_roles: bool,
    flow_loss_mask: Tensor | None = None,
    flow_timesteps: Tensor | None = None,
    flow_noise: Tensor | None = None,
) -> dict[str, int]:
    """Count local objectives and validate supplied flow inputs without sampling or a model forward."""
    if loss_plan not in {"text_ntp_fast_flow_action", "flow_matching_action_prediction"}:
        raise ValueError(f"Unsupported ISAAC loss plan {loss_plan!r}.")
    if train_samples_per_chunk < 1:
        raise ValueError("train_samples_per_chunk must be >= 1.")
    flow = _collect_flow_batch(
        stream,
        action_dim=action_dim,
        action_horizon=action_horizon,
        mask_padded_action_rows=flow_mask_padded_action_rows,
        device=torch.device("cpu"),
        exclude_non_agent_roles=exclude_non_agent_roles,
    )
    _coerce_flow_loss_mask(
        flow=flow,
        sample_count=train_samples_per_chunk,
        loss_mask=flow_loss_mask,
    )
    if flow_timesteps is not None:
        _coerce_flow_timesteps(flow=flow, sample_count=train_samples_per_chunk, timesteps=flow_timesteps)
    if flow_noise is not None:
        _coerce_flow_noise(flow=flow, sample_count=train_samples_per_chunk, noise=flow_noise)
    # Each draw is normalized within its chunk; the update denominator counts chunks only.
    denominators = {"flow": flow.targets.shape[0]}
    if loss_plan == "text_ntp_fast_flow_action":
        text_count = _text_label_count(
            stream,
            exclude_non_agent_roles=exclude_non_agent_roles,
        )
        denominators["text"] = int(text_count.item())
    return denominators


def perceptron_isaac_loss_denominators(
    stream: TensorStream,
    *,
    loss_plan: str,
    action_dim: int,
    action_horizon: int,
    train_samples_per_chunk: int,
    flow_mask_padded_action_rows: bool,
    exclude_non_agent_roles: bool,
    flow_loss_mask: Tensor | None = None,
    flow_timesteps: Tensor | None = None,
    flow_noise: Tensor | None = None,
) -> dict[str, Tensor]:
    """Keep the existing tensor-denominator API; stock preparation uses integer counts."""
    counts = perceptron_isaac_objective_counts(
        stream,
        loss_plan=loss_plan,
        action_dim=action_dim,
        action_horizon=action_horizon,
        train_samples_per_chunk=train_samples_per_chunk,
        flow_mask_padded_action_rows=flow_mask_padded_action_rows,
        exclude_non_agent_roles=exclude_non_agent_roles,
        flow_loss_mask=flow_loss_mask,
        flow_timesteps=flow_timesteps,
        flow_noise=flow_noise,
    )
    return {name: torch.tensor(count, dtype=torch.float32) for name, count in counts.items()}


def _flow_loss(
    output: Any,
    stream: TensorStream,
    *,
    train_samples_per_chunk: int,
    detach_vlm_activations: bool,
    mask_padded_action_rows: bool,
    timesteps: Tensor | None,
    noise: Tensor | None,
    loss_mask: Tensor | None,
    exclude_non_agent_roles: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    expert = output.flow_matching_expert
    if expert is None:
        raise ValueError("ISAAC model output has no flow-matching expert.")
    if train_samples_per_chunk < 1:
        raise ValueError("train_samples_per_chunk must be >= 1.")
    flow = _collect_flow_batch(
        stream,
        action_dim=int(expert.args.action_dim),
        action_horizon=int(expert.args.action_horizon),
        mask_padded_action_rows=mask_padded_action_rows,
        device=output.final_activations.device,
        exclude_non_agent_roles=exclude_non_agent_roles,
    )
    if stream.collective_only:
        # A finite context traverses the expert without inventing supervision or RNG draws.
        activations = output.final_activations
        draws = train_samples_per_chunk
        dummy_actions = activations.new_zeros(
            draws, 1, int(expert.args.action_horizon), int(expert.args.action_dim)
        )
        dummy_tau = activations.new_zeros(draws, 1, dtype=torch.float32)
        predicted = expert(
            activations,
            torch.ones(1, activations.shape[1], device=activations.device, dtype=torch.bool),
            dummy_actions[0] if draws == 1 else dummy_actions,
            dummy_tau[0] if draws == 1 else dummy_tau,
            action_mask=torch.ones(
                1, int(expert.args.action_horizon), device=activations.device, dtype=torch.bool
            ),
        )
        if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(activations).all()):
            raise ValueError("ISAAC collective-only forward must be finite.")
        zero = predicted.float().sum() * 0 + activations.float().sum() * 0
        return zero, zero.detach(), zero.reshape(1)[:0], zero.detach(), zero.detach()
    tau_grid, expert_tau, eps = _coerce_flow_randomness(
        expert=expert,
        flow=flow,
        sample_count=train_samples_per_chunk,
        timesteps=timesteps,
        noise=noise,
    )
    effective_mask, per_draw_denominators = _coerce_flow_loss_mask(
        flow=flow,
        sample_count=train_samples_per_chunk,
        loss_mask=loss_mask,
    )
    targets = flow.targets.unsqueeze(0)
    x_tau = tau_grid * eps + (1.0 - tau_grid) * targets
    clean_at_0 = expert.args.clean_at_0
    velocity_target = eps - targets if clean_at_0 else targets - eps

    activations = output.final_activations.detach() if detach_vlm_activations else output.final_activations
    context_mask, _ = build_action_context_mask(
        stream,
        action_batch_indices=flow.batch_indices,
        action_start_indices=flow.action_start_indices,
        action_provenance_keys=flow.provenance_keys,
        l_model=activations.shape[1],
        device=activations.device,
    )
    if not bool(context_mask.any(dim=1).all()):
        raise ValueError("ISAAC flow target has an empty VLM context.")
    expert_input = x_tau[0] if train_samples_per_chunk == 1 else x_tau
    expert_tau = expert_tau[0] if train_samples_per_chunk == 1 else expert_tau
    predicted = expert(
        activations,
        context_mask,
        expert_input.to(dtype=activations.dtype),
        expert_tau,
        action_mask=flow.horizon_mask,
    ).float()
    if train_samples_per_chunk == 1:
        predicted = predicted.unsqueeze(0)
    squared_error = (predicted - velocity_target).square() * effective_mask
    per_sample = (squared_error.sum(dim=(2, 3)) / per_draw_denominators).mean(dim=0)
    valid_tau = flow.horizon_mask[None, :, :, None].expand(train_samples_per_chunk, -1, -1, -1)
    mean_timestep = tau_grid.masked_select(valid_tau).mean()
    return (
        per_sample.sum(),
        per_sample.new_tensor(float(per_sample.numel())),
        per_sample,
        mean_timestep,
        effective_mask.sum(dtype=torch.float32),
    )


def _global_sum_detached(value: Tensor) -> Tensor:
    result = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _global_objective_mean(
    numerator: Tensor,
    denominator: Tensor,
    *,
    global_denominator: Tensor | None = None,
    accumulation_scale: int = 1,
    allow_empty: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize local gradients by one global count spanning the complete optimizer update."""
    if accumulation_scale < 1:
        raise ValueError("ISAAC loss accumulation_scale must be >= 1.")
    denominator = denominator.to(device=numerator.device, dtype=torch.float32)
    global_numerator = _global_sum_detached(numerator)
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    if global_denominator is None:
        global_denominator = _global_sum_detached(denominator)
        metric_scale = 1.0
    else:
        global_denominator = torch.as_tensor(
            global_denominator,
            device=numerator.device,
            dtype=torch.float32,
        )
        metric_scale = float(accumulation_scale)
    if global_denominator.numel() != 1 or not bool(torch.isfinite(global_denominator)):
        raise ValueError("ISAAC global objective denominator must be one finite scalar.")
    if allow_empty and bool(global_denominator == 0):
        if bool(_global_sum_detached(denominator) > 0):
            raise ValueError("ISAAC zero global objective denominator cannot hide real local supervision.")
        return numerator * 0, global_numerator * 0, global_denominator
    if not bool(global_denominator > 0):
        raise ValueError("ISAAC global objective denominator must be positive.")
    backward = numerator * (world_size * metric_scale / global_denominator)
    value = global_numerator * metric_scale / global_denominator
    # The public scalar is global; DDP still averages only rank-local derivatives.
    return backward + (value - backward.detach()), value, global_denominator


def perceptron_isaac_training_loss(
    output: Any,
    stream: TensorStream,
    *,
    loss_plan: str = "text_ntp_fast_flow_action",
    exclude_non_agent_roles: bool = True,
    flow_matching_detach_vlm_activations: bool = False,
    train_samples_per_chunk: int = 1,
    flow_mask_padded_action_rows: bool = True,
    softmax_auxiliary_loss_scale: float = 0.0,
    moe_load_balancing_loss_coeff: float = 0.02,
    moe_z_loss_coeff: float = 0.001,
    flow_timesteps: Tensor | None = None,
    flow_noise: Tensor | None = None,
    flow_loss_mask: Tensor | None = None,
    global_denominators: dict[str, Tensor] | None = None,
    accumulation_scale: int = 1,
    reduction: str = "mean",
) -> IsaacLossOutput:
    """Compute ISAAC's objective-normalized native training loss."""
    if loss_plan not in {"text_ntp_fast_flow_action", "flow_matching_action_prediction"}:
        raise ValueError(f"Unsupported ISAAC loss plan {loss_plan!r}.")
    if reduction not in {"mean", "none"}:
        raise ValueError("ISAAC loss reduction must be 'mean' or 'none'.")
    for name, coefficient in (
        ("softmax_auxiliary_loss_scale", softmax_auxiliary_loss_scale),
        ("moe_load_balancing_loss_coeff", moe_load_balancing_loss_coeff),
        ("moe_z_loss_coeff", moe_z_loss_coeff),
    ):
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if accumulation_scale < 1:
        raise ValueError("ISAAC loss accumulation_scale must be >= 1.")
    expected_denominators = {"flow", "text"} if loss_plan == "text_ntp_fast_flow_action" else {"flow"}
    if global_denominators is not None and set(global_denominators) != expected_denominators:
        raise ValueError(f"ISAAC global denominators must define {sorted(expected_denominators)} exactly.")

    flow_num, flow_count, flow_per_sample, mean_timestep, element_count = _flow_loss(
        output,
        stream,
        train_samples_per_chunk=train_samples_per_chunk,
        detach_vlm_activations=flow_matching_detach_vlm_activations,
        mask_padded_action_rows=flow_mask_padded_action_rows,
        timesteps=flow_timesteps,
        noise=flow_noise,
        loss_mask=flow_loss_mask,
        exclude_non_agent_roles=exclude_non_agent_roles,
    )
    flow_backward, flow_global, flow_global_count = _global_objective_mean(
        flow_num,
        flow_count,
        global_denominator=None if global_denominators is None else global_denominators["flow"],
        accumulation_scale=accumulation_scale,
        allow_empty=True,
    )
    flow_chunk_count = _global_sum_detached(flow_count)
    text_global_count = flow_global_count.new_zeros(())
    metrics = {
        "flow_loss": flow_global,
        "flow_chunk_count": flow_chunk_count,
        "flow_element_count": _global_sum_detached(element_count),
        "flow_mean_timestep": mean_timestep.float().detach(),
    }
    total_backward = flow_backward
    per_sample = flow_per_sample
    if loss_plan == "text_ntp_fast_flow_action":
        text_num_by_sample, text_count_by_sample, text_per_sample = _text_loss(
            output,
            stream,
            exclude_non_agent_roles=exclude_non_agent_roles,
            softmax_auxiliary_loss_scale=softmax_auxiliary_loss_scale,
        )
        text_num = text_num_by_sample.sum()
        text_count = text_count_by_sample.sum()
        text_backward, text_global, text_global_count = _global_objective_mean(
            text_num,
            text_count,
            global_denominator=None if global_denominators is None else global_denominators["text"],
            accumulation_scale=accumulation_scale,
            allow_empty=True,
        )
        total_backward = total_backward + text_backward
        per_sample = per_sample + text_per_sample
        metrics.update({"text_loss": text_global, "text_token_count": _global_sum_detached(text_count)})
    else:
        metrics.update(
            {
                "text_loss": output.final_activations.new_zeros((), dtype=torch.float32),
                "text_token_count": output.final_activations.new_zeros((), dtype=torch.float32),
            }
        )
    if not bool(flow_global_count + text_global_count > 0):
        raise ValueError("ISAAC primary objective window is globally empty.")
    auxiliary = total_backward.new_zeros(())
    for key, coefficient in (
        ("load_balancing_loss", moe_load_balancing_loss_coeff),
        ("router_z_loss", moe_z_loss_coeff),
    ):
        # Do not attach disabled branches: even zero times an overflowing loss can poison backward.
        if coefficient == 0:
            continue
        layers = output.additional_info.get(key, ())
        if layers:
            layer_mean = torch.stack(layers).mean()
            metrics[key] = layer_mean.detach()
            auxiliary = auxiliary + coefficient * layer_mean
    total_backward = total_backward + auxiliary
    per_sample = per_sample + auxiliary
    loss = total_backward if reduction == "mean" else per_sample
    metrics["loss"] = metrics["flow_loss"] + metrics["text_loss"] + auxiliary.detach()
    return IsaacLossOutput(loss=loss, per_sample_loss=per_sample, metrics=metrics)
