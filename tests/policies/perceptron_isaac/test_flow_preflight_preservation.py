"""Shared count/forward randomness contracts, without model allocation or sampling in counts."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    _coerce_flow_randomness,
    _collect_flow_batch,
    perceptron_isaac_loss_denominators,
    perceptron_isaac_objective_counts,
)
from tests.policies.perceptron_isaac._fixtures import flow_stream


def test_forward_randomness_keeps_timestep_then_noise_draw_order() -> None:
    class SamplingExpert:
        def sample_timesteps(self, count: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
            return torch.rand(count, device=device, dtype=dtype)

    flow = _collect_flow_batch(
        flow_stream(action_is_pad=[False, True]),
        action_dim=2,
        action_horizon=2,
        mask_padded_action_rows=True,
        device=torch.device("cpu"),
    )
    live = torch.get_rng_state()
    try:
        expected_timesteps = torch.stack([torch.rand(1) for _ in range(3)])
        expected_noise = torch.randn(3, 1, 2, 2) * flow.full_mask.unsqueeze(0)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(live)
        grid, timesteps, noise = _coerce_flow_randomness(
            expert=SamplingExpert(), flow=flow, sample_count=3, timesteps=None, noise=None
        )
        assert grid.dtype == timesteps.dtype == noise.dtype == torch.float32
        assert torch.equal(timesteps, expected_timesteps)
        assert torch.equal(grid, expected_timesteps[:, :, None, None].expand(-1, -1, 2, 1))
        assert torch.equal(noise, expected_noise)
        assert torch.equal(torch.get_rng_state(), expected_rng)
    finally:
        torch.set_rng_state(live)


@pytest.mark.parametrize("supplied", ["neither", "timesteps", "noise", "both"])
def test_counts_validate_only_supplied_randomness_and_keep_padded_grid_rows(supplied: str) -> None:
    # Padded timesteps need not equal valid rows, but still must lie in [0, 1].
    timesteps = torch.tensor([[[[0.25], [0.75]]]], dtype=torch.float64)
    noise = torch.ones(1, 2, 2, dtype=torch.float64)  # Supported single-draw shorthand.
    original_timesteps, original_noise = timesteps.clone(), noise.clone()
    flow_timesteps = timesteps if supplied in {"timesteps", "both"} else None
    flow_noise = noise if supplied in {"noise", "both"} else None
    live = torch.get_rng_state()
    assert perceptron_isaac_objective_counts(
        flow_stream(action_is_pad=[False, True]),
        loss_plan="flow_matching_action_prediction",
        action_dim=2,
        action_horizon=2,
        train_samples_per_chunk=1,
        flow_mask_padded_action_rows=True,
        exclude_non_agent_roles=True,
        flow_timesteps=flow_timesteps,
        flow_noise=flow_noise,
    ) == {"flow": 1}
    denominators = perceptron_isaac_loss_denominators(
        flow_stream(action_is_pad=[False, True]),
        loss_plan="flow_matching_action_prediction",
        action_dim=2,
        action_horizon=2,
        train_samples_per_chunk=1,
        flow_mask_padded_action_rows=True,
        exclude_non_agent_roles=True,
        flow_timesteps=flow_timesteps,
        flow_noise=flow_noise,
    )
    assert denominators["flow"].item() == 1
    assert torch.equal(torch.get_rng_state(), live)
    assert timesteps.dtype == noise.dtype == torch.float64
    assert torch.equal(timesteps, original_timesteps) and torch.equal(noise, original_noise)


@pytest.mark.parametrize("invalid", ["grid", "noise", "timestep-range"])
def test_direct_count_and_denominator_apis_refuse_supplied_randomness(invalid: str) -> None:
    timesteps = torch.tensor([[[[0.25], [0.75]]]]) if invalid == "grid" else torch.tensor([[0.5]])
    noise = torch.zeros(1, 1, 2, 3 if invalid == "noise" else 2)
    if invalid == "timestep-range":
        timesteps.fill_(2.0)
    for count in (perceptron_isaac_objective_counts, perceptron_isaac_loss_denominators):
        with pytest.raises(ValueError, match="ISAAC flow"):
            count(
                flow_stream(),
                loss_plan="flow_matching_action_prediction",
                action_dim=2,
                action_horizon=2,
                train_samples_per_chunk=1,
                flow_mask_padded_action_rows=True,
                exclude_non_agent_roles=True,
                flow_timesteps=timesteps,
                flow_noise=noise,
            )
