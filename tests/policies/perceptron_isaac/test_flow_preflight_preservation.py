"""Live Flow randomness validation preserves supplied fixtures and draw order."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    _coerce_flow_noise,
    _coerce_flow_randomness,
    _coerce_flow_timesteps,
    _collect_flow_batch,
    perceptron_isaac_training_loss,
)
from tests.policies.perceptron_isaac._fixtures import ZeroExpert, flow_output, flow_stream


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


@pytest.mark.parametrize("supplied", ["timesteps", "noise", "both"])
def test_flow_fixture_validation_is_rng_free_and_preserves_inputs(supplied: str) -> None:
    flow = _collect_flow_batch(
        flow_stream(action_is_pad=[False, True]),
        action_dim=2,
        action_horizon=2,
        mask_padded_action_rows=True,
        device=torch.device("cpu"),
    )
    timesteps = torch.tensor([[[[0.25], [0.75]]]], dtype=torch.float64)
    noise = torch.ones(1, 2, 2, dtype=torch.float64)
    original_timesteps, original_noise = timesteps.clone(), noise.clone()
    live = torch.get_rng_state()
    if supplied in {"timesteps", "both"}:
        grid, tau = _coerce_flow_timesteps(flow=flow, sample_count=1, timesteps=timesteps)
        assert grid.dtype == tau.dtype == torch.float32
    if supplied in {"noise", "both"}:
        validated_noise = _coerce_flow_noise(flow=flow, sample_count=1, noise=noise)
        assert validated_noise.dtype == torch.float32
    assert torch.equal(torch.get_rng_state(), live)
    assert timesteps.dtype == noise.dtype == torch.float64
    assert torch.equal(timesteps, original_timesteps) and torch.equal(noise, original_noise)


@pytest.mark.parametrize("invalid", ["grid", "noise", "timestep-range"])
def test_live_loss_refuses_invalid_flow_randomness_before_sampling(invalid: str) -> None:
    timesteps = torch.tensor([[[[0.25], [0.75]]]]) if invalid == "grid" else torch.tensor([[0.5]])
    noise = torch.zeros(1, 1, 2, 3 if invalid == "noise" else 2)
    if invalid == "timestep-range":
        timesteps.fill_(2.0)
    stream = flow_stream()
    output = flow_output(stream, ZeroExpert())
    live = torch.get_rng_state()
    with pytest.raises(ValueError, match="ISAAC flow"):
        perceptron_isaac_training_loss(
            output,
            stream,
            loss_plan="flow_matching_action_prediction",
            flow_timesteps=timesteps,
            flow_noise=noise,
        )
    assert torch.equal(torch.get_rng_state(), live)
