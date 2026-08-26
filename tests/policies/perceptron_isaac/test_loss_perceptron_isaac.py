from __future__ import annotations

import pytest
import torch

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    _global_objective_mean,
    perceptron_isaac_loss_denominators,
    perceptron_isaac_training_loss,
)
from tests.policies.perceptron_isaac._fixtures import (
    ZeroExpert,
    flow_output,
    flow_stream,
)


def test_flow_loss_uses_injected_noise_and_timestep_tensors():
    stream = flow_stream()
    expert = ZeroExpert()
    output = flow_output(stream, expert)
    noise = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    timesteps = torch.tensor([[0.25]])

    result = perceptron_isaac_training_loss(
        output,
        stream,
        loss_plan="flow_matching_action_prediction",
        train_samples_per_chunk=1,
        flow_timesteps=timesteps,
        flow_noise=noise,
    )

    # clean-at-1 velocity target = action - noise = [[1,1],[1,1]].
    torch.testing.assert_close(result.loss, torch.tensor(1.0))
    torch.testing.assert_close(result.per_sample_loss, torch.tensor([1.0]))
    result.loss.backward()
    assert expert.scale.grad is not None


def test_flow_loss_accepts_canonical_timestep_grid_and_weight_mask():
    stream = flow_stream()
    expert = ZeroExpert()
    output = flow_output(stream, expert)
    noise = torch.zeros(1, 1, 2, 2)
    timesteps = torch.full((1, 1, 2, 1), 0.25)
    loss_mask = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])

    result = perceptron_isaac_training_loss(
        output,
        stream,
        loss_plan="flow_matching_action_prediction",
        train_samples_per_chunk=1,
        flow_timesteps=timesteps,
        flow_noise=noise,
        flow_loss_mask=loss_mask,
    )

    # clean-at-1 and zero noise/expert: weighted mean of 1^2 and 3^2.
    torch.testing.assert_close(result.loss, torch.tensor(11.0 / 3.0))
    torch.testing.assert_close(result.metrics["flow_mean_timestep"], torch.tensor(0.25))


def test_flow_timestep_grid_requires_scalar_tau_until_dual_timestep_is_implemented():
    stream = flow_stream()
    output = flow_output(stream, ZeroExpert())

    with pytest.raises(ValueError, match="constant over valid horizon"):
        perceptron_isaac_training_loss(
            output,
            stream,
            loss_plan="flow_matching_action_prediction",
            flow_timesteps=torch.tensor([[[[0.25], [0.75]]]]),
            flow_noise=torch.zeros(1, 1, 2, 2),
        )


@pytest.mark.parametrize(
    ("loss_mask", "message"),
    [
        (torch.tensor([[[[1.0, 1.0], [0.0, 2.0]]]]), "finite values in"),
        (torch.zeros(1, 1, 2, 2), "at least one element"),
    ],
)
def test_flow_loss_mask_fails_closed(loss_mask, message):
    stream = flow_stream()
    output = flow_output(stream, ZeroExpert())

    with pytest.raises(ValueError, match=message):
        perceptron_isaac_training_loss(
            output,
            stream,
            loss_plan="flow_matching_action_prediction",
            flow_timesteps=torch.tensor([[[[0.5], [0.5]]]]),
            flow_noise=torch.zeros(1, 1, 2, 2),
            flow_loss_mask=loss_mask,
        )


def test_flow_loss_masks_terminal_padded_rows():
    stream = flow_stream(action_is_pad=[False, True])
    expert = ZeroExpert()
    output = flow_output(stream, expert)
    noise = torch.tensor([[[[0.0, 0.0], [100.0, 100.0]]]])

    result = perceptron_isaac_training_loss(
        output,
        stream,
        loss_plan="flow_matching_action_prediction",
        train_samples_per_chunk=1,
        flow_timesteps=torch.tensor([[0.5]]),
        flow_noise=noise,
    )

    # Only row zero participates: mean([1^2, 2^2]) = 2.5.
    torch.testing.assert_close(result.loss, torch.tensor(2.5))


def test_joint_loss_sums_token_normalized_ce_and_chunk_normalized_flow():
    stream = flow_stream()
    expert = ZeroExpert()
    output = flow_output(stream, expert)
    result = perceptron_isaac_training_loss(
        output,
        stream,
        train_samples_per_chunk=1,
        flow_timesteps=torch.tensor([[0.5]]),
        flow_noise=torch.zeros(1, 1, 2, 2),
    )

    # Zero logits -> CE=log(vocab); zero expert -> mean(action^2)=7.5.
    torch.testing.assert_close(result.loss, torch.log(torch.tensor(32.0)) + 7.5)
    assert result.metrics["text_token_count"] > 0
    assert result.metrics["flow_chunk_count"] == 1


def test_loss_denominators_count_valid_elements_across_all_flow_draws():
    stream = flow_stream(action_is_pad=[False, True])

    denominators = perceptron_isaac_loss_denominators(
        stream,
        loss_plan="flow_matching_action_prediction",
        action_dim=2,
        action_horizon=2,
        train_samples_per_chunk=8,
        flow_mask_padded_action_rows=True,
        exclude_non_agent_roles=True,
    )

    # One valid action row x two action dimensions x eight flow draws.
    torch.testing.assert_close(denominators["flow"], torch.tensor(16.0))


def test_loss_denominators_include_joint_text_tokens():
    denominators = perceptron_isaac_loss_denominators(
        flow_stream(),
        loss_plan="text_ntp_fast_flow_action",
        action_dim=2,
        action_horizon=2,
        train_samples_per_chunk=1,
        flow_mask_padded_action_rows=True,
        exclude_non_agent_roles=True,
    )

    assert set(denominators) == {"flow", "text"}
    torch.testing.assert_close(denominators["flow"], torch.tensor(4.0))
    assert denominators["text"] > 0


def test_update_wide_denominator_matches_one_global_objective_under_accumulation():
    first_numerator = torch.tensor(4.0, requires_grad=True)
    second_numerator = torch.tensor(18.0, requires_grad=True)
    update_denominator = torch.tensor(8.0)

    first_loss, first_metric, _ = _global_objective_mean(
        first_numerator,
        torch.tensor(2.0),
        global_denominator=update_denominator,
        accumulation_scale=2,
    )
    second_loss, second_metric, _ = _global_objective_mean(
        second_numerator,
        torch.tensor(6.0),
        global_denominator=update_denominator,
        accumulation_scale=2,
    )
    # Accelerate divides each backward contribution by the accumulation count.
    accumulated_objective = (first_loss + second_loss) / 2
    accumulated_metric = (first_metric + second_metric) / 2

    torch.testing.assert_close(accumulated_objective, torch.tensor(22.0 / 8.0))
    torch.testing.assert_close(accumulated_metric, torch.tensor(22.0 / 8.0))
    accumulated_objective.backward()
    torch.testing.assert_close(first_numerator.grad, torch.tensor(1.0 / 8.0))
    torch.testing.assert_close(second_numerator.grad, torch.tensor(1.0 / 8.0))


def test_flow_randomness_shapes_fail_closed():
    stream = flow_stream()
    output = flow_output(stream, ZeroExpert())

    with pytest.raises(ValueError, match="timesteps must have shape"):
        perceptron_isaac_training_loss(
            output,
            stream,
            loss_plan="flow_matching_action_prediction",
            train_samples_per_chunk=2,
            flow_timesteps=torch.tensor([[0.5]]),
            flow_noise=torch.zeros(2, 1, 2, 2),
        )


def test_internally_sampled_timestep_is_reported():
    stream = flow_stream()
    output = flow_output(stream, ZeroExpert())

    result = perceptron_isaac_training_loss(
        output,
        stream,
        loss_plan="flow_matching_action_prediction",
        train_samples_per_chunk=2,
    )

    torch.testing.assert_close(result.metrics["flow_mean_timestep"], torch.tensor(0.5))
    assert torch.isfinite(result.metrics["loss"])
