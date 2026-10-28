"""P2.01: real native DiT clean-at-one values and parameter gradients."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import perceptron_isaac_training_loss
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import ModelOutput, build_action_expert_head
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_action_expert_contract import _dit_contract


@pytest.mark.parametrize("draws", [1, 2])
@pytest.mark.parametrize("tau", [0.0, 0.25, 1.0])
def test_real_dit_loss_and_head_gradients_use_clean_at_one(draws: int, tau: float) -> None:
    stream = flow_stream()
    with torch.random.fork_rng():
        torch.manual_seed(61)
        head = build_action_expert_head(_dit_contract(action_dim=2, action_horizon=2), vlm_dim=16)
        # Nonzero prediction distinguishes signs in both loss values and gradients.
        with torch.no_grad():
            head.action_expert.final_layer.linear.weight.normal_(std=0.03)
            head.action_expert.final_layer.linear.bias.fill_(0.2)
        batch, length = stream.shape
        output = ModelOutput(
            final_activations=torch.linspace(-0.4, 0.6, batch * (length - 1) * 16).reshape(
                batch, length - 1, 16
            ),
            final_embedding=torch.nn.Linear(16, 32, bias=False),
            flow_matching_expert=head,
        )
        noise = torch.linspace(-0.5, 0.8, draws * 4).reshape(draws, 1, 2, 2)
        times = torch.full((draws, 1), tau)
        target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        interpolated = times[:, :, None, None] * noise + (1 - times[:, :, None, None]) * target
        mask = torch.tensor([[True, True, False, False, False]])
        prediction = head(
            output.final_activations,
            mask,
            interpolated[0] if draws == 1 else interpolated,
            times[0] if draws == 1 else times,
            action_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        oracle = (prediction - (target - noise)[0 if draws == 1 else slice(None)]).square().mean()
        result = perceptron_isaac_training_loss(
            output,
            stream,
            loss_plan="flow_matching_action_prediction",
            train_samples_per_chunk=draws,
            flow_timesteps=times,
            flow_noise=noise,
        )
        parameters = (
            head.action_expert.final_layer.linear.bias,
            head.action_expert.final_layer.linear.weight,
        )
        actual_gradients = torch.autograd.grad(result.loss, parameters, retain_graph=True)
        expected_gradients = torch.autograd.grad(oracle, parameters)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(result.loss, oracle)
