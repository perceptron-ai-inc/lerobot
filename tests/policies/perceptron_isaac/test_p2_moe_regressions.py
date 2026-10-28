"""P2.10 reference null-aware balance/z objective, not the LM-logit penalty.

Reference commit 6bd5d0a4b151b815ff990be85b6335630a4555cd:
moe/core.py + moe/load_balancing.py expand the shared null row logically,
count selected slots without dividing by top-k, and average z over tokens.
moe_auxiliary_losses.py averages each term over layers before applying 0.02/0.001.
The frozen block passes every forwarded position to MoE without a loss mask:
user tokens, flow marker, and forwarded padding count too, not just agent labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import perceptron_isaac_training_loss
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    Mk1Qwen36VLAForActionGeneration,
    _build_config,
)
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import ModelOutput
from lerobot.policies.perceptron_isaac.modeling_qwen36_moe import (
    GenesisNullSparseMoeBlock,
    GenesisNullTopKRouter,
)
from lerobot.policies.perceptron_isaac.tensor_stream import Event, TensorStream, TextType, create_stream
from tests.policies.perceptron_isaac._fixtures import ZeroExpert, flow_stream
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config
from tests.policies.perceptron_isaac.test_modeling_qwen36_moe import _text_config


@dataclass(frozen=True)
class AuxiliaryOracle:
    balance: Tensor
    z: Tensor

    def weighted(self) -> Tensor:
        return 0.02 * self.balance + 0.001 * self.z


def _reference_auxiliary(physical_logits: Tensor, *, real: int, null: int, top_k: int) -> AuxiliaryOracle:
    logical = torch.cat((physical_logits[:, :real], physical_logits[:, real:].expand(-1, null)), -1).float()
    probabilities = logical.softmax(-1)
    selected = torch.argsort(probabilities.detach(), dim=-1, descending=True, stable=True)[:, :top_k]
    # Logical null slots count separately. Neither physical softmax nor real-only
    # renormalized task weights is the probability distribution for this loss.
    counts = functional.one_hot(selected, num_classes=real + null).float().sum((0, 1))
    frequency = counts / logical.shape[0]
    balance = (real + null) * (frequency * probabilities.mean(0)).sum()
    z = logical.logsumexp(-1).square().mean()
    return AuxiliaryOracle(balance, z)


def _capture_router_logits(module: nn.Module, inputs: tuple[Tensor, ...], captures: list[Tensor]) -> None:
    assert isinstance(module, GenesisNullSparseMoeBlock)
    hidden = inputs[0].reshape(-1, module.gate.in_features)
    captures.append(functional.linear(hidden, module.gate.weight))


@pytest.mark.parametrize("lm_penalty", [0.0, 0.0001])
@pytest.mark.parametrize("padding", [0, 2])
def test_native_null_model_joint_loss_has_exact_layer_mean_auxiliary_and_null_gradients(
    lm_penalty: float,
    padding: int,
) -> None:
    raw = _config(test_geometry=True, mtp_present=False)
    raw["text_config"]["layer_types"] = ["full_attention", "full_attention"]
    raw["text_config"]["full_attention_interval"] = 1
    contract = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
    with torch.random.fork_rng():
        torch.manual_seed(47)
        model = Mk1Qwen36VLAForActionGeneration(_build_config(raw, contract)).float().train()
    blocks = [module for module in model.modules() if isinstance(module, GenesisNullSparseMoeBlock)]
    assert len(blocks) == 2
    with torch.no_grad():
        for layer, block in enumerate(blocks):
            block.gate.weight.zero_()
            # At least real expert 1 always survives top-k (real-first ties).
            # This keeps task gradients connected even when expert 0 loses to null.
            block.gate.weight[0, 0] = 0.7 * layer
    captures: list[Tensor] = []
    handles = [
        block.register_forward_pre_hook(
            lambda module, inputs: _capture_router_logits(module, inputs, captures)
        )
        for block in blocks
    ]
    stream = flow_stream()
    if padding:
        # NTP drops the last pad; the terminal control and remaining pads are forwarded.
        events = list(stream.streams[0].events)
        events.append(
            Event.from_text_tokens(
                torch.zeros(padding, dtype=torch.long),
                type=TextType.padding,
                role=None,
                time=(0, 0),
            )
        )
        stream = TensorStream([create_stream(events, list(TextType))])
    try:
        output = model.train_forward(stream)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captures) == 2
    assert all(logits.shape == (5 + padding, 3) for logits in captures)
    layers = [_reference_auxiliary(logits, real=2, null=2, top_k=2) for logits in captures]
    auxiliary = AuxiliaryOracle(
        torch.stack([layer.balance for layer in layers]).mean(),
        torch.stack([layer.z for layer in layers]).mean(),
    )
    # NTP labels select only agent action/control tokens; router auxiliaries do not.
    logits = output.final_activations[0, 3:5].float() @ model.lm_head.weight.float().t()
    text = functional.cross_entropy(logits, torch.tensor([9, 2]))
    text += lm_penalty * logits.logsumexp(-1).square().mean()
    target = torch.zeros(1, 3, 4)
    target[0, :2, :2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    prediction = model.action_expert(
        output.final_activations,
        torch.tensor([[True, True] + [False] * (3 + padding)]),
        target * 0.5,
        torch.tensor([0.5]),
        action_mask=torch.tensor([[True, True, False]]),
    )
    flow = (prediction[0, :2, :2] - target[0, :2, :2]).square().mean()
    task = text + flow
    result = perceptron_isaac_training_loss(
        output,
        stream,
        flow_timesteps=torch.tensor([[0.5]]),
        flow_noise=torch.zeros(1, 1, 3, 4),
        softmax_auxiliary_loss_scale=lm_penalty,
    )
    parameters = tuple(block.gate.weight for block in blocks)
    task_gradients = torch.autograd.grad(task, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(task + auxiliary.weighted(), parameters, retain_graph=True)
    actual_gradients = torch.autograd.grad(result.loss, parameters)
    # Real-route renormalization cancels task signal to the shared null row;
    # the reference auxiliaries supply a distinct, nonzero gradient there.
    for task_gradient, expected, actual in zip(
        task_gradients, expected_gradients, actual_gradients, strict=True
    ):
        torch.testing.assert_close(task_gradient[-1], torch.zeros_like(task_gradient[-1]), atol=1e-6, rtol=0)
        assert expected[-1].abs().max() > 1e-6
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.loss, task + auxiliary.weighted())
    torch.testing.assert_close(result.metrics["loss"], (task + auxiliary.weighted()).detach())


@pytest.mark.parametrize("routing", ["all_null", "real_first_tie"])
def test_reference_router_terms_reach_backward_even_when_task_has_no_null_signal(routing: str) -> None:
    router = GenesisNullTopKRouter(_text_config(num_null_experts=3))
    with torch.no_grad():
        router.weight.zero_()
        if routing == "all_null":
            router.weight[2, 0] = 2
    hidden = torch.zeros(5, 24)
    hidden[:, 0] = torch.arange(1, 6).float()
    routed = router.route(hidden)
    if routing == "all_null":
        assert (routed.selected_experts >= 2).all()
        assert routed.real_route_weights.count_nonzero() == 0
    else:
        assert torch.equal(routed.selected_experts, torch.tensor([[0, 1]]).expand(5, -1))
    oracle = _reference_auxiliary(routed.physical_logits, real=2, null=3, top_k=2)
    if routing == "real_first_tie":
        # Top-k is SUMMED, not averaged: balanced ties have balance=K=2, not 1.
        torch.testing.assert_close(oracle.balance, torch.tensor(2.0))
        torch.testing.assert_close(oracle.z, torch.log(torch.tensor(5.0)).square())
    stream = flow_stream()
    expert = ZeroExpert()
    output = ModelOutput(
        final_activations=torch.zeros(1, 5, 24),
        final_embedding=nn.Linear(24, 32, bias=False),
        flow_matching_expert=expert,
        additional_info={"load_balancing_loss": [oracle.balance], "router_z_loss": [oracle.z]},
    )
    result = perceptron_isaac_training_loss(
        output,
        stream,
        loss_plan="flow_matching_action_prediction",
        flow_timesteps=torch.tensor([[0.5]]),
        flow_noise=torch.zeros(1, 1, 2, 2),
    )
    expected_gradient = torch.autograd.grad(oracle.weighted(), router.weight, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(result.loss, router.weight, allow_unused=True)[0]
    assert actual_gradient is not None, "native loss discarded differentiable real-router auxiliary terms"
    assert expected_gradient[-1].abs().max() > 0
    torch.testing.assert_close(actual_gradient, expected_gradient)
    torch.testing.assert_close(result.loss, 7.5 + oracle.weighted())
