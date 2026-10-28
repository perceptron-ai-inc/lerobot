"""Focused native recorder/materializer controls; no injected auxiliary loss tensors.

Fixed positive token embeddings and zero mixer/shared weights keep both real
experts below the shared null row at every position, including forwarded pads.
The overflow variant changes only native gate weights, not the materializer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import pytest
import torch
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
    GenesisNullDispatchTrace,
    GenesisNullSparseMoeBlock,
)
from lerobot.policies.perceptron_isaac.tensor_stream import Event, TensorStream, TextType, create_stream
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config
from tests.policies.perceptron_isaac.test_p2_moe_regressions import (
    AuxiliaryOracle,
    _capture_router_logits,
    _reference_auxiliary,
)


@dataclass(frozen=True)
class NativeRouterCapture:
    output: ModelOutput
    logits: tuple[Tensor, ...]
    dispatch: tuple[GenesisNullDispatchTrace, ...]


def _all_null_native_model() -> Mk1Qwen36VLAForActionGeneration:
    raw = _config(test_geometry=True, mtp_present=False)
    raw["text_config"]["layer_types"] = ["full_attention", "full_attention"]
    raw["text_config"]["full_attention_interval"] = 1
    contract = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
    with torch.random.fork_rng():
        torch.manual_seed(47)
        model = Mk1Qwen36VLAForActionGeneration(_build_config(raw, contract)).float().train()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.model.language_model.embed_tokens.weight.copy_(torch.linspace(0.1, 0.8, 8).expand(32, -1))
        for layer, block in enumerate(_native_blocks(model)):
            block.gate.weight[-1, 0] = 0.7 + 0.3 * layer
    return model


def _native_blocks(model: Mk1Qwen36VLAForActionGeneration) -> tuple[GenesisNullSparseMoeBlock, ...]:
    blocks = tuple(module for module in model.modules() if isinstance(module, GenesisNullSparseMoeBlock))
    assert len(blocks) == 2
    return blocks


def _padded_stream(padding: int) -> TensorStream:
    stream = flow_stream()
    if not padding:
        return stream
    events = list(stream.streams[0].events)
    events.append(
        Event.from_text_tokens(
            torch.zeros(padding, dtype=torch.long), type=TextType.padding, role=None, time=(0, 0)
        )
    )
    return TensorStream([create_stream(events, list(TextType))])


def _capture_native_forward(
    model: Mk1Qwen36VLAForActionGeneration, stream: TensorStream
) -> NativeRouterCapture:
    blocks = _native_blocks(model)
    logits: list[list[Tensor]] = [[], []]
    dispatch: list[list[GenesisNullDispatchTrace]] = [[], []]
    handles = []
    for layer, block in enumerate(blocks):
        handles.append(
            block.register_forward_pre_hook(partial(_capture_router_logits, captures=logits[layer]))
        )
        block.dispatch_trace_observer = dispatch[layer].append
    try:
        output = model.train_forward(stream)
    finally:
        for handle in handles:
            handle.remove()
        for block in blocks:
            block.dispatch_trace_observer = None
    assert all(len(values) == 1 for values in logits)
    assert all(len(values) == 1 for values in dispatch)
    return NativeRouterCapture(
        output, tuple(values[0] for values in logits), tuple(values[0] for values in dispatch)
    )


def _logical_reference(capture: NativeRouterCapture) -> AuxiliaryOracle:
    layers = [_reference_auxiliary(logits, real=2, null=2, top_k=2) for logits in capture.logits]
    return AuxiliaryOracle(
        torch.stack([layer.balance for layer in layers]).mean(),
        torch.stack([layer.z for layer in layers]).mean(),
    )


def _task_only_output(output: ModelOutput) -> ModelOutput:
    # Reference view of the same real native task graph, without auxiliary attachment.
    return ModelOutput(
        final_activations=output.final_activations,
        final_embedding=output.final_embedding,
        flow_matching_expert=output.flow_matching_expert,
    )


def _native_loss(output: ModelOutput, stream: TensorStream, *, balance: float, z: float) -> Tensor:
    return perceptron_isaac_training_loss(
        output,
        stream,
        flow_timesteps=torch.tensor([[0.5]]),
        flow_noise=torch.zeros(1, 1, 3, 4),
        moe_load_balancing_loss_coeff=balance,
        moe_z_loss_coeff=z,
    ).loss


def _assert_gradients_match(actual: Tensor, expected: Tensor, parameters: tuple[nn.Parameter, ...]) -> None:
    expected_gradients = torch.autograd.grad(expected, parameters, retain_graph=True, allow_unused=True)
    actual_gradients = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        if expected_gradient is None:
            assert actual_gradient is None or not bool(actual_gradient.count_nonzero())
        else:
            assert torch.isfinite(expected_gradient).all()
            assert actual_gradient is not None
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("padding", [0, 3])
def test_native_all_null_capture_materializes_every_layer_position_and_router_gradient(padding: int) -> None:
    model = _all_null_native_model()
    stream = _padded_stream(padding)
    capture = _capture_native_forward(model, stream)
    tokens = 5 + padding  # Six real input tokens; NTP removes only the final input position.
    assert capture.output.final_activations.shape == (1, tokens, 8)
    info = capture.output.additional_info
    for key in ("load_balancing_loss", "router_z_loss", "router_dispatch_counts", "router_token_count"):
        assert len(info[key]) == 2
    for layer, (logits, dispatch) in enumerate(zip(capture.logits, capture.dispatch, strict=True)):
        assert logits.shape == (tokens, 3)
        torch.testing.assert_close(dispatch.selected_experts, torch.tensor([[2, 3]]).expand(tokens, -1))
        assert dispatch.dispatch_token_indices.numel() == 0
        assert dispatch.dispatch_expert_indices.numel() == 0
        assert dispatch.dispatch_route_weights.numel() == 0
        torch.testing.assert_close(
            info["router_dispatch_counts"][layer], torch.tensor([0, 0, tokens, tokens]).float()
        )
        torch.testing.assert_close(info["router_token_count"][layer], torch.tensor(float(tokens)))
        oracle = _reference_auxiliary(logits, real=2, null=2, top_k=2)
        torch.testing.assert_close(info["load_balancing_loss"][layer], oracle.balance)
        torch.testing.assert_close(info["router_z_loss"][layer], oracle.z)
    task = _native_loss(_task_only_output(capture.output), stream, balance=0, z=0)
    torch.testing.assert_close(task, 7.5 + torch.log(torch.tensor(32.0)))
    reference = _logical_reference(capture)
    parameters = tuple(block.gate.weight for block in _native_blocks(model))
    task_gradients = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    assert all(gradient is None or not bool(gradient.count_nonzero()) for gradient in task_gradients)
    expected_gradients = torch.autograd.grad(reference.weighted(), parameters, retain_graph=True)
    assert all(bool(gradient[-1].abs().max() > 1e-6) for gradient in expected_gradients)
    # Separate enable controls prevent either coefficient from silently controlling both terms.
    for balance, z in ((0.02, 0.0), (0.0, 0.001), (0.02, 0.001)):
        actual = _native_loss(capture.output, stream, balance=balance, z=z)
        expected = task + balance * reference.balance + z * reference.z
        torch.testing.assert_close(actual, expected)
        _assert_gradients_match(actual, expected, parameters)


@pytest.mark.parametrize("balance", [0.0, 0.02], ids=["both_disabled", "balance_enabled"])
def test_disabled_native_z_overflow_preserves_finite_task_loss_and_gradients(balance: float) -> None:
    model = _all_null_native_model()
    with torch.no_grad():
        for block in _native_blocks(model):
            # Equal huge finite logits retain real routes and finite probabilities/task output.
            block.gate.weight[:, 0] = 2e20
    stream = _padded_stream(0)
    capture = _capture_native_forward(model, stream)
    for logits in capture.logits:
        assert torch.isfinite(logits).all()
        assert bool((logits.abs() > 1e19).all())
        assert bool(torch.isinf(logits.logsumexp(-1).square()).all())
    assert all(torch.isfinite(trace.logical_probabilities).all() for trace in capture.dispatch)
    assert all(not bool(torch.isfinite(z)) for z in capture.output.additional_info["router_z_loss"])
    task = _native_loss(_task_only_output(capture.output), stream, balance=0, z=0)
    assert torch.isfinite(task)
    expected = task
    if balance:
        # Do not multiply the overflowing reference z by zero either.
        expected = (
            expected + balance * torch.stack(capture.output.additional_info["load_balancing_loss"]).mean()
        )
    actual = _native_loss(capture.output, stream, balance=balance, z=0)
    assert torch.isfinite(actual), "zero z coefficient attached an overflowing native materialized branch"
    torch.testing.assert_close(actual, expected)
    parameters = (
        model.model.language_model.embed_tokens.weight,
        model.lm_head.weight,
        *tuple(block.gate.weight for block in _native_blocks(model)),
        dict(model.named_parameters())["model.action_expert.action_expert.final_layer.linear.bias"],
    )
    _assert_gradients_match(actual, expected, parameters)


def _record_layer_visit(module: nn.Module, inputs: tuple[Tensor, ...], *, visits: list[int]) -> None:
    assert isinstance(module, GenesisNullSparseMoeBlock)
    visits.append(inputs[0].shape[1])


def test_native_all_null_auxiliary_only_gradients_survive_activation_recomputation() -> None:
    reference_model = _all_null_native_model()
    model = _all_null_native_model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.is_gradient_checkpointing
    stream = _padded_stream(3)
    reference_capture = _capture_native_forward(reference_model, stream)
    oracle = _logical_reference(reference_capture)
    reference_parameters = (
        reference_model.model.language_model.embed_tokens.weight,
        *tuple(block.gate.weight for block in _native_blocks(reference_model)),
    )
    expected = torch.autograd.grad(oracle.weighted(), reference_parameters)
    visits: list[list[int]] = [[], []]
    handles = [
        block.register_forward_pre_hook(partial(_record_layer_visit, visits=visits[layer]))
        for layer, block in enumerate(_native_blocks(model))
    ]
    try:
        output = model.train_forward(stream)
        assert all(len(layer) == 1 for layer in visits)
        info = output.additional_info
        assert len(info["load_balancing_loss"]) == len(info["router_z_loss"]) == 2
        auxiliary = 0.02 * torch.stack(info["load_balancing_loss"]).mean()
        auxiliary = auxiliary + 0.001 * torch.stack(info["router_z_loss"]).mean()
        torch.testing.assert_close(auxiliary, oracle.weighted())
        auxiliary.backward()  # No task gradient can conceal a detached recorder side channel.
        assert all(len(layer) >= 2 for layer in visits), "decoder layers did not recompute"
    finally:
        for handle in handles:
            handle.remove()
    parameters = (
        model.model.language_model.embed_tokens.weight,
        *tuple(block.gate.weight for block in _native_blocks(model)),
    )
    for parameter, gradient in zip(parameters, expected, strict=True):
        assert bool(gradient.abs().max() > 1e-6)
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, gradient, atol=2e-6, rtol=2e-5)
    for count, tokens in zip(info["router_dispatch_counts"], info["router_token_count"], strict=True):
        torch.testing.assert_close(count, torch.tensor([0.0, 0.0, 8.0, 8.0]))
        torch.testing.assert_close(tokens, torch.tensor(8.0))
