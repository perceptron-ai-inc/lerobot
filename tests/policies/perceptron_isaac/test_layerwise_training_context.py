"""Regressions for training producers and both consumers of layerwise Flow context."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Literal, cast

import pytest
import torch

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import perceptron_isaac_training_loss
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import Mk1Qwen36VLAForActionGeneration
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
    ModelOutput,
    Qwen35VLAForActionGeneration,
    setup_qwen35_vla_for_training,
)
from lerobot.policies.perceptron_isaac.tensor_stream import TensorStream
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_layerwise_native_context import (
    CrossAttentionRead,
    _expected_projected_kv,
    _record_cross_attention,
    _rms_norm,
    _tiny_backbone,
)


def _training_model(
    *, draws: int = 1, serialization: Literal["schema1", "action-schema2"] = "action-schema2"
) -> Mk1Qwen36VLAForActionGeneration:
    model = _tiny_backbone(serialization=serialization, hidden_norm="final_norm")
    setup_qwen35_vla_for_training(cast(Qwen35VLAForActionGeneration, model), train_samples_per_chunk=draws)
    for parameter in model.get_input_embeddings().parameters():
        parameter.requires_grad_(False)
    return model


@pytest.mark.parametrize(
    ("generic_producer", "serialization"),
    [
        pytest.param(False, "action-schema2", id="native-layerwise"),
        pytest.param(True, "action-schema2", id="generic-layerwise"),
        pytest.param(False, "schema1", id="shared-final-control"),
    ],
)
def test_training_routes_owned_context_and_backpropagates_both_losses(
    generic_producer: bool, serialization: Literal["schema1", "action-schema2"]
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        model = _training_model(serialization=serialization)
        head = model.model.action_expert
        assert head is not None
        decoder_outputs: list[torch.Tensor] = []
        owned_context: list[object] = []
        reads: list[list[CrossAttentionRead]] = [[], []]
        stream = flow_stream()
        with ExitStack() as stack:
            stack.enter_context(
                model.model.register_forward_hook(
                    lambda _module, _args, value: owned_context.append(value.get("flow_matching_context"))
                )
            )
            for layer in model.model.language_model.layers:
                stack.enter_context(
                    layer.register_forward_hook(lambda _module, _args, value: decoder_outputs.append(value))
                )
            for block, block_reads in zip(head.action_expert.blocks, reads, strict=True):
                stack.enter_context(
                    block.cross_attn.register_forward_pre_hook(
                        _record_cross_attention(block_reads), with_kwargs=True
                    )
                )
            output = (
                Qwen35VLAForActionGeneration.train_forward(cast(Qwen35VLAForActionGeneration, model), stream)
                if generic_producer
                else model.train_forward(stream)
            )
            result = perceptron_isaac_training_loss(
                output, stream, flow_timesteps=torch.full((1, 1), 0.5), flow_noise=torch.zeros(1, 1, 3, 4)
            )
        assert len(owned_context) == 1 and owned_context[0] is output.flow_matching_context
        if serialization == "schema1":
            assert output.flow_matching_context is output.final_activations
            expected = (output.final_activations,) * 2
        else:
            assert len(decoder_outputs) == 4
            expected = tuple(
                _rms_norm(decoder_outputs[index]) * (1 + model.model.language_model.norm.weight)
                for index in (1, 3)
            )
            assert isinstance(output.flow_matching_context, tuple)
            torch.testing.assert_close(output.flow_matching_context, expected)
            assert output.flow_matching_context[-1] is output.final_activations
        for block_reads, context in zip(reads, expected, strict=True):
            assert len(block_reads) == 1
            key, value = _expected_projected_kv(head, context)
            torch.testing.assert_close(block_reads[0].key, key)
            torch.testing.assert_close(block_reads[0].value, value)
        assert torch.isfinite(result.loss)
        assert all(result.metrics[name].item() > 0 for name in ("text_loss", "flow_loss"))
        result.loss.backward()
        for parameter in (model.lm_head.weight, head.action_expert.context_k_proj.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero() > 0
        assert all(parameter.grad is None for parameter in model.get_input_embeddings().parameters())
        if serialization == "action-schema2":
            missing = ModelOutput(output.final_activations, output.final_embedding, flow_matching_expert=head)
            with pytest.raises(ValueError, match="layerwise context requires a sequence of Tensor layers"):
                perceptron_isaac_training_loss(
                    missing,
                    stream,
                    flow_timesteps=torch.full((1, 1), 0.5),
                    flow_noise=torch.zeros(1, 1, 3, 4),
                )


@pytest.mark.parametrize(
    ("serialization", "draws", "detach"),
    [("schema1", 1, True), ("action-schema2", 8, False), ("action-schema2", 8, True)],
)
def test_flow_only_detach_controls_every_selected_context_gradient(
    serialization: Literal["schema1", "action-schema2"], draws: int, detach: bool
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        model = _training_model(draws=draws, serialization=serialization)
        stream = flow_stream()
        output = model.train_forward(stream)
        context = output.flow_matching_context
        contexts = (context,) if isinstance(context, torch.Tensor) else context
        for context in contexts:
            context.retain_grad()
        result = perceptron_isaac_training_loss(
            output,
            stream,
            loss_plan="flow_matching_action_prediction",
            flow_matching_detach_vlm_activations=detach,
            train_samples_per_chunk=draws,
            moe_load_balancing_loss_coeff=0,
            moe_z_loss_coeff=0,
            flow_timesteps=torch.full((draws, 1), 0.5),
            flow_noise=torch.zeros(draws, 1, 3, 4),
        )
        result.loss.backward()
        for context in contexts:
            if detach:
                assert context.grad is None
            else:
                assert context.grad is not None and torch.isfinite(context.grad).all()
                assert context.grad.count_nonzero() > 0
        if detach:
            assert all(parameter.grad is None for parameter in model.model.language_model.parameters())
        else:
            gradient = model.model.language_model.norm.weight.grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
        assert model.lm_head.weight.grad is None
        head = model.model.action_expert
        assert head is not None
        gradient = head.action_expert.context_k_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.count_nonzero() > 0


@pytest.mark.parametrize(("draws", "detach"), [(1, False), (8, True)])
def test_collective_only_context_keeps_zero_counts_rng_and_backward(draws: int, detach: bool) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        model = _training_model(draws=draws)
        stream = TensorStream.collective_dummy()
        output = model.train_forward(stream)
        rng = torch.random.get_rng_state()
        result = perceptron_isaac_training_loss(
            output,
            stream,
            train_samples_per_chunk=draws,
            flow_matching_detach_vlm_activations=detach,
            global_denominators={"flow": torch.tensor(1), "text": torch.tensor(2)},
        )
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert result.loss.item() == 0
        assert all(
            result.metrics[name].item() == 0
            for name in ("flow_chunk_count", "text_token_count", "flow_element_count")
        )
        result.loss.backward()
        head = model.model.action_expert
        assert head is not None
        for parameter in (head.action_expert.context_k_proj.weight, model.model.language_model.norm.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero() == 0
