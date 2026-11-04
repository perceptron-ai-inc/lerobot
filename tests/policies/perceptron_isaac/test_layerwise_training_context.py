"""Real tiny native training must retain layerwise Flow context beside FAST activations."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from typing import cast

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


@pytest.mark.parametrize("draws", [1, 8])
@pytest.mark.parametrize("generic_producer", [False, True], ids=["native", "generic-body"])
@pytest.mark.parametrize("layerwise", [True, False], ids=["layerwise", "shared-final"])
def test_real_mixed_training_loss_backward(draws: int, generic_producer: bool, layerwise: bool) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        model = _tiny_backbone(
            serialization="action-schema2" if layerwise else "schema1", hidden_norm="final_norm"
        )
        # The native wrapper intentionally reuses the generic VLA shell's descriptors.
        # This cast exercises its producer body, not a different dense-backbone implementation.
        shell = cast(Qwen35VLAForActionGeneration, model)
        setup_qwen35_vla_for_training(shell, train_samples_per_chunk=draws)
        for parameter in model.get_input_embeddings().parameters():
            parameter.requires_grad_(False)
        stream = flow_stream()
        output = (
            Qwen35VLAForActionGeneration.train_forward(shell, stream)
            if generic_producer
            else model.train_forward(stream)
        )
        # Execute the real loss first: the baseline must fail inside the expert's
        # layerwise contract, not on a newly invented output attribute or a mock.
        result = perceptron_isaac_training_loss(
            output,
            stream,
            train_samples_per_chunk=draws,
            flow_timesteps=torch.full((draws, 1), 0.5),
            flow_noise=torch.zeros(draws, 1, 3, 4),
        )
        assert torch.isfinite(result.loss)
        assert result.metrics["flow_chunk_count"].item() == 1
        assert result.metrics["text_token_count"].item() == 2
        assert result.metrics["flow_element_count"].item() == 4 * draws
        assert result.metrics["flow_loss"].item() > 0
        assert result.metrics["text_loss"].item() > 0
        result.loss.backward()
        expert = model.model.action_expert
        assert expert is not None
        for parameter in (
            expert.action_expert.context_k_proj.weight,
            model.model.language_model.norm.weight,
            model.lm_head.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in model.get_input_embeddings().parameters())


def _layerwise_training_model(
    *, draws: int = 1, hidden_norm: str = "final_norm"
) -> Mk1Qwen36VLAForActionGeneration:
    model = _tiny_backbone(serialization="action-schema2", hidden_norm=hidden_norm)
    setup_qwen35_vla_for_training(cast(Qwen35VLAForActionGeneration, model), train_samples_per_chunk=draws)
    for parameter in model.get_input_embeddings().parameters():
        parameter.requires_grad_(False)
    return model


@pytest.mark.parametrize("generic_producer", [False, True], ids=["native", "generic-body"])
@pytest.mark.parametrize("hidden_norm", ["final_norm", "none"])
def test_training_routes_exact_selected_layers_inside_owning_forward(
    generic_producer: bool, hidden_norm: str
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        model = _layerwise_training_model(hidden_norm=hidden_norm)
        head = model.model.action_expert
        assert head is not None
        decoder_outputs: list[torch.Tensor] = []
        contexts_owned: list[bool] = []
        reads: list[list[CrossAttentionRead]] = [[], []]

        def record_decoder(module: torch.nn.Module, args: tuple[object, ...], value: object) -> None:
            del module, args
            assert isinstance(value, torch.Tensor)
            decoder_outputs.append(value)

        def record_owner(module: torch.nn.Module, args: tuple[object, ...], value: object) -> None:
            del module, args
            assert isinstance(value, Mapping)
            contexts_owned.append("flow_matching_context" in value)

        stream = flow_stream()
        with ExitStack() as stack:
            stack.enter_context(model.model.register_forward_hook(record_owner))
            for layer in model.model.language_model.layers:
                stack.enter_context(layer.register_forward_hook(record_decoder))
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
        assert contexts_owned == [True]
        assert len(decoder_outputs) == 4
        contexts = output.flow_matching_context
        assert isinstance(contexts, tuple) and len(contexts) == 2
        selected = (1, 3) if hidden_norm == "final_norm" else (0, 2)
        for index, layer in enumerate(selected):
            expected = decoder_outputs[layer]
            if hidden_norm == "final_norm":
                expected = _rms_norm(expected) * (1 + model.model.language_model.norm.weight)
            torch.testing.assert_close(contexts[index], expected)
            expected_key, expected_value = _expected_projected_kv(head, expected)
            assert len(reads[index]) == 1
            torch.testing.assert_close(reads[index][0].key, expected_key)
            torch.testing.assert_close(reads[index][0].value, expected_value)
            expected_mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32)
            torch.testing.assert_close(
                reads[index][0].mask[:, 0, 0], (1 - expected_mask) * torch.finfo(torch.float32).min
            )
        if hidden_norm == "final_norm":
            assert contexts[-1] is output.final_activations
        assert output.final_activations.shape == (1, 5, 8)
        assert output.logits.shape == (1, 5, 32)
        if generic_producer:
            assert output.additional_info == {}
        else:
            for key in (
                "load_balancing_loss",
                "router_z_loss",
                "router_token_count",
                "router_dispatch_counts",
            ):
                assert len(output.additional_info[key]) == 4
                assert all(torch.isfinite(value).all() for value in output.additional_info[key])
            assert all(value.item() == 5 for value in output.additional_info["router_token_count"])
            assert {"load_balancing_loss", "router_z_loss"} <= result.metrics.keys()
        result.loss.backward()


@pytest.mark.parametrize("draws", [1, 8])
@pytest.mark.parametrize("detach", [False, True])
def test_flow_only_detach_controls_every_selected_context_gradient(draws: int, detach: bool) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        model = _layerwise_training_model(draws=draws)
        stream = flow_stream()
        output = model.train_forward(stream)
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
        contexts = output.flow_matching_context
        assert isinstance(contexts, tuple)
        for context in contexts:
            assert context.requires_grad
            context.retain_grad()
        result.loss.backward()
        backbone = tuple(model.model.language_model.parameters())
        if detach:
            assert all(context.grad is None for context in contexts)
            assert all(parameter.grad is None for parameter in backbone)
        else:
            for context in contexts:
                assert context.grad is not None and torch.isfinite(context.grad).all()
                assert context.grad.abs().sum() > 0
            gradient = model.model.language_model.norm.weight.grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert model.lm_head.weight.grad is None
        expert = model.model.action_expert
        assert expert is not None
        gradient = expert.action_expert.context_k_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("draws", [1, 8])
@pytest.mark.parametrize("detach", [False, True])
def test_layerwise_collective_only_keeps_zero_counts_rng_and_backward(draws: int, detach: bool) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        model = _layerwise_training_model(draws=draws)
        stream = TensorStream.collective_dummy()
        output = model.train_forward(stream)
        rng = torch.random.get_rng_state()
        result = perceptron_isaac_training_loss(
            output,
            stream,
            flow_matching_detach_vlm_activations=detach,
            train_samples_per_chunk=draws,
            global_denominators={"flow": torch.tensor(1), "text": torch.tensor(2)},
        )
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert torch.isfinite(result.loss) and result.loss.item() == 0
        for key in ("flow_chunk_count", "text_token_count", "flow_element_count"):
            assert result.metrics[key].item() == 0
        for key in ("router_token_count", "router_dispatch_counts"):
            assert len(output.additional_info[key]) == 4
            assert all(value.count_nonzero().item() == 0 for value in output.additional_info[key])
        result.loss.backward()
        expert = model.model.action_expert
        assert expert is not None
        for parameter in (expert.action_expert.context_k_proj.weight, model.model.language_model.norm.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.count_nonzero().item() == 0


def test_missing_layerwise_context_still_fails_closed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        model = _layerwise_training_model()
        stream = flow_stream()
        output = model.train_forward(stream)
        missing_context = ModelOutput(
            output.final_activations, output.final_embedding, flow_matching_expert=output.flow_matching_expert
        )
        with pytest.raises(ValueError, match="layerwise context requires a sequence of Tensor layers"):
            perceptron_isaac_training_loss(
                missing_context,
                stream,
                flow_timesteps=torch.full((1, 1), 0.5),
                flow_noise=torch.zeros(1, 1, 3, 4),
            )
