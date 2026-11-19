"""CPU numeric oracles for real layer capture, final norm and DiT block mapping.

Hooks only observe real tiny modules. No model is replaced with a stub. Decoder
outputs are post-residual, zero-based; block i consumes context_layers[i].
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

import pytest
import torch
from torch import nn
from torch.nn import functional

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    Mk1Qwen36VLAForActionGeneration,
    _build_config,
    _configure_attention_backend,
    _select_gdn_backend,
)
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
    ActionExpertContext,
    DiTActionExpertHead,
    MolmoActExpertHead,
    build_action_expert_head,
)
from lerobot.policies.perceptron_isaac.tensor_stream import (
    ALL_TYPES,
    Event,
    TensorStream,
    TextType,
    create_stream,
)
from tests.policies.perceptron_isaac._layerwise_fixtures import layerwise_action_config
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config


def _tiny_head(*, layerwise: bool) -> DiTActionExpertHead:
    action = layerwise_action_config()
    if not layerwise:
        action["schema_version"] = 1
        for field in ("context_source", "context_layers", "context_hidden_norm"):
            del action[field]
    head = build_action_expert_head(action, vlm_dim=8)
    assert isinstance(head, DiTActionExpertHead)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.uniform_(-0.3, 0.3)
        head.action_expert.context_k_proj.weight.copy_(torch.eye(8))
        head.action_expert.context_v_proj.weight.copy_(torch.eye(8).flip(0))
    return head.float().eval()


def _distinct_layers() -> tuple[torch.Tensor, torch.Tensor]:
    # B=3 differs from the layer count=2. Values vary by batch, token and channel;
    # scalar multiples would become identical after RMSNorm and hide substitutions.
    first = torch.arange(3 * 5 * 8, dtype=torch.float32).reshape(3, 5, 8) / 19 - 2
    second = first.flip(-1).roll(1, dims=1) + torch.tensor([0.0, 1.0, -2.0, 3.0, 1.0, -1.0, 2.0, -3.0])
    return first, second


def _rms_norm(value: torch.Tensor) -> torch.Tensor:
    """Independent F32 oracle, not the production norm helper."""
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)


def _expected_projected_kv(
    head: MolmoActExpertHead, context: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    expert = head.action_expert
    key = _rms_norm(functional.linear(context, expert.context_k_proj.weight)).reshape(
        *context.shape[:2], 2, 4
    )
    value = _rms_norm(functional.linear(context, expert.context_v_proj.weight)).reshape(
        *context.shape[:2], 2, 4
    )
    return _rms_norm(key), value


def _build_context(
    head: DiTActionExpertHead, activations: torch.Tensor | Sequence[torch.Tensor], mask: torch.Tensor
) -> ActionExpertContext:
    return head._build_single_context(
        activations,
        mask,
        None,
        seq_len=3,
        batch_size=mask.shape[0],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_schema1_shared_final_context_control_keeps_projection_math_and_mask() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(41)
        head = _tiny_head(layerwise=False)
        layers = _distinct_layers()
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
        context = _build_context(head, layers[-1], mask)
        expected_k, expected_v = _expected_projected_kv(head, layers[-1])
        assert len(context.kv_contexts) == 2
        for actual_k, actual_v in context.kv_contexts:
            torch.testing.assert_close(actual_k, expected_k)
            torch.testing.assert_close(actual_v, expected_v)
        assert context.cross_mask is not None
        torch.testing.assert_close(
            context.cross_mask[:, 0, 0], (1 - mask).float() * torch.finfo(torch.float32).min
        )


def test_schema2_distinct_layers_feed_corresponding_blocks_through_shared_projections() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(41)
        head = _tiny_head(layerwise=True)
        layers = _distinct_layers()
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
        context = _build_context(head, layers, mask)
        assert len(context.kv_contexts) == len(layers)
        for layer, (actual_k, actual_v) in zip(layers, context.kv_contexts, strict=True):
            expected_k, expected_v = _expected_projected_kv(head, layer)
            torch.testing.assert_close(actual_k, expected_k)
            torch.testing.assert_close(actual_v, expected_v)
        assert not torch.allclose(context.kv_contexts[0][1], context.kv_contexts[1][1])
        assert context.cross_mask is not None
        torch.testing.assert_close(
            context.cross_mask[:, 0, 0], (1 - mask).float() * torch.finfo(torch.float32).min
        )


def test_schema2_real_velocity_matches_existing_per_layer_action_expert_math() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(71)
        head = _tiny_head(layerwise=True)
        layers = _distinct_layers()
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
        actions = torch.randn(3, 3, 4)
        tau = torch.tensor([0.2, 0.4, 0.8])
        reference_context = head.action_expert.prepare_context(
            encoder_kv_states=[(layer, layer) for layer in layers],
            encoder_attention_mask=mask,
            batch_size=3,
            seq_len=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        expected = head.action_expert.forward_with_context(actions, 1 - tau, context=reference_context)
        actual = head(layers, mask, actions, tau)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        swapped = head(tuple(reversed(layers)), mask, actions, tau)
        assert not torch.equal(actual, swapped), "fixture must expose reversed DiT context mapping"


@pytest.mark.parametrize("samples", [1, 8])
def test_schema2_identical_layers_preserve_schema1_sample_values_and_rng(samples: int) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(89)
        shared = _tiny_head(layerwise=False)
        layerwise = _tiny_head(layerwise=True)
        layerwise.load_state_dict(shared.state_dict(), strict=True)
        context = _distinct_layers()[0]
        mask = torch.ones(3, 5, dtype=torch.bool)
        torch.manual_seed(123)
        expected = shared.sample(context, mask, num_steps=2, num_flow_samples=samples)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(123)
        actual = layerwise.sample((context, context), mask, num_steps=2, num_flow_samples=samples)
        assert actual.shape == (3, 3, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize(
    "defect", ["final-tensor", "short", "long", "rank", "batch", "length", "width", "dtype"]
)
def test_schema2_context_shape_mismatches_fail_closed(defect: str) -> None:
    with torch.random.fork_rng():
        head = _tiny_head(layerwise=True)
        first, second = _distinct_layers()
        activations: torch.Tensor | Sequence[torch.Tensor] = (first, second)
        if defect == "final-tensor":
            activations = second
        elif defect == "short":
            activations = (first,)
        elif defect == "long":
            activations = (first, second, first)
        elif defect == "rank":
            activations = (first, second[0])
        elif defect == "batch":
            activations = (first, second[:1])
        elif defect == "length":
            activations = (first, second[:, :-1])
        elif defect == "width":
            activations = (first, second[..., :-1])
        elif defect == "dtype":
            activations = (first, second.double())
        with pytest.raises((ValueError, TypeError), match="context|layer|shape|dtype|Tensor"):
            _build_context(head, activations, torch.ones(3, 5))


@dataclass(frozen=True)
class CrossAttentionRead:
    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor


def _record_cross_attention(
    reads: list[CrossAttentionRead],
) -> Callable[[nn.Module, tuple[object, ...], dict[str, Any]], None]:
    def record(module: nn.Module, args: tuple[object, ...], kwargs: dict[str, Any]) -> None:
        del module, args
        key, value, mask = kwargs["kv_k"], kwargs["kv_v"], kwargs["attn_mask"]
        assert isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        reads.append(CrossAttentionRead(key.detach().clone(), value.detach().clone(), mask.detach().clone()))

    return record


def _history_stream() -> TensorStream:
    def text(tokens: list[int], kind: TextType, time: float) -> Event:
        return Event.from_text_tokens(torch.tensor(tokens), time=(time, time), type=kind, role="user")

    # Unequal prefix lengths, distinct rows, two observation times, padding and
    # post-action control tokens. No executed-action or mistake history is added.
    return TensorStream(
        [
            create_stream(
                [
                    text([6, 7], TextType.text, 0),
                    text([8], TextType.text, 1),
                    text([0], TextType.action_c, 1),
                    text([2, 3], TextType.control, 1),
                ],
                ALL_TYPES,
            ),
            create_stream(
                [
                    text([9], TextType.text, 0),
                    text([10], TextType.text, 1),
                    text([0], TextType.action_c, 1),
                    text([2], TextType.control, 1),
                    text([0, 0], TextType.padding, 1),
                ],
                ALL_TYPES,
            ),
        ]
    )


def _tiny_backbone(
    *, serialization: Literal["schema1", "action-schema2", "vla-schema2"], hidden_norm: str
) -> Mk1Qwen36VLAForActionGeneration:
    raw = _config(test_geometry=True, mtp_present=False)
    raw["text_config"].update(
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    )
    # Parse the existing geometry independently; the constructor receives the new
    # declaration without requiring a future loader/fixture API to exist first.
    geometry = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
    action = layerwise_action_config(
        layers=(1, 3) if hidden_norm == "final_norm" else (0, 2), hidden_norm=hidden_norm
    )
    raw = copy.deepcopy(raw)
    if serialization == "vla-schema2":
        # Archived migrated MK1 uses the VLA envelope's schema 2 with the legacy
        # MolmoAct weight ABI; it must not silently ignore the conditioning triple.
        canonical_action = raw["genesis_vla"]["action_expert"]
        for field in ("context_source", "context_layers", "context_hidden_norm"):
            canonical_action[field] = action[field]
        raw["genesis_vla"]["schema_version"] = 2
    else:
        if serialization == "schema1":
            action["schema_version"] = 1
            for field in ("context_source", "context_layers", "context_hidden_norm"):
                del action[field]
        raw["genesis_vla"]["action_expert"] = action
    config = _build_config(raw, geometry)
    _configure_attention_backend(config, "torch_sdpa_v1")
    model = Mk1Qwen36VLAForActionGeneration(config)
    _select_gdn_backend(model, "cpu")
    model.float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.3, 0.3)
        # Nonuniform, nonidentity final gain makes missing/double normalization
        # observably different, even after the expert's post-projection RMSNorm.
        model.model.language_model.norm.weight.copy_(torch.linspace(-0.4, 1.2, 8))
    return model


@pytest.mark.parametrize(
    ("serialization", "hidden_norm"),
    [
        ("schema1", "final_norm"),
        ("action-schema2", "final_norm"),
        ("action-schema2", "none"),
        ("vla-schema2", "final_norm"),
        ("vla-schema2", "none"),
    ],
    ids=[
        "schema1-control",
        "action-schema2-final-norm",
        "action-schema2-raw-intermediate",
        "vla-schema2-final-norm",
        "vla-schema2-raw-intermediate",
    ],
)
def test_real_native_sample_action_uses_selected_post_residual_layers_in_batch_history_order(
    serialization: Literal["schema1", "action-schema2", "vla-schema2"], hidden_norm: str
) -> None:
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(2026)
        model = _tiny_backbone(serialization=serialization, hidden_norm=hidden_norm)
        head = model.model.action_expert
        assert isinstance(head, MolmoActExpertHead)
        layerwise = serialization != "schema1"
        decoder_outputs: list[torch.Tensor] = []
        reads: list[list[CrossAttentionRead]] = [[], []]

        def record_decoder(module: nn.Module, args: tuple[object, ...], output: object) -> None:
            del module, args
            assert isinstance(output, torch.Tensor), "qualified decoder ABI must return its residual stream"
            decoder_outputs.append(output.detach().clone())

        with ExitStack() as stack:
            for layer in model.model.language_model.layers:
                stack.enter_context(layer.register_forward_hook(record_decoder))
            for block, block_reads in zip(head.action_expert.blocks, reads, strict=True):
                stack.enter_context(
                    block.cross_attn.register_forward_pre_hook(
                        _record_cross_attention(block_reads), with_kwargs=True
                    )
                )
            actual = model.sample_action(_history_stream(), num_steps=2, num_action_steps=3, action_dim=4)

        assert actual.shape == (2, 3, 4)
        assert torch.isfinite(actual).all()
        assert len(decoder_outputs) == 4  # One backbone pass, not one pass per flow step/block.
        assert all(output.shape == (2, 5, 8) for output in decoder_outputs)  # NTP drops exactly one token.
        assert not torch.allclose(decoder_outputs[1], decoder_outputs[3])
        gain = 1 + model.model.language_model.norm.weight
        final = _rms_norm(decoder_outputs[-1]) * gain
        selected = (1, 3) if hidden_norm == "final_norm" else (0, 2)
        expected_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)
        for index, block_reads in enumerate(reads):
            assert len(block_reads) == 2
            source = decoder_outputs[selected[index]] if layerwise else final
            if layerwise and hidden_norm == "final_norm":
                source = _rms_norm(source) * gain
            expected_k, expected_v = _expected_projected_kv(head, source)
            for read in block_reads:
                torch.testing.assert_close(read.key, expected_k)
                torch.testing.assert_close(read.value, expected_v)
                torch.testing.assert_close(
                    read.mask[:, 0, 0], (1 - expected_mask) * torch.finfo(torch.float32).min
                )
