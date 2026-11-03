"""CPU sampler contracts: sample-major candidates, full-width Euler and explicit noise.

Only the velocity leaf is replaced. Native context validation/projection, sampling,
masking, integration and candidate reduction run unchanged. No backbone is built.
These are component contracts, not historical end-to-end bit-parity claims.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from transformers.modeling_outputs import BaseModelOutputWithPast

from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
    ActionExpert,
    ActionExpertContext,
    DiTActionExpertHead,
    MolmoAct2ActionExpertConfig,
    MolmoActExpertHead,
    Qwen35VLAForActionGeneration,
    build_action_expert_head,
)
from lerobot.policies.perceptron_isaac.rtc import ActionExpertStepModulation
from lerobot.policies.perceptron_isaac.tensor_stream import (
    ALL_TYPES,
    Event,
    TensorStream,
    TextType,
    create_stream,
)
from tests.policies.perceptron_isaac.test_action_expert_contract import _dit_contract


@dataclass(frozen=True)
class VelocityRead:
    state: torch.Tensor
    time: torch.Tensor
    context: ActionExpertContext


class DeterministicVelocityExpert(ActionExpert):
    """Keep real context projections; use a nonzero state/time/context-dependent velocity."""

    def __init__(self, config: MolmoAct2ActionExpertConfig) -> None:
        super().__init__(config, llm_dim=8, llm_kv_dim=8, llm_num_layers=2)
        self.reads: list[VelocityRead] = []
        with torch.no_grad():
            self.context_k_proj.weight.copy_(torch.eye(8))
            self.context_v_proj.weight.copy_(torch.eye(8))

    def forward_with_context(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: ActionExpertContext,
        modulation: ActionExpertStepModulation | None = None,
    ) -> torch.Tensor:
        del modulation
        self.reads.append(VelocityRead(actions.clone(), timesteps.clone(), context))
        assert context.cross_mask is not None
        visible = (context.cross_mask[:, 0, 0] == 0).float()
        bias = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
        for index, (_, value) in enumerate(context.kv_contexts):
            bias = bias + (index + 1) * (value[:, :, 0, 0] * visible).sum(1) / visible.sum(1)
        return 0.25 * actions + timesteps[:, None, None] + bias[:, None, None]


@pytest.fixture(autouse=True)
def preserve_cpu_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        yield


def _tiny_action_config(*, layerwise: bool) -> dict[str, Any]:
    config = _dit_contract(
        action_dim=64,
        action_horizon=10,
        num_inference_steps=10,
        num_layers=2,
        hidden_dim=8,
        num_heads=2,
        timestep_embed_dim=4,
        ffn_multiple_of=4,
        context_layer_norm=False,
        qk_norm=False,
        rope=False,
    )
    if layerwise:
        config.update(
            schema_version=2,
            context_source="layer_hidden",
            context_layers=[0, 1],
            context_hidden_norm="final_norm",
        )
    return config


def _tiny_sampler(*, layerwise: bool) -> tuple[DiTActionExpertHead, DeterministicVelocityExpert]:
    head = build_action_expert_head(_tiny_action_config(layerwise=layerwise), vlm_dim=8)
    assert isinstance(head, DiTActionExpertHead)
    expert = DeterministicVelocityExpert(head.action_expert.config)
    head.action_expert = expert
    return head.float().eval(), expert


def _context_inputs(batch: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    first = torch.arange(batch * 4 * 8, dtype=torch.float32).reshape(batch, 4, 8) / 32
    second = first.flip(-1) * -0.5 + torch.arange(8, dtype=torch.float32)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)[:batch]
    return (first, second), mask


def _reference_euler(
    noise: torch.Tensor, layers: tuple[torch.Tensor, ...], mask: torch.Tensor, action_dim: int
) -> torch.Tensor:
    """Independent ten-step clean-at-one recurrence; raw channel zero survives identity projections."""
    bias = torch.zeros_like(noise[:, 0, 0], dtype=torch.float32)
    for index, layer in enumerate(layers):
        bias = bias + (index + 1) * (layer[:, :, 0] * mask).sum(1) / mask.sum(1)
    dimension_mask = (torch.arange(64) < action_dim).float()
    state = noise.float().clone() * dimension_mask
    for step in range(10):
        velocity = state * 0.25 + step / 10 + bias[:, None, None]
        state = (state + 0.1 * (velocity * dimension_mask)) * dimension_mask
    return state


@pytest.mark.parametrize("layerwise", [False, True], ids=["shared-final", "layerwise"])
@pytest.mark.parametrize(("batch", "candidates"), [(1, 1), (1, 8), (3, 8)])
@pytest.mark.parametrize("action_dim", [7, 64])
def test_candidates_use_one_velocity_batch_and_continue_the_global_rng(
    layerwise: bool, batch: int, candidates: int, action_dim: int
) -> None:
    head, expert = _tiny_sampler(layerwise=layerwise)
    layers, mask = _context_inputs(batch)
    activations = layers if layerwise else layers[-1]
    reference_layers = layers if layerwise else (layers[-1], layers[-1])
    expanded_layers = tuple(layer.repeat(candidates, 1, 1) for layer in reference_layers)
    expanded_mask = mask.repeat(candidates, 1)
    projection_inputs: list[torch.Tensor] = []

    def observe_projection(module: nn.Module, args: tuple[object, ...]) -> None:
        del module
        assert isinstance(args[0], torch.Tensor)
        projection_inputs.append(args[0].clone())

    torch.manual_seed(314159)
    initial_rng = torch.get_rng_state().clone()
    # The recovered benchmark client draws once per candidate, then batches the expert.
    # Two consecutive replans also prove that sampling does not silently reseed.
    reference_runs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(2):
        noise = torch.stack(
            [torch.randn(batch, 10, 64, device="cpu", dtype=torch.float32) for _ in range(candidates)]
        ).reshape(candidates * batch, 10, 64)
        integrated = _reference_euler(noise, expanded_layers, expanded_mask, action_dim)
        expected = integrated.reshape(candidates, batch, 10, 64).mean(0)
        reference_runs.append((expected, torch.get_rng_state().clone()))
    torch.set_rng_state(initial_rng)

    for expected, expected_rng in reference_runs:
        expert.reads.clear()
        projection_inputs.clear()
        with (
            patch.object(torch, "randn", wraps=torch.randn) as noise_draws,
            expert.context_v_proj.register_forward_pre_hook(observe_projection),
        ):
            actual = head.sample(activations, mask, action_dim=action_dim, num_flow_samples=candidates)
        assert actual.shape == (batch, 10, 64)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert len(expert.reads) == 10, "K candidates must share each Euler velocity evaluation"
        # The wrapped factory still draws real tensors; output/velocity/RNG assertions above are primary.
        assert noise_draws.call_count == candidates
        for call in noise_draws.call_args_list:
            draw_shape = call.args
            if len(draw_shape) == 1:
                draw_shape = tuple(draw_shape[0])
            assert draw_shape == (batch, 10, 64)
            assert call.kwargs["dtype"] == torch.float32
        for step, read in enumerate(expert.reads):
            assert read.state.shape == (candidates * batch, 10, 64)
            torch.testing.assert_close(read.time, torch.full((candidates * batch,), step / 10))
            assert torch.count_nonzero(read.state[..., action_dim:]) == 0
            assert read.context.cross_mask is not None
            torch.testing.assert_close(read.context.cross_mask[:, 0, 0] == 0, expanded_mask)
        expected_projection_inputs = expanded_layers if layerwise else expanded_layers[:1]
        assert len(projection_inputs) == len(expected_projection_inputs)
        for actual_input, expected_input in zip(projection_inputs, expected_projection_inputs, strict=True):
            # Expansion must precede projection, not only repeat the prepared K/V cache.
            torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
        assert torch.count_nonzero(actual[..., action_dim:]) == 0


@pytest.mark.parametrize("rank", [2, 3], ids=["broadcast-chunk", "exact-candidate-batch"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("contiguous", [False, True], ids=["strided", "contiguous"])
@pytest.mark.parametrize("candidates", [1, 8])
@pytest.mark.parametrize("action_dim", [7, 64])
def test_explicit_noise_is_coerced_without_rng_or_input_mutation(
    rank: int, dtype: torch.dtype, contiguous: bool, candidates: int, action_dim: int
) -> None:
    head, expert = _tiny_sampler(layerwise=True)
    layers, mask = _context_inputs(3)
    shape = (64, 10) if rank == 2 else (candidates * 3, 64, 10)
    # F32 contiguous inputs expose in-place mutation; strided inputs also require layout coercion.
    noise = (torch.arange(torch.Size(shape).numel(), dtype=dtype).reshape(shape) / 1000).transpose(-1, -2)
    if contiguous:
        noise = noise.contiguous()
    original = noise.clone()
    full_noise = noise.unsqueeze(0).expand(candidates * 3, -1, -1) if rank == 2 else noise
    expected = (
        _reference_euler(
            full_noise,
            tuple(layer.repeat(candidates, 1, 1) for layer in layers),
            mask.repeat(candidates, 1),
            action_dim,
        )
        .reshape(candidates, 3, 10, 64)
        .mean(0)
    )
    outputs: list[torch.Tensor] = []
    for seed in (17, 902):
        torch.manual_seed(seed)
        before = torch.get_rng_state().clone()
        actual = head.sample(layers, mask, action_dim=action_dim, num_flow_samples=candidates, noise=noise)
        assert torch.equal(torch.get_rng_state(), before), "supplied noise must consume no RNG"
        torch.testing.assert_close(noise, original, rtol=0, atol=0)
        assert actual.dtype == torch.float32
        assert actual.device == layers[0].device
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        outputs.append(actual)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert all(read.state.dtype == torch.float32 for read in expert.reads)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (640,),
        (8, 3, 10, 64),
        (10, 7),
        (10, 63),
        (9, 64),
        (3, 10, 64),
        (1, 10, 64),
        (25, 10, 64),
        (10, 24, 64),
        (24, 10, 7),
    ],
)
def test_explicit_noise_rejects_wrong_rank_padded_width_horizon_or_candidate_batch(
    shape: tuple[int, ...],
) -> None:
    head, _ = _tiny_sampler(layerwise=True)
    layers, mask = _context_inputs(3)
    noise = torch.zeros(shape)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="noise|shape|rank|width|batch"):
        head.sample(layers, mask, action_dim=7, num_flow_samples=8, noise=noise)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("layerwise", [False, True], ids=["shared-final", "layerwise"])
@pytest.mark.parametrize("batch", [1, 3])
def test_real_expert_batch8_matches_serial_explicit_noise(batch: int, layerwise: bool) -> None:
    """Run actual attention/MLP velocity layers, not the deterministic velocity replacement."""
    head = build_action_expert_head(_tiny_action_config(layerwise=layerwise), vlm_dim=8)
    assert isinstance(head, DiTActionExpertHead)
    head = head.float().eval()
    # Exercise nonzero attention, modulation and output weights rather than a zero-init velocity.
    with torch.no_grad():
        for module in head.action_expert.modules():
            if isinstance(module, nn.Linear):
                module.weight.uniform_(-0.05, 0.05)
                if module.bias is not None:
                    module.bias.uniform_(-0.05, 0.05)
    layers, mask = _context_inputs(batch)
    activations = layers if layerwise else layers[-1]
    noise = torch.linspace(-1, 1, 8 * batch * 10 * 64).reshape(8, batch, 10, 64)
    original = noise.clone()
    before = torch.get_rng_state().clone()
    joint = head.sample(activations, mask, action_dim=7, num_flow_samples=8, noise=noise.flatten(0, 1))
    serial = torch.stack(
        [head.sample(activations, mask, action_dim=7, num_flow_samples=1, noise=draw) for draw in noise]
    ).mean(0)
    torch.testing.assert_close(joint, serial, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(joint[..., :7], noise.mean(0)[..., :7])
    assert torch.count_nonzero(joint[..., 7:]) == 0
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(noise, original, rtol=0, atol=0)


class ContextOnlyBackbone(nn.Module):
    """Supply fixed post-residual tensors, not a transformer or a checkpoint."""

    def __init__(self, head: MolmoActExpertHead, layers: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self.action_expert = head
        self.hidden_layers = layers
        self.language_model = nn.Module()
        self.language_model.norm = nn.Identity()
        self.language_model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, stream: TensorStream) -> BaseModelOutputWithPast:
        assert stream.shape == (1, 5)
        # Transformers types use Torch's legacy FloatTensor alias; these fixtures are float32.
        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, self.hidden_layers[-1]),
            hidden_states=tuple(
                cast(torch.FloatTensor, tensor)
                for tensor in (torch.zeros_like(self.hidden_layers[0]), *self.hidden_layers)
            ),
        )


class SamplerModelShell(nn.Module):
    """Exercise the real sample_action method without constructing its backbone."""

    sample_action = Qwen35VLAForActionGeneration.sample_action

    def __init__(self, head: MolmoActExpertHead, layers: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self.model = ContextOnlyBackbone(head, layers)


def _prepacked_stream() -> TensorStream:
    events = [
        Event.from_text_tokens(torch.tensor([4, 5, 6]), time=(0.0, 0.0), type=TextType.text, role="user"),
        Event.from_text_tokens(torch.tensor([0]), time=(0.0, 0.0), type=TextType.action_c, role="user"),
        Event.from_text_tokens(torch.tensor([2]), time=(0.0, 0.0), type=TextType.control, role="user"),
    ]
    return TensorStream([create_stream(events, ALL_TYPES)])


def test_model_forwards_flow_noise_to_expert_noise_and_crops_only_after_sampling() -> None:
    head, expert = _tiny_sampler(layerwise=True)
    layers, _ = _context_inputs(1)
    model = SamplerModelShell(head, layers)
    noise = torch.arange(8 * 10 * 64, dtype=torch.float64).reshape(8, 10, 64) / 1000
    original = noise.clone()
    expected = _reference_euler(
        noise, tuple(layer.repeat(8, 1, 1) for layer in layers), torch.tensor([[1, 1, 1, 0]]).repeat(8, 1), 7
    ).mean(0, keepdim=True)[..., :7]
    before = torch.get_rng_state().clone()
    actual = model.sample_action(
        _prepacked_stream(), num_action_steps=10, action_dim=7, num_flow_samples=8, flow_noise=noise
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == (1, 10, 7)
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(noise, original, rtol=0, atol=0)
    assert len(expert.reads) == 10
    assert all(read.state.shape == (8, 10, 64) for read in expert.reads)


def test_model_rejects_explicit_noise_for_non_dit_head_without_rng() -> None:
    legacy_head = MolmoActExpertHead(_tiny_action_config(layerwise=False), vlm_dim=8).float().eval()
    layers, _ = _context_inputs(1)
    model = SamplerModelShell(legacy_head, layers)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="flow_noise.*DiT"):
        model.sample_action(_prepacked_stream(), flow_noise=torch.zeros(10, 64))
    assert torch.equal(torch.get_rng_state(), before)
