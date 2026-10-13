from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen35_modeling
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)

from lerobot.policies.perceptron_isaac.modeling_qwen36_moe import (
    DETERMINISTIC_ROUTE_REDUCTION,
    GENESIS_ROTARY_PRECISION,
    GenesisNullMoeContract,
    GenesisNullSparseMoeBlock,
    GenesisNullTopKRouter,
    GenesisQwen36Attention,
    GenesisQwen36DecoderLayer,
    GenesisQwen36GatedDeltaNet,
    GenesisQwen36Model,
    GenesisQwen36TextModel,
    GenesisQwen36TextRotaryEmbedding,
    GenesisQwen36VisionAttention,
    GenesisQwen36VisionModel,
    Mk1Qwen36MoeConfig,
    deterministic_token_segment_sum,
)


def _metadata(
    *,
    num_real_experts: int = 2,
    num_null_experts: int = 2,
    top_k: int = 2,
    route_scale: float = 1.0,
) -> dict[str, object]:
    return {
        "router_contract_version": 1,
        "router_contract": [num_real_experts, num_null_experts],
        "num_real_experts": num_real_experts,
        "num_null_experts": num_null_experts,
        "physical_router_outputs": num_real_experts + 1,
        "logical_router_outputs": num_real_experts + num_null_experts,
        "shared_null_router_row": True,
        "top_k": top_k,
        "score_func": "softmax",
        "route_norm": True,
        "route_scale": route_scale,
        "score_before_experts": False,
        "null_expert_semantics": "skip_compute_renormalize_real_routes",
        "shared_expert_mode": "sigmoid_gated_additive",
        "uses_expert_bias": False,
    }


def _text_config(
    *,
    num_real_experts: int = 2,
    num_null_experts: int = 2,
    top_k: int = 2,
    route_scale: float = 1.0,
    layer_types: list[str] | None = None,
    full_attention_interval: int | None = None,
    moe_intermediate_size: int = 6,
    attention_implementation: str = "eager",
) -> Qwen3_5MoeTextConfig:
    layer_types = layer_types or ["full_attention"]
    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=24,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=64,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=12,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=moe_intermediate_size,
        num_experts_per_tok=top_k,
        num_experts=num_real_experts,
        layer_types=layer_types,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 1],
        },
        genesis_moe=_metadata(
            num_real_experts=num_real_experts,
            num_null_experts=num_null_experts,
            top_k=top_k,
            route_scale=route_scale,
        ),
    )
    config._attn_implementation = attention_implementation
    if full_attention_interval is not None:
        config.full_attention_interval = full_attention_interval
    return config


def _vision_config(*, depth: int = 1, num_position_embeddings: int | None = None) -> Qwen3_5MoeVisionConfig:
    kwargs = {}
    if num_position_embeddings is not None:
        kwargs["num_position_embeddings"] = num_position_embeddings
    return Qwen3_5MoeVisionConfig(
        depth=depth,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
        out_hidden_size=24,
        **kwargs,
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_deterministic_route_reducer_is_bitwise_stable_and_preserves_ordered_sum(device: str) -> None:
    base = torch.tensor([[1.0, -1.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.bfloat16, device=device)
    token_indices = torch.tensor([1, 0, 1, 1, 0], dtype=torch.long, device=device)
    routes = torch.tensor(
        [[0.5, 1.0], [2.0, -2.0], [4.0, 8.0], [-1.0, 0.25], [0.5, 3.0]],
        dtype=torch.bfloat16,
        device=device,
    )
    expected = torch.tensor([[3.5, 0.0], [5.5, 12.25], [4.0, 5.0]], dtype=torch.bfloat16, device=device)

    outputs = [deterministic_token_segment_sum(base, token_indices, routes) for _ in range(20)]

    assert torch.equal(outputs[0], expected)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


def test_null_moe_defaults_to_qualified_stable_route_reduction() -> None:
    config = _text_config()
    block = GenesisNullSparseMoeBlock(config)

    assert block.deterministic_route_reduction is True
    config._mk1_route_reduction = "repeated_index_add"
    with pytest.raises(ValueError, match="Unsupported MK1 route reduction"):
        GenesisNullSparseMoeBlock(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped GEMM required")
def test_deterministic_null_moe_block_is_bitwise_stable_on_cuda() -> None:
    torch.manual_seed(17)
    block = GenesisNullSparseMoeBlock(
        _text_config(
            num_real_experts=4,
            num_null_experts=4,
            top_k=4,
            moe_intermediate_size=16,
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-0.02, 0.02)
        block.gate.weight.zero_()
    block.deterministic_route_reduction = True
    block.eval()
    x = torch.linspace(-1.0, 1.0, steps=5 * 24, device="cuda", dtype=torch.bfloat16).reshape(1, 5, 24)

    with torch.inference_mode():
        outputs = [block(x) for _ in range(10)]

    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("physical_router_outputs", 4, "physical_router_outputs"),
        ("logical_router_outputs", 5, "logical_router_outputs"),
        ("score_func", "sigmoid", "score_func"),
        ("route_norm", False, "route_norm"),
        ("score_before_experts", True, "score_before_experts"),
        ("shared_expert_mode", "ungated_additive", "shared_expert_mode"),
        ("uses_expert_bias", True, "uses_expert_bias"),
    ],
)
def test_contract_rejects_unsupported_metadata(key: str, value: object, message: str) -> None:
    metadata = _metadata()
    metadata[key] = value

    with pytest.raises(ValueError, match=message):
        GenesisNullMoeContract.from_metadata(metadata)


def test_mk1_config_requires_identical_root_and_text_contracts() -> None:
    text_config = _text_config()
    vision_config = _vision_config()
    config = Mk1Qwen36MoeConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        genesis_moe=copy.deepcopy(text_config.genesis_moe),
    )
    assert config.text_config.genesis_moe == config.genesis_moe

    mismatched = copy.deepcopy(text_config.genesis_moe)
    mismatched["route_scale"] = 0.5
    with pytest.raises(ValueError, match="must be identical"):
        Mk1Qwen36MoeConfig(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            genesis_moe=mismatched,
        )


def test_text_rotary_matches_genesis_model_dtype_inv_freq_contract() -> None:
    config = _text_config()
    rotary = GenesisQwen36TextRotaryEmbedding(config).to(dtype=torch.bfloat16)
    hidden_states = torch.zeros(1, 4, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.tensor(
        [
            [[0, 1, 13, 29]],
            [[0, 2, 17, 31]],
            [[0, 3, 19, 37]],
        ]
    )

    cos, sin = rotary(hidden_states, position_ids)

    rotary_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
    theta = float(config.rope_parameters["rope_theta"])
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    quantized_inv_freq = inv_freq.to(torch.bfloat16).float()
    phases = position_ids.float().unsqueeze(-1) * quantized_inv_freq.view(1, 1, 1, -1)
    interleaved = phases[0].clone()
    for axis, offset in enumerate((1, 2), start=1):
        length = config.rope_parameters["mrope_section"][axis] * 3
        interleaved[..., slice(offset, length, 3)] = phases[axis, ..., slice(offset, length, 3)]
    embedding = torch.cat((interleaved, interleaved), dim=-1)

    assert rotary.precision_contract == GENESIS_ROTARY_PRECISION
    assert rotary.inv_freq.dtype == torch.bfloat16
    assert rotary.state_dict() == {}
    torch.testing.assert_close(cos, embedding.cos().to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(sin, embedding.sin().to(torch.bfloat16), rtol=0, atol=0)


def test_text_attention_matches_genesis_explicit_kv_sdpa_oracle(monkeypatch) -> None:
    torch.manual_seed(13)
    config = _text_config()
    attention = GenesisQwen36Attention(config, layer_idx=0).eval()
    hidden_states = torch.randn(1, 4, config.hidden_size)
    rotary_dim = config.head_dim // 2
    position_embeddings = (
        torch.ones(1, 4, rotary_dim),
        torch.zeros(1, 4, rotary_dim),
    )

    with torch.no_grad():
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query, gate = torch.chunk(
            attention.q_proj(hidden_states).view(*input_shape, -1, attention.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query = attention.q_norm(query.view(hidden_shape))
        key = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape))
        value = attention.v_proj(hidden_states).view(hidden_shape)
        query, key = qwen35_modeling.apply_rotary_pos_emb(
            query,
            key,
            *position_embeddings,
            unsqueeze_dim=2,
        )
        key = (
            key.unsqueeze(3)
            .expand(1, 4, config.num_key_value_heads, attention.num_key_value_groups, 12)
            .reshape(1, 4, config.num_attention_heads, 12)
            .transpose(1, 2)
        )
        value = (
            value.unsqueeze(3)
            .expand(1, 4, config.num_key_value_heads, attention.num_key_value_groups, 12)
            .reshape(1, 4, config.num_attention_heads, 12)
            .transpose(1, 2)
        )
        query = query.transpose(1, 2)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            scale=attention.scaling,
            is_causal=True,
        )
        expected = expected.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        expected = attention.o_proj(expected * torch.sigmoid(gate))

    calls = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def recording_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
    traces = []
    attention.trace_observer = traces.append
    with torch.no_grad():
        actual, weights = attention(hidden_states, position_embeddings, None)

    assert weights is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 1
    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape[1] == key_shape[1] == value_shape[1] == config.num_attention_heads
    assert kwargs["is_causal"] is True
    assert kwargs["attn_mask"] is None
    assert "enable_gqa" not in kwargs
    assert len(traces) == 1
    trace = traces[0]
    assert trace.query_projection.shape == (1, 4, 2, 12)
    assert trace.query_gate.shape == (1, 4, 24)
    assert trace.key_projection.shape == (1, 4, 1, 12)
    assert trace.rotary_query.shape == (1, 4, 2, 12)
    assert trace.sdpa_query.shape == trace.repeated_key.shape == (1, 2, 4, 12)
    assert trace.repeated_value.shape == (1, 2, 4, 12)
    torch.testing.assert_close(trace.token_mixer, actual, rtol=0, atol=0)


def test_packed_vision_attention_matches_genesis_sdpa_oracle(monkeypatch) -> None:
    torch.manual_seed(14)
    config = _vision_config()
    attention = GenesisQwen36VisionAttention(config).eval()
    hidden_states = torch.randn(5, config.hidden_size)
    position_embeddings = (
        torch.ones(5, attention.head_dim),
        torch.zeros(5, attention.head_dim),
    )
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    with torch.no_grad():
        query, key, value = (
            attention.qkv(hidden_states).reshape(5, 3, attention.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query, key = qwen35_modeling.apply_rotary_pos_emb_vision(query, key, *position_embeddings)
        outputs = []
        for start, end in ((0, 2), (2, 5)):
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query[start:end].transpose(0, 1).unsqueeze(0),
                    key[start:end].transpose(0, 1).unsqueeze(0),
                    value[start:end].transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                    scale=attention.scaling,
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        expected = attention.proj(torch.cat(outputs, dim=0).reshape(5, -1).contiguous())

    calls = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def recording_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording_sdpa)
    with torch.no_grad():
        actual = attention(hidden_states, cu_seqlens, position_embeddings=position_embeddings)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [call[0][2] for call in calls] == [2, 3]
    assert all(call[0][1] == call[1][1] == call[2][1] == config.num_heads for call in calls)
    assert all(call[3]["is_causal"] is False and call[3]["attn_mask"] is None for call in calls)
    assert all("enable_gqa" not in call[3] for call in calls)


def test_vision_position_interpolation_uses_genesis_bf16_reduction() -> None:
    torch.manual_seed(19)
    config = _vision_config(num_position_embeddings=64)
    model = GenesisQwen36VisionModel(config).to(dtype=torch.bfloat16).eval()
    grid = torch.tensor([[1, 16, 16]], dtype=torch.long)

    height_indices = torch.linspace(0, model.num_grid_per_side - 1, 16)
    width_indices = torch.linspace(0, model.num_grid_per_side - 1, 16)
    height_floor = height_indices.int()
    width_floor = width_indices.int()
    height_ceil = (height_floor + 1).clip(max=model.num_grid_per_side - 1)
    width_ceil = (width_floor + 1).clip(max=model.num_grid_per_side - 1)
    height_delta = height_indices - height_floor
    width_delta = width_indices - width_floor
    base_height = height_floor * model.num_grid_per_side
    base_height_ceil = height_ceil * model.num_grid_per_side
    indices = torch.stack(
        [
            (base_height[:, None] + width_floor[None, :]).flatten(),
            (base_height[:, None] + width_ceil[None, :]).flatten(),
            (base_height_ceil[:, None] + width_floor[None, :]).flatten(),
            (base_height_ceil[:, None] + width_ceil[None, :]).flatten(),
        ]
    )
    weights = torch.stack(
        [
            ((1 - height_delta)[:, None] * (1 - width_delta)[None, :]).flatten(),
            ((1 - height_delta)[:, None] * width_delta[None, :]).flatten(),
            (height_delta[:, None] * (1 - width_delta)[None, :]).flatten(),
            (height_delta[:, None] * width_delta[None, :]).flatten(),
        ]
    ).to(torch.bfloat16)
    interpolated = model.pos_embed(indices) * weights[:, :, None]
    genesis_base = interpolated.sum(dim=0)
    chained_base = interpolated[0] + interpolated[1] + interpolated[2] + interpolated[3]
    expected = genesis_base.view(1, 8, 2, 8, 2, config.hidden_size).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)

    actual = model.fast_pos_embed_interpolate(grid)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(genesis_base, chained_base)


def test_vision_position_interpolation_preserves_mixed_grid_order_and_state_abi() -> None:
    config = _vision_config(num_position_embeddings=64)
    stock = qwen35_modeling.Qwen3_5MoeVisionModel(copy.deepcopy(config))
    custom = GenesisQwen36VisionModel(copy.deepcopy(config))
    custom.load_state_dict(stock.state_dict())
    mixed_grid = torch.tensor([[1, 4, 6], [2, 8, 4]], dtype=torch.long)

    mixed = custom.fast_pos_embed_interpolate(mixed_grid)
    separate = torch.cat(
        [custom.fast_pos_embed_interpolate(row.unsqueeze(0)) for row in mixed_grid],
        dim=0,
    )

    torch.testing.assert_close(mixed, separate, rtol=0, atol=0)
    assert mixed.shape == (4 * 6 + 2 * 8 * 4, config.hidden_size)
    assert set(custom.state_dict()) == set(stock.state_dict())
    assert [(name, value.dtype) for name, value in custom.named_buffers()] == [
        (name, value.dtype) for name, value in stock.named_buffers()
    ]


def test_multimodal_model_installs_explicit_sdpa_for_text_and_vision() -> None:
    text_config = _text_config(full_attention_interval=1)
    vision_config = _vision_config(depth=2)
    config = Mk1Qwen36MoeConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        genesis_moe=copy.deepcopy(text_config.genesis_moe),
    )

    model = GenesisQwen36Model(config)

    assert isinstance(model.language_model.layers[0].self_attn, GenesisQwen36Attention)
    assert isinstance(model.visual, GenesisQwen36VisionModel)
    assert all(isinstance(block.attn, GenesisQwen36VisionAttention) for block in model.visual.blocks)
    assert all(layer.mlp.deterministic_route_reduction for layer in model.language_model.layers)
    assert DETERMINISTIC_ROUTE_REDUCTION == "stable_token_segment_sum_v1"


def test_router_uses_compact_state_and_stable_real_first_ties() -> None:
    router = GenesisNullTopKRouter(_text_config())

    assert router.weight.shape == (3, 24)
    assert set(router.state_dict()) == {"weight"}
    torch.testing.assert_close(router._router_contract, torch.tensor([2, 2]))

    with torch.no_grad():
        router.weight.zero_()
    tied = router.route(torch.ones(1, 24))
    torch.testing.assert_close(tied.selected_experts, torch.tensor([[0, 1]]))


def test_router_expands_shared_null_logit_and_renormalizes_real_routes() -> None:
    router = GenesisNullTopKRouter(_text_config())
    with torch.no_grad():
        router.weight.zero_()
        router.weight[:, 0] = torch.tensor([2.0, -10.0, 1.0])

    hidden_states = torch.zeros(1, 24)
    hidden_states[:, 0] = 1
    routing = router.route(hidden_states)

    expected_probabilities = torch.softmax(torch.tensor([[2.0, -10.0, 1.0, 1.0]]), dim=-1)
    torch.testing.assert_close(routing.logical_probabilities, expected_probabilities)
    torch.testing.assert_close(routing.selected_experts, torch.tensor([[0, 2]]))
    torch.testing.assert_close(routing.real_route_weights, torch.tensor([[1.0, 0.0]]))

    with torch.no_grad():
        router.weight.zero_()
        router.weight[2, 0] = 10
    all_null = router.route(hidden_states)
    torch.testing.assert_close(all_null.selected_experts, torch.tensor([[2, 3]]))
    torch.testing.assert_close(all_null.real_route_weights, torch.zeros(1, 2))


def test_production_router_cardinality_is_257_physical_to_512_logical_top8() -> None:
    router = GenesisNullTopKRouter(_text_config(num_real_experts=256, num_null_experts=256, top_k=8))
    assert router.weight.shape == (257, 24)

    with torch.no_grad():
        router.weight.zero_()
        router.weight[256, 0] = 10
    hidden_states = torch.zeros(1, 24)
    hidden_states[:, 0] = 1
    routing = router.route(hidden_states)

    assert routing.logical_probabilities.shape == (1, 512)
    torch.testing.assert_close(routing.selected_experts, torch.arange(256, 264).reshape(1, 8))
    torch.testing.assert_close(routing.real_route_weights, torch.zeros(1, 8))


@pytest.mark.parametrize("case", ["random", "all_tied", "null_dominant", "nan"])
def test_compact_stable_topk_matches_expanded_oracle(case: str) -> None:
    torch.manual_seed(17)
    router = GenesisNullTopKRouter(_text_config(num_real_experts=256, num_null_experts=256, top_k=8))
    with torch.no_grad():
        if case == "random":
            router.weight.normal_()
        else:
            router.weight.zero_()
        if case == "null_dominant":
            router.weight[256, 0] = 10
        elif case == "nan":
            router.weight[256, 0] = torch.nan

    hidden_states = torch.randn(11, 24)
    hidden_states[:, 0] = 1
    routing = router.route(hidden_states)

    expected = router._expanded_stable_topk(routing.logical_probabilities)  # noqa: SLF001
    torch.testing.assert_close(routing.selected_experts, expected)


def test_shared_null_router_gradient_matches_independent_logical_rows() -> None:
    config = _text_config(num_real_experts=2, num_null_experts=2, top_k=3)
    router = GenesisNullTopKRouter(config).double()
    physical_weight = torch.zeros(3, 24, dtype=torch.float64)
    physical_weight[:, :2] = torch.tensor([[1.2, -0.3], [-0.4, 0.9], [0.25, -0.7]], dtype=torch.float64)
    with torch.no_grad():
        router.weight.copy_(physical_weight)

    actual_input = torch.zeros(2, 24, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        actual_input[:, :2] = torch.tensor([[0.8, -0.2], [-0.5, 1.1]], dtype=torch.float64)
    actual = router.route(actual_input)
    probability_coefficients = torch.tensor(
        [[0.1, -0.3, 0.8, -0.6], [-0.7, 0.2, 0.4, 0.9]],
        dtype=torch.float64,
    )
    route_coefficients = torch.tensor([[0.7, -0.2, 1.3], [-0.4, 0.9, 0.5]], dtype=torch.float64)
    actual_loss = (actual.logical_probabilities * probability_coefficients).sum()
    actual_loss = actual_loss + (actual.real_route_weights * route_coefficients).sum()
    actual_loss.backward()

    reference_input = actual_input.detach().clone().requires_grad_()
    reference_real_weight = physical_weight[:2].clone().requires_grad_()
    reference_null_weights = physical_weight[2:].expand(2, -1).clone().requires_grad_()
    reference_logits = torch.cat(
        [reference_input @ reference_real_weight.T, reference_input @ reference_null_weights.T],
        dim=-1,
    )
    reference_probabilities = torch.softmax(reference_logits.float(), dim=-1)
    reference_ids = torch.argsort(reference_probabilities, dim=-1, descending=True, stable=True)[:, :3]
    reference_selected = reference_probabilities.gather(-1, reference_ids)
    reference_real = torch.where(reference_ids < 2, reference_selected, torch.zeros_like(reference_selected))
    reference_mass = reference_real.sum(-1, keepdim=True)
    reference_weights = reference_real / torch.where(
        reference_mass > 0, reference_mass, torch.ones_like(reference_mass)
    )
    reference_loss = (reference_probabilities * probability_coefficients).sum()
    reference_loss = reference_loss + (reference_weights * route_coefficients).sum()
    reference_loss.backward()

    torch.testing.assert_close(actual.selected_experts, reference_ids)
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    assert router.weight.grad is not None
    assert reference_real_weight.grad is not None
    assert reference_null_weights.grad is not None
    torch.testing.assert_close(router.weight.grad[:2], reference_real_weight.grad)
    torch.testing.assert_close(router.weight.grad[2], reference_null_weights.grad.sum(dim=0))


def test_all_null_routes_skip_nan_experts_but_keep_shared_expert_active() -> None:
    block = GenesisNullSparseMoeBlock(_text_config())
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.weight[2, 0] = 10
        block.experts.gate_up_proj.fill_(torch.nan)
        block.experts.down_proj.fill_(torch.nan)
        for parameter in block.shared_expert.parameters():
            parameter.fill_(0.1)
        block.shared_expert_gate.weight.zero_()

    hidden_states = torch.zeros(1, 2, 24)
    hidden_states[..., 0] = 1
    branches = block.forward_with_branches(hidden_states)

    torch.testing.assert_close(branches.routed_output, torch.zeros_like(branches.routed_output))
    torch.testing.assert_close(branches.output, branches.shared_expert_output)
    assert torch.isfinite(branches.output).all()
    assert torch.count_nonzero(branches.shared_expert_output) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_route_score_is_applied_after_swiglu_before_down_projection(dtype: torch.dtype) -> None:
    device = torch.device("cuda" if dtype is torch.bfloat16 and torch.cuda.is_available() else "cpu")
    block = GenesisNullSparseMoeBlock(_text_config(route_scale=0.5)).to(device=device, dtype=dtype)
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.weight[:, 0] = torch.tensor([2.0, -10.0, 1.0], device=device, dtype=dtype)
        block.experts.gate_up_proj.zero_()
        block.experts.gate_up_proj[0, :, 0] = torch.arange(1, 13, device=device, dtype=dtype) / 12
        block.experts.down_proj.zero_()
        block.experts.down_proj[0, :, :] = (
            torch.arange(1, 145, device=device, dtype=dtype).reshape(24, 6) / 144
        )
        for parameter in block.shared_expert.parameters():
            parameter.zero_()
        block.shared_expert_gate.weight.zero_()

    hidden_states = torch.zeros(1, 1, 24, device=device, dtype=dtype)
    hidden_states[..., 0] = 1
    branches = block.forward_with_branches(hidden_states)

    flattened = hidden_states.reshape(1, 24)
    gate, up = torch.nn.functional.linear(flattened, block.experts.gate_up_proj[0]).chunk(2, dim=-1)
    swiglu = block.experts.act_fn(gate) * up
    expected = torch.nn.functional.linear((swiglu.float() * 0.5).to(dtype), block.experts.down_proj[0])
    torch.testing.assert_close(branches.routed_output.reshape(1, 24), expected, rtol=0, atol=0)


def test_null_moe_backward_has_finite_router_expert_and_shared_gradients() -> None:
    block = GenesisNullSparseMoeBlock(_text_config(num_real_experts=3, num_null_experts=2, top_k=4))
    with torch.no_grad():
        for parameter in block.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        block.gate.weight.zero_()
        block.gate.weight[:, 0] = torch.tensor([3.0, 2.0, -10.0, 2.5])

    hidden_states = torch.randn(2, 2, 24, requires_grad=True)
    hidden_states.data[..., 0].abs_().add_(0.5)
    branches = block.forward_with_branches(hidden_states)
    loss = branches.output.square().mean() + 0.01 * branches.routing.logical_probabilities.square().mean()
    loss.backward()

    gradients = [
        hidden_states.grad,
        block.gate.weight.grad,
        block.experts.gate_up_proj.grad,
        block.experts.down_proj.grad,
        block.shared_expert.gate_proj.weight.grad,
        block.shared_expert_gate.weight.grad,
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_compact_routes_match_genesis_two_stage_real_renormalization() -> None:
    block = GenesisNullSparseMoeBlock(
        _text_config(
            num_real_experts=3,
            num_null_experts=2,
            top_k=4,
            route_scale=0.7,
        )
    )
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.weight[:, 0] = torch.tensor([3.0, 2.0, 1.0, 2.5])
    hidden_states = torch.zeros(3, 24)
    hidden_states[:, 0] = torch.tensor([1.0, -1.0, 0.5])
    routing = block.gate.route(hidden_states)

    routes = block._compact_real_routes(hidden_states, routing)  # noqa: SLF001

    selected_scores = routing.logical_probabilities.gather(-1, routing.selected_experts)
    selected_scores = selected_scores / (selected_scores.sum(-1, keepdim=True) + 1e-20) * 0.7
    flat_experts = routing.selected_experts.flatten()
    flat_tokens = torch.arange(3).unsqueeze(1).expand(-1, 4).flatten()
    flat_scores = selected_scores.flatten()
    logical_order = torch.argsort(flat_experts, stable=True)
    real_order = logical_order[flat_experts[logical_order] < 3]
    expected_experts = flat_experts[real_order]
    expected_tokens = flat_tokens[real_order]
    unscaled_real_scores = flat_scores[real_order] / 0.7
    real_mass = torch.zeros(3)
    real_mass.index_add_(0, expected_tokens, unscaled_real_scores)
    expected_scores = unscaled_real_scores / real_mass.clamp_min(1e-6)[expected_tokens] * 0.7

    torch.testing.assert_close(routes.expert_indices, expected_experts)
    torch.testing.assert_close(routes.token_indices, expected_tokens)
    torch.testing.assert_close(routes.expert_scores, expected_scores, rtol=0, atol=0)
    expected_offsets = torch.bincount(expected_experts, minlength=3).cumsum(0, dtype=torch.int32)
    torch.testing.assert_close(routes.offsets, expected_offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped_mm parity requires a GPU")
def test_grouped_mm_compact_dispatch_matches_eager_oracle() -> None:
    torch.manual_seed(23)
    block = GenesisNullSparseMoeBlock(
        _text_config(
            num_real_experts=8,
            num_null_experts=8,
            top_k=4,
            moe_intermediate_size=8,
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(std=0.05)

    hidden_states = torch.randn(3, 7, 24, device="cuda", dtype=torch.bfloat16)
    flattened = hidden_states.reshape(-1, 24)
    routing = block.gate.route(flattened)
    shared_output = block._shared_expert_output(flattened)  # noqa: SLF001
    routes = block._compact_real_routes(flattened, routing)  # noqa: SLF001

    assert block.dispatch_backend(flattened) == "grouped_mm"
    eager_routed, eager_combined = block._dispatch_real_experts_eager(  # noqa: SLF001
        flattened, routes, shared_output
    )
    grouped_routed, grouped_combined = block._dispatch_real_experts_grouped_mm(  # noqa: SLF001
        flattened, routes, shared_output
    )

    torch.testing.assert_close(grouped_routed, eager_routed, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(grouped_combined, eager_combined, rtol=2e-2, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA all-null dispatch requires a GPU")
def test_grouped_backend_all_null_routes_never_reads_nan_experts() -> None:
    block = GenesisNullSparseMoeBlock(
        _text_config(
            num_real_experts=8,
            num_null_experts=8,
            top_k=4,
            moe_intermediate_size=8,
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.weight[8, 0] = 10
        block.experts.gate_up_proj.fill_(torch.nan)
        block.experts.down_proj.fill_(torch.nan)
        for parameter in block.shared_expert.parameters():
            parameter.fill_(0.1)
        block.shared_expert_gate.weight.zero_()

    hidden_states = torch.zeros(1, 2, 24, device="cuda", dtype=torch.bfloat16)
    hidden_states[..., 0] = 1
    assert block.dispatch_backend(hidden_states.reshape(-1, 24)) == "grouped_mm"
    branches = block.forward_with_branches(hidden_states)

    torch.testing.assert_close(branches.routed_output, torch.zeros_like(branches.routed_output))
    torch.testing.assert_close(branches.output, branches.shared_expert_output)
    assert torch.isfinite(branches.output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped_mm gradients require a GPU")
def test_grouped_mm_compact_dispatch_backward_is_finite() -> None:
    torch.manual_seed(29)
    block = GenesisNullSparseMoeBlock(
        _text_config(
            num_real_experts=8,
            num_null_experts=8,
            top_k=4,
            moe_intermediate_size=8,
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(std=0.05)

    hidden_states = torch.randn(2, 5, 24, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert block.dispatch_backend(hidden_states.reshape(-1, 24)) == "grouped_mm"

    loss = block(hidden_states).float().square().mean()
    loss.backward()

    gradients = [
        hidden_states.grad,
        block.gate.weight.grad,
        block.experts.gate_up_proj.grad,
        block.experts.down_proj.grad,
        block.shared_expert.gate_proj.weight.grad,
        block.shared_expert_gate.weight.grad,
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def _production_dispatch_fixture(
    activation_dtype: torch.dtype,
    gate_up_dtype: torch.dtype = torch.float32,
    down_dtype: torch.dtype = torch.float32,
    expert_device: str = "cuda:0",
) -> tuple[GenesisNullSparseMoeBlock, torch.Tensor]:
    # Allocate only fake CPU tensors, then model CUDA metadata without initializing CUDA.
    with FakeTensorMode():
        block = GenesisNullSparseMoeBlock(
            _text_config(
                num_real_experts=256,
                num_null_experts=256,
                top_k=8,
                moe_intermediate_size=8,
            )
        )
        block.experts.gate_up_proj = torch.nn.Parameter(block.experts.gate_up_proj.to(gate_up_dtype))
        block.experts.down_proj = torch.nn.Parameter(block.experts.down_proj.to(down_dtype))
        hidden_states = torch.empty(1, 24, dtype=activation_dtype)
    for parameter in block.parameters():
        assert isinstance(parameter, FakeTensor)
        parameter.fake_device = torch.device(expert_device)
    assert isinstance(hidden_states, FakeTensor)
    hidden_states.fake_device = torch.device("cuda:0")
    return block, hidden_states


@pytest.mark.parametrize("grad_enabled", [True, False], ids=["grad", "no-grad"])
@pytest.mark.parametrize(
    "activation_dtype,autocast_enabled",
    [
        pytest.param(torch.float32, False, id="fp32"),
        pytest.param(torch.bfloat16, True, id="bf16"),
        pytest.param(torch.float32, True, id="fp32-bf16-autocast"),
    ],
)
def test_production_dispatch_fp32_training_uses_eager(
    activation_dtype: torch.dtype, autocast_enabled: bool, grad_enabled: bool
) -> None:
    block, hidden_states = _production_dispatch_fixture(activation_dtype)
    block.train()
    with (
        torch.set_grad_enabled(grad_enabled),
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.is_autocast_enabled", return_value=autocast_enabled),
        patch("torch.get_autocast_dtype", return_value=torch.bfloat16),
    ):
        assert block.dispatch_backend(hidden_states) == "eager"


@pytest.mark.parametrize("grad_enabled", [True, False], ids=["grad", "no-grad"])
@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_production_dispatch_fp32_eval_stays_closed(
    activation_dtype: torch.dtype, grad_enabled: bool
) -> None:
    block, hidden_states = _production_dispatch_fixture(activation_dtype)
    block.eval()
    with (
        torch.set_grad_enabled(grad_enabled),
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.is_autocast_enabled", return_value=True),
        patch("torch.get_autocast_dtype", return_value=torch.bfloat16),
        pytest.raises(RuntimeError, match="requires BF16 torch grouped_mm"),
    ):
        block.dispatch_backend(hidden_states)


@pytest.mark.parametrize(
    "activation_dtype,gate_up_dtype,down_dtype,expert_device,autocast_enabled,autocast_dtype",
    [
        pytest.param(
            torch.float16, torch.float32, torch.float32, "cuda:0", True, torch.float16, id="fp16-input"
        ),
        pytest.param(
            torch.float32, torch.float16, torch.float16, "cuda:0", False, torch.bfloat16, id="fp16-weights"
        ),
        pytest.param(
            torch.float32, torch.bfloat16, torch.float32, "cuda:0", False, torch.bfloat16, id="bf16-gate-up"
        ),
        pytest.param(
            torch.float32, torch.float32, torch.bfloat16, "cuda:0", False, torch.bfloat16, id="bf16-down"
        ),
        pytest.param(
            torch.float32, torch.float32, torch.float32, "cpu", False, torch.bfloat16, id="cpu-weights"
        ),
        pytest.param(
            torch.float32, torch.float32, torch.float32, "cuda:1", False, torch.bfloat16, id="other-device"
        ),
        pytest.param(
            torch.bfloat16, torch.float32, torch.float32, "cuda:0", False, torch.bfloat16, id="no-autocast"
        ),
        pytest.param(
            torch.bfloat16, torch.float32, torch.float32, "cuda:0", True, torch.float16, id="fp16-autocast"
        ),
    ],
)
def test_production_dispatch_unsupported_training_stays_closed(
    activation_dtype: torch.dtype,
    gate_up_dtype: torch.dtype,
    down_dtype: torch.dtype,
    expert_device: str,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> None:
    block, hidden_states = _production_dispatch_fixture(
        activation_dtype, gate_up_dtype, down_dtype, expert_device
    )
    block.train()
    with (
        patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        patch("torch.is_autocast_enabled", return_value=autocast_enabled),
        patch("torch.get_autocast_dtype", return_value=autocast_dtype),
        pytest.raises(RuntimeError, match="requires BF16 torch grouped_mm"),
    ):
        block.dispatch_backend(hidden_states)


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("capability", [(8, 0), (7, 0)], ids=["sm80", "sm70"])
def test_production_dispatch_bf16_preserves_grouped_mm(training: bool, capability: tuple[int, int]) -> None:
    block, hidden_states = _production_dispatch_fixture(torch.bfloat16, torch.bfloat16, torch.bfloat16)
    block.train(training)
    with patch("torch.cuda.get_device_capability", return_value=capability):
        if capability[0] >= 8:
            assert block.dispatch_backend(hidden_states) == "grouped_mm"
        else:
            with pytest.raises(RuntimeError, match="requires BF16 torch grouped_mm"):
                block.dispatch_backend(hidden_states)


@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("deterministic", [True, False], ids=["segment", "index-add"])
@pytest.mark.parametrize("real_routes", [2, 1, 0], ids=["real", "mixed", "null"])
def test_fp32_eager_experts_accept_cpu_bf16_autocast_forward(
    activation_dtype: torch.dtype, deterministic: bool, real_routes: int
) -> None:
    block = GenesisNullSparseMoeBlock(_text_config()).train()
    block.deterministic_route_reduction = deterministic
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.fill_(0.1)
        block.shared_expert_gate.weight.zero_()
        if real_routes != 2:
            block.gate.weight.zero_()
            block.gate.weight[:, 0] = torch.tensor([2.0, -10.0, 1.0])
        if real_routes == 0:
            block.gate.weight[2, 0] = 10
            block.experts.gate_up_proj.fill_(torch.nan)
            block.experts.down_proj.fill_(torch.nan)
    hidden_states = torch.ones(1, 2, 24, dtype=activation_dtype)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        branches = block.forward_with_branches(hidden_states)
        # The zero shared gate gives exactly sigmoid(0) = 0.5.
        expected_shared = block.shared_expert(hidden_states) * 0.5
        if real_routes:
            # Both real experts have identical weights; only the route score differs.
            gate, up = torch.nn.functional.linear(hidden_states, block.experts.gate_up_proj[0]).chunk(
                2, dim=-1
            )
            swiglu = block.experts.act_fn(gate) * up
            weighted_swiglu = (swiglu.float() / real_routes).to(swiglu.dtype)
            expected_row = torch.nn.functional.linear(weighted_swiglu, block.experts.down_proj[0])

    assert branches.output.dtype is activation_dtype
    assert branches.routed_output.dtype is activation_dtype
    assert branches.shared_expert_output.dtype is torch.bfloat16
    assert torch.isfinite(branches.output).all()
    assert all(parameter.dtype is torch.float32 for parameter in block.parameters())
    selected_real = branches.routing.selected_experts < block.contract.num_real_experts
    assert torch.all(selected_real.sum(dim=-1) == real_routes)
    expected_weights = selected_real.float() / max(real_routes, 1)
    torch.testing.assert_close(branches.routing.real_route_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(branches.shared_expert_output, expected_shared, rtol=0, atol=0)
    assert torch.count_nonzero(branches.shared_expert_output) > 0
    assert branches.output.data_ptr() != branches.shared_expert_output.data_ptr()

    expected_routed = torch.zeros_like(hidden_states)
    expected_output = expected_shared.to(activation_dtype, copy=True)
    for _ in range(real_routes):
        expected_routed = expected_routed + expected_row.to(activation_dtype)
        if not deterministic:
            # Preserve direct scatter into the shared base, including BF16 rounding.
            expected_output = expected_output + expected_row.to(activation_dtype)
    if deterministic:
        expected_output = expected_output + expected_routed
    torch.testing.assert_close(branches.routed_output, expected_routed, rtol=0, atol=0)
    torch.testing.assert_close(branches.output, expected_output, rtol=0, atol=0)
    if real_routes:
        assert torch.count_nonzero(branches.routed_output) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production backend guard requires a GPU")
def test_production_cuda_dispatch_fails_closed_without_bf16_grouped_mm() -> None:
    block = (
        GenesisNullSparseMoeBlock(
            _text_config(
                num_real_experts=256,
                num_null_experts=256,
                top_k=8,
                moe_intermediate_size=8,
            )
        )
        .cuda()
        .eval()
    )

    with pytest.raises(RuntimeError, match="requires BF16 torch grouped_mm"):
        block.dispatch_backend(torch.zeros(1, 24, device="cuda", dtype=torch.float32))


def test_hybrid_text_model_constructs_custom_layers_and_runs_forward() -> None:
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    model = GenesisQwen36TextModel(_text_config(layer_types=layer_types)).eval()

    assert [layer.layer_type for layer in model.layers] == layer_types
    assert all(isinstance(layer, GenesisQwen36DecoderLayer) for layer in model.layers)
    assert hasattr(model.layers[0], "linear_attn") and not hasattr(model.layers[0], "self_attn")
    assert hasattr(model.layers[3], "self_attn") and not hasattr(model.layers[3], "linear_attn")
    assert isinstance(model.layers[3].self_attn, GenesisQwen36Attention)
    assert model.layers[0].mlp.gate.weight.shape == (3, 24)
    state_dict = model.state_dict()
    assert "layers.0.mlp.gate.weight" in state_dict
    assert "layers.0.mlp.gate._router_contract" not in state_dict
    torch.testing.assert_close(model.layers[0].input_layernorm.weight, torch.zeros(24))

    with torch.no_grad():
        output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    assert output.last_hidden_state.shape == (1, 3, 24)
    assert torch.isfinite(output.last_hidden_state).all()


def test_hybrid_text_model_rejects_non_mk1_layer_pattern() -> None:
    with pytest.raises(ValueError, match="full attention every 4 layers"):
        GenesisQwen36TextModel(
            _text_config(
                layer_types=["full_attention", "linear_attention", "linear_attention", "full_attention"]
            )
        )


def test_hybrid_text_model_uses_validated_reduced_attention_interval() -> None:
    model = GenesisQwen36TextModel(
        _text_config(
            layer_types=["linear_attention", "full_attention"],
            full_attention_interval=2,
        )
    )

    assert [layer.layer_type for layer in model.layers] == ["linear_attention", "full_attention"]


def test_hybrid_cache_incremental_token_matches_full_forward() -> None:
    torch.manual_seed(31)
    model = GenesisQwen36TextModel(
        _text_config(
            layer_types=["linear_attention", "full_attention"],
            full_attention_interval=2,
            attention_implementation="sdpa",
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False).last_hidden_state
        prefix = model(input_ids=input_ids[:, :-1], use_cache=True)
        assert prefix.past_key_values is not None
        assert prefix.past_key_values.get_seq_length() == 3
        incremental = model(
            input_ids=input_ids[:, -1:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )

    assert incremental.past_key_values is prefix.past_key_values
    assert incremental.past_key_values.get_seq_length() == 4
    torch.testing.assert_close(incremental.last_hidden_state[:, -1], full[:, -1], rtol=1e-5, atol=1e-6)


def test_three_axis_vla_mrope_positions_match_equivalent_four_axis_positions() -> None:
    torch.manual_seed(37)
    model = GenesisQwen36TextModel(
        _text_config(
            layer_types=["linear_attention", "full_attention"],
            full_attention_interval=2,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    one_dimensional_positions = torch.arange(input_ids.shape[1]).expand(input_ids.shape[0], -1)
    vla_mrope_positions = one_dimensional_positions.unsqueeze(0).expand(3, -1, -1).contiguous()
    four_axis_positions = torch.cat(
        [one_dimensional_positions.unsqueeze(0), vla_mrope_positions],
        dim=0,
    )

    with torch.no_grad():
        three_axis = model(
            input_ids=input_ids,
            position_ids=vla_mrope_positions,
            use_cache=False,
        ).last_hidden_state
        four_axis = model(
            input_ids=input_ids,
            position_ids=four_axis_positions,
            use_cache=False,
        ).last_hidden_state

    torch.testing.assert_close(three_axis, four_axis, rtol=0, atol=0)


def test_hybrid_attention_is_causal_and_left_padding_preserves_valid_tokens() -> None:
    torch.manual_seed(41)
    model = GenesisQwen36TextModel(
        _text_config(
            layer_types=["linear_attention", "full_attention"],
            full_attention_interval=2,
            attention_implementation="sdpa",
        )
    ).eval()
    original = torch.tensor([[5, 6, 7], [9, 10, 11]])
    changed_future = torch.tensor([[5, 6, 8], [9, 10, 12]])
    left_padded = torch.tensor([[0, 0, 5, 6, 7], [0, 0, 9, 10, 11]])
    left_padding_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]])

    with torch.no_grad():
        original_output = model(input_ids=original, use_cache=False).last_hidden_state
        changed_output = model(input_ids=changed_future, use_cache=False).last_hidden_state
        padded_output = model(
            input_ids=left_padded,
            attention_mask=left_padding_mask,
            use_cache=False,
        ).last_hidden_state

    torch.testing.assert_close(original_output[:, :2], changed_output[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(original_output, padded_output[:, -3:], rtol=1e-5, atol=1e-6)


def test_hybrid_text_model_constructs_directly_on_meta() -> None:
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    with torch.device("meta"):
        model = GenesisQwen36TextModel(_text_config(layer_types=layer_types))

    assert all(parameter.is_meta for parameter in model.parameters())
    assert all(buffer.is_meta for buffer in model.buffers())
    assert model.layers[0].mlp.gate.weight.shape == (3, 24)


@pytest.mark.skipif(
    not qwen35_modeling.is_fast_path_available,
    reason="regression requires installed causal-conv1d and FLA fast dependencies",
)
def test_installed_fast_gdn_dependencies_preserve_cpu_and_meta_execution() -> None:
    config = _text_config(layer_types=["linear_attention"])
    gdn = GenesisQwen36GatedDeltaNet(config, layer_idx=0).eval()
    expected_state_keys = {
        "A_log",
        "conv1d.weight",
        "dt_bias",
        "in_proj_a.weight",
        "in_proj_b.weight",
        "in_proj_qkv.weight",
        "in_proj_z.weight",
        "norm.weight",
        "out_proj.weight",
    }

    assert set(gdn.state_dict()) == expected_state_keys
    assert all(parameter.device.type == "cpu" for parameter in gdn.parameters())
    with torch.no_grad():
        output = gdn(
            torch.randn(2, 3, config.hidden_size),
            attention_mask=torch.ones(2, 3, dtype=torch.bool),
        )
    assert output.shape == (2, 3, config.hidden_size)
    assert torch.isfinite(output).all()

    with torch.device("meta"):
        meta_gdn = GenesisQwen36GatedDeltaNet(config, layer_idx=0)
    assert set(meta_gdn.state_dict()) == expected_state_keys
    assert all(parameter.is_meta for parameter in meta_gdn.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not qwen35_modeling.is_fast_path_available,
    reason="CUDA parity requires an H100-class GPU and installed GDN fast dependencies",
)
def test_device_aware_gdn_cuda_matches_stock_fast_path() -> None:
    config = _text_config(layer_types=["linear_attention"])
    custom = GenesisQwen36GatedDeltaNet(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16).eval()
    stock = qwen35_modeling.Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).to(
        device="cuda", dtype=torch.bfloat16
    )
    stock.load_state_dict(custom.state_dict())
    hidden_states = torch.randn(2, 65, config.hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        actual = custom(hidden_states)
        expected = stock(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
