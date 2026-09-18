from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    MK1_TEST_GEOMETRY_MARKER,
    Mk1CheckpointContract,
    Mk1CheckpointContractError,
    read_and_validate_safetensors_index,
    validate_mk1_checkpoint,
    validate_mk1_tokenizer,
)


def _moe(*, real: int, top_k: int) -> dict:
    return {
        "router_contract_version": 1,
        "router_contract": [real, real],
        "num_real_experts": real,
        "num_null_experts": real,
        "physical_router_outputs": real + 1,
        "logical_router_outputs": 2 * real,
        "shared_null_router_row": True,
        "top_k": top_k,
        "score_func": "softmax",
        "route_norm": True,
        "route_scale": 1.0,
        "score_before_experts": False,
        "null_expert_semantics": "skip_compute_renormalize_real_routes",
        "shared_expert_mode": "sigmoid_gated_additive",
        "uses_expert_bias": False,
    }


def _config(*, test_geometry: bool, mtp_present: bool = True, coord_enabled: bool = True) -> dict:
    if test_geometry:
        hidden = 8
        num_layers = 2
        vocab_size = 32
        num_experts = 2
        top_k = 2
        heads = 2
        kv_heads = 1
        head_dim = 4
        moe_intermediate = 4
        full_interval = 2
        linear_conv = 2
        linear_key_dim = 2
        linear_key_heads = 2
        linear_value_heads = 2
        linear_value_dim = 2
        max_positions = 128
        partial_rotary = 0.5
        mrope_section = [1, 0, 0]
        rope_theta = 1000
        vision = {
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": hidden,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }
        vector_max_states = 4
        action = {
            "action_dim": 4,
            "action_horizon": 3,
            "num_layers": 2,
            "hidden_dim": 8,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "timestep_embed_dim": 4,
            "ffn_multiple_of": 4,
        }
    else:
        hidden = 2048
        num_layers = 40
        vocab_size = 256279
        num_experts = 256
        top_k = 8
        heads = 16
        kv_heads = 2
        head_dim = 256
        moe_intermediate = 512
        full_interval = 4
        linear_conv = 4
        linear_key_dim = 128
        linear_key_heads = 16
        linear_value_heads = 32
        linear_value_dim = 128
        max_positions = 262144
        partial_rotary = 0.25
        mrope_section = [11, 11, 10]
        rope_theta = 10_000_000
        vision = {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": hidden,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }
        vector_max_states = 128
        action = {
            "action_dim": 64,
            "action_horizon": 64,
            "num_layers": 36,
            "hidden_dim": 768,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "timestep_embed_dim": 256,
            "ffn_multiple_of": 256,
        }

    moe = _moe(real=num_experts, top_k=top_k)
    config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["GenesisQwen3_5MoeForConditionalGeneration"],
        "tie_word_embeddings": False,
        "genesis_artifact": {
            "schema_version": 1,
            "artifact_kind": "trained_policy",
            "trained_steps": 100_000,
        },
        "image_token_id": 4 if test_geometry else 248056,
        "video_token_id": 5 if test_geometry else 248057,
        "vision_start_token_id": 2 if test_geometry else 248053,
        "vision_end_token_id": 3 if test_geometry else 248054,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": "bfloat16",
            "full_attention_interval": full_interval,
            "head_dim": head_dim,
            "hidden_act": "silu",
            "hidden_size": hidden,
            "layer_types": [
                "full_attention" if (index + 1) % full_interval == 0 else "linear_attention"
                for index in range(num_layers)
            ],
            "linear_conv_kernel_dim": linear_conv,
            "linear_key_head_dim": linear_key_dim,
            "linear_num_key_heads": linear_key_heads,
            "linear_num_value_heads": linear_value_heads,
            "linear_value_head_dim": linear_value_dim,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": max_positions,
            "bos_token_id": 1 if test_geometry else 248044,
            "eos_token_id": 1 if test_geometry else 248044,
            "pad_token_id": None,
            "moe_intermediate_size": moe_intermediate,
            "mtp_num_hidden_layers": 1 if mtp_present else 0,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": heads,
            "num_experts": num_experts,
            "num_experts_per_tok": top_k,
            "num_hidden_layers": num_layers,
            "num_key_value_heads": kv_heads,
            "partial_rotary_factor": partial_rotary,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": mrope_section,
                "partial_rotary_factor": partial_rotary,
                "rope_theta": rope_theta,
                "rope_type": "default",
            },
            "shared_expert_intermediate_size": moe_intermediate,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": vocab_size,
            "genesis_moe": copy.deepcopy(moe),
        },
        "vision_config": {
            **vision,
            "deepstack_visual_indexes": [],
            "hidden_act": "gelu_pytorch_tanh",
            "model_type": "qwen3_5_moe",
        },
        "genesis_moe": copy.deepcopy(moe),
        "genesis_coord_tokens": (
            {
                "enabled": True,
                "offset": 24 if test_geometry else 248320,
                "size": 4 if test_geometry else 1001,
            }
            if coord_enabled
            else {"enabled": False}
        ),
        "genesis_vla": {
            "schema_version": 1,
            "backbone_family": "mk1_qwen3_6_moe",
            "state_dict_schema": "pr3154_v1",
            "rmsnorm_weight_convention": "zero_centered_1_plus_weight",
            "vector_encoder": {
                "type": "linear_silu_linear",
                "max_states": vector_max_states,
                "hidden_dim": hidden,
                "output_dim": hidden,
                "bias": False,
            },
            "action_expert": {
                "type": "molmoact",
                **action,
                "clean_at_0": False,
                "qk_norm": True,
                "qk_norm_eps": 1e-6,
                "rope": True,
                "context_layer_norm": True,
                "causal_attn": False,
            },
            "mtp": {
                "present": mtp_present,
                "physical_layers": 1 if mtp_present else 0,
                "rollout_steps": 3 if mtp_present else 0,
                "action_runtime": "exclude",
            },
        },
    }
    if test_geometry:
        config[MK1_TEST_GEOMETRY_MARKER] = True
    return config


def _parse_test_config(config: dict | None = None) -> Mk1CheckpointContract:
    return Mk1CheckpointContract.parse_allowlisted(
        _config(test_geometry=True) if config is None else config,
        allow_test_only_reduced_geometry=True,
    )


def _write_checkpoint(
    root: Path,
    *,
    config: dict | None = None,
    missing_key: str | None = None,
    extra_key: str | None = None,
    unindexed_key: str | None = None,
    duplicate_key: str | None = None,
    shape_override: tuple[str, tuple[int, ...]] | None = None,
    dtype_override: tuple[str, torch.dtype] | None = None,
) -> Mk1CheckpointContract:
    root.mkdir()
    checkpoint_config = _config(test_geometry=True) if config is None else copy.deepcopy(config)
    (root / "config.json").write_text(json.dumps(checkpoint_config), encoding="utf-8")
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"vocab": {"!": 0, "<|endoftext|>": 1}},
                "added_tokens": [
                    {"content": "<|vision_start|>", "id": 2, "special": True},
                    {"content": "<|vision_end|>", "id": 3, "special": True},
                    {"content": "<|image_pad|>", "id": 4, "special": True},
                    {"content": "<|video_pad|>", "id": 5, "special": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    contract = _parse_test_config(checkpoint_config)
    shapes = contract.expected_tensor_shapes()
    if missing_key is not None:
        shapes.pop(missing_key)
    if extra_key is not None:
        shapes[extra_key] = (1,)

    ordered = sorted(shapes)
    middle = len(ordered) // 2
    shard_keys = [ordered[:middle], ordered[middle:]]
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map: dict[str, str] = {}
    shard_states: list[dict[str, torch.Tensor]] = []
    for keys, shard_name in zip(shard_keys, shard_names, strict=True):
        state: dict[str, torch.Tensor] = {}
        for key in keys:
            shape = shapes[key]
            if shape_override is not None and key == shape_override[0]:
                shape = shape_override[1]
            dtype = torch.bfloat16
            if dtype_override is not None and key == dtype_override[0]:
                dtype = dtype_override[1]
            state[key] = torch.zeros(shape, dtype=dtype)
            weight_map[key] = shard_name
        shard_states.append(state)
    if unindexed_key is not None:
        shard_states[1][unindexed_key] = torch.zeros((1,), dtype=torch.bfloat16)
    if duplicate_key is not None:
        source_index = 0 if duplicate_key in shard_states[0] else 1
        target_index = 1 - source_index
        shard_states[target_index][duplicate_key] = shard_states[source_index][duplicate_key].clone()
    for shard_name, state in zip(shard_names, shard_states, strict=True):
        save_file(state, root / shard_name)
    total_size = sum(
        tensor.numel() * tensor.element_size() for state in shard_states for tensor in state.values()
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return contract


def test_production_contract_parses_exact_documented_geometry() -> None:
    contract = Mk1CheckpointContract.parse_allowlisted(_config(test_geometry=False))

    assert contract.text.hidden_size == 2048
    assert contract.text.layer_types.count("full_attention") == 10
    assert contract.moe.physical_router_outputs == 257
    assert contract.moe.logical_router_outputs == 512
    assert contract.text.vocab_size == 256279
    assert contract.special_tokens.image_token_id == 248056
    assert contract.action_expert.num_layers == 36
    assert contract.mtp.present is True
    assert contract.coord_tokens.enabled is True
    assert contract.artifact.artifact_kind == "trained_policy"
    assert contract.artifact.trained_steps == 100_000
    assert contract.excluded_runtime_prefixes == ("mtp.",)
    shapes = contract.expected_tensor_shapes()
    assert shapes["model.language_model.layers.0.mlp.gate.weight"] == (257, 2048)
    assert shapes["model.action_expert.action_expert.context_k_proj.weight"] == (768, 2048)
    assert shapes["mtp.layers.0.mlp.gate.weight"] == (257, 2048)


def test_production_contract_accepts_actual_non_mtp_coord_disabled_variant() -> None:
    config = _config(test_geometry=False, mtp_present=False, coord_enabled=False)
    config["genesis_artifact"]["trained_steps"] = 10_000

    contract = Mk1CheckpointContract.parse_allowlisted(config)
    shapes = contract.expected_tensor_shapes()

    assert contract.artifact.trained_steps == 10_000
    assert contract.text.vocab_size == 256279
    assert contract.mtp.present is False
    assert contract.coord_tokens.enabled is False
    assert contract.coord_tokens.offset is None
    assert contract.coord_tokens.size is None
    assert contract.excluded_runtime_prefixes == ()
    assert len(shapes) == 1616
    assert shapes["model.language_model.embed_tokens.weight"] == (256279, 2048)
    assert not any(key.startswith("mtp.") for key in shapes)


def test_coord_reserved_group_must_use_genesis_name_and_checkpoint_range() -> None:
    contract = _parse_test_config()
    contract.validate_coord_reserved_token_groups([{"name": "coord", "offset": 24, "size": 4}])

    for groups, match in (
        ([{"name": "coords", "offset": 24, "size": 4}], "exactly one"),
        ([{"name": "coord", "offset": 25, "size": 4}], "disagrees"),
        ([{"name": "coord", "offset": 24, "size": 3}], "disagrees"),
        (
            [
                {"name": "coord", "offset": 24, "size": 4},
                {"name": "coord", "offset": 24, "size": 4},
            ],
            "exactly one",
        ),
    ):
        with pytest.raises(Mk1CheckpointContractError, match=match):
            contract.validate_coord_reserved_token_groups(groups)


def test_production_reserved_groups_validate_enabled_coord_and_fast_ranges() -> None:
    contract = Mk1CheckpointContract.parse_allowlisted(_config(test_geometry=False))
    contract.validate_coord_reserved_token_groups(
        [
            {"name": "coord", "offset": 248320, "size": 1001},
            {
                "name": None,
                "offset": 249321,
                "size": 2048,
                "tokenizer": "physical-intelligence/fast",
            },
        ]
    )

    with pytest.raises(Mk1CheckpointContractError, match="FAST reserved-token range"):
        contract.validate_coord_reserved_token_groups(
            [
                {"name": "coord", "offset": 248320, "size": 1001},
                {
                    "name": None,
                    "offset": 249322,
                    "size": 2048,
                    "tokenizer": "physical-intelligence/fast",
                },
            ]
        )


@pytest.mark.parametrize("coord_enabled", [True, False])
def test_production_trained_policy_requires_exactly_one_fast_reserved_group(
    coord_enabled: bool,
) -> None:
    contract = Mk1CheckpointContract.parse_allowlisted(
        _config(test_geometry=False, mtp_present=False, coord_enabled=coord_enabled)
    )
    coord_groups = [{"name": "coord", "offset": 248320, "size": 1001}] if coord_enabled else []

    with pytest.raises(Mk1CheckpointContractError, match="exactly one physical-intelligence/fast"):
        contract.validate_coord_reserved_token_groups(
            [*coord_groups, {"name": None, "offset": 249321, "size": 2048, "tokenizer": "other"}]
        )

    fast_group = {
        "name": None,
        "offset": 249321,
        "size": 2048,
        "tokenizer": "physical-intelligence/fast",
    }
    with pytest.raises(Mk1CheckpointContractError, match="exactly one physical-intelligence/fast"):
        contract.validate_coord_reserved_token_groups([*coord_groups, fast_group, fast_group])


def test_coord_disabled_contract_accepts_fast_only_and_rejects_coord_group() -> None:
    contract = Mk1CheckpointContract.parse_allowlisted(
        _config(test_geometry=False, mtp_present=False, coord_enabled=False)
    )
    fast_group = {
        "name": None,
        "offset": 249321,
        "size": 2048,
        "tokenizer": "physical-intelligence/fast",
    }
    contract.validate_coord_reserved_token_groups([fast_group])

    with pytest.raises(Mk1CheckpointContractError, match="coordinates are disabled"):
        contract.validate_coord_reserved_token_groups(
            [{"name": "coord", "offset": 248320, "size": 1001}, fast_group]
        )


def test_tokenizer_special_ids_and_coordinate_reservation_are_checkpoint_owned(tmp_path: Path) -> None:
    contract = _parse_test_config()
    added_tokens: list[dict[str, str | int | bool]] = [
        {"content": "<|vision_start|>", "id": 2, "special": True},
        {"content": "<|vision_end|>", "id": 3, "special": True},
        {"content": "<|image_pad|>", "id": 4, "special": True},
        {"content": "<|video_pad|>", "id": 5, "special": True},
    ]
    tokenizer = {
        "model": {"vocab": {"!": 0, "<|endoftext|>": 1}},
        "added_tokens": added_tokens,
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    validate_mk1_tokenizer(tmp_path, contract)

    added_tokens[2]["id"] = 6
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    with pytest.raises(Mk1CheckpointContractError, match="image_pad.*expected 4"):
        validate_mk1_tokenizer(tmp_path, contract)

    added_tokens[2]["id"] = 4
    added_tokens[2]["special"] = False
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    with pytest.raises(Mk1CheckpointContractError, match="image_pad.*not marked special"):
        validate_mk1_tokenizer(tmp_path, contract)

    added_tokens[2]["special"] = True
    added_tokens.append({"content": "<|unexpected_coord|>", "id": 24})
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    with pytest.raises(Mk1CheckpointContractError, match="reserved coordinate range"):
        validate_mk1_tokenizer(tmp_path, contract)


def test_coord_disabled_tokenizer_has_no_reserved_coordinate_collision_check(tmp_path: Path) -> None:
    contract = _parse_test_config(_config(test_geometry=True, mtp_present=False, coord_enabled=False))
    tokenizer = {
        "model": {"vocab": {"!": 0, "<|endoftext|>": 1}},
        "added_tokens": [
            {"content": "<|vision_start|>", "id": 2, "special": True},
            {"content": "<|vision_end|>", "id": 3, "special": True},
            {"content": "<|image_pad|>", "id": 4, "special": True},
            {"content": "<|video_pad|>", "id": 5, "special": True},
            {"content": "<|ordinary_token|>", "id": 24, "special": False},
        ],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    validate_mk1_tokenizer(tmp_path, contract)


@pytest.mark.parametrize("coord_enabled", [True, False])
def test_production_tokenizer_rejects_explicit_ids_in_reserved_fast_range(
    tmp_path: Path,
    coord_enabled: bool,
) -> None:
    contract = Mk1CheckpointContract.parse_allowlisted(
        _config(test_geometry=False, mtp_present=False, coord_enabled=coord_enabled)
    )
    tokenizer = {
        "model": {"vocab": {"!": 0, "<|endoftext|>": 248044}},
        "added_tokens": [
            {"content": "<|vision_start|>", "id": 248053, "special": True},
            {"content": "<|vision_end|>", "id": 248054, "special": True},
            {"content": "<|image_pad|>", "id": 248056, "special": True},
            {"content": "<|video_pad|>", "id": 248057, "special": True},
            {"content": "<|unexpected_fast|>", "id": 249321, "special": False},
        ],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(Mk1CheckpointContractError, match="reserved FAST range"):
        validate_mk1_tokenizer(tmp_path, contract)


def test_reduced_geometry_requires_serialized_and_call_site_opt_in() -> None:
    config = _config(test_geometry=True)
    with pytest.raises(Mk1CheckpointContractError, match="test-only"):
        Mk1CheckpointContract.parse_allowlisted(config)

    del config[MK1_TEST_GEOMETRY_MARKER]
    with pytest.raises(Mk1CheckpointContractError, match="production MK1"):
        Mk1CheckpointContract.parse_allowlisted(config, allow_test_only_reduced_geometry=True)


def test_reduced_geometry_permission_accepts_unmarked_production_geometry() -> None:
    config = _config(test_geometry=False)
    config["genesis_artifact"] = {
        "schema_version": 1,
        "artifact_kind": "neutral_debug",
        "trained_steps": 0,
    }

    contract = Mk1CheckpointContract.parse_allowlisted(
        config,
        allow_test_only_reduced_geometry=True,
    )

    assert contract.test_only_reduced_geometry is False


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ({"artifact_kind": "trained_policy", "trained_steps": 1}, "missing required"),
        ({"schema_version": 2, "artifact_kind": "trained_policy", "trained_steps": 1}, "schema_version"),
        ({"schema_version": 1, "artifact_kind": "other", "trained_steps": 1}, "artifact_kind"),
        ({"schema_version": 1, "artifact_kind": "trained_policy", "trained_steps": 0}, "positive"),
        ({"schema_version": 1, "artifact_kind": "neutral_debug", "trained_steps": 1}, "trained_steps=0"),
        ({"schema_version": 1, "artifact_kind": "trained_policy", "trained_steps": True}, "integer"),
        (
            {
                "schema_version": 1,
                "artifact_kind": "trained_policy",
                "trained_steps": 1,
                "unbound_provenance": "x",
            },
            "fields do not match",
        ),
    ],
)
def test_artifact_identity_is_versioned_and_kind_consistent(artifact: dict, message: str) -> None:
    config = _config(test_geometry=True)
    config["genesis_artifact"] = artifact

    with pytest.raises(Mk1CheckpointContractError, match=message):
        _parse_test_config(config)


def test_neutral_artifact_accepts_inert_exporter_provenance() -> None:
    config = _config(test_geometry=True)
    config["genesis_artifact"] = {
        "schema_version": 1,
        "artifact_kind": "neutral_debug",
        "trained_steps": 0,
        "provenance_sha256": "0" * 64,
    }

    contract = _parse_test_config(config)

    assert contract.artifact.artifact_kind == "neutral_debug"
    assert contract.artifact.trained_steps == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("genesis_vla", "schema_version", 2), "schema_version"),
        (("genesis_vla", "backbone_family", "other"), "backbone_family"),
        (("genesis_vla", "rmsnorm_weight_convention", "legacy"), "rmsnorm"),
        (("genesis_vla", "mtp", "present", False), "present"),
        (("genesis_vla", "mtp", "action_runtime", "execute"), "action_runtime"),
        (("genesis_coord_tokens", "enabled", False), "unexpected"),
        (("image_token_id", 99), "inside the exported vocabulary"),
        (("text_config", "tie_word_embeddings", True), "tie_word_embeddings"),
        (("genesis_vla", "action_expert", "num_layers", 0), "positive"),
        (("output_hidden_states", True), "output_hidden_states"),
        (("return_dict", False), "return_dict"),
        (("attn_implementation", "flash_attention_2"), "must not be serialized"),
        (("text_config", "attention_dropout", 0.1), "attention_dropout"),
        (("text_config", "use_cache", False), "use_cache"),
        (("text_config", "output_router_logits", True), "output_router_logits"),
        (("text_config", "norm_topk_prob", False), "norm_topk_prob"),
        (("text_config", "return_dict", False), "return_dict"),
        (("vision_config", "output_attentions", True), "output_attentions"),
    ],
)
def test_structural_config_mutations_fail_closed(mutation: tuple, message: str) -> None:
    config = _config(test_geometry=True)
    target = config
    for key in mutation[:-2]:
        target = target[key]
    target[mutation[-2]] = mutation[-1]

    with pytest.raises(Mk1CheckpointContractError, match=message):
        _parse_test_config(config)


def test_root_and_text_null_contracts_must_be_identical() -> None:
    config = _config(test_geometry=True)
    config["text_config"]["genesis_moe"]["route_scale"] = 0.5

    with pytest.raises(Mk1CheckpointContractError, match="must be identical"):
        _parse_test_config(config)


def test_unknown_custom_contract_field_fails_closed() -> None:
    config = _config(test_geometry=True)
    config["genesis_vla"]["action_expert"]["local_default"] = True

    with pytest.raises(Mk1CheckpointContractError, match="unexpected"):
        _parse_test_config(config)


def test_valid_composite_inventory_uses_header_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    expected = _write_checkpoint(model_dir)

    contract, inventory = validate_mk1_checkpoint(
        model_dir,
        allow_test_only_reduced_geometry=True,
    )

    assert contract == expected
    assert inventory.keys == frozenset(contract.expected_tensor_shapes())
    assert {header.dtype for header in inventory.tensors.values()} == {"BF16"}
    assert len(inventory.shards) == 2
    assert set(inventory.shard_identities) == set(inventory.shards)
    for shard in inventory.shards:
        inventory.verify_shard_identity(shard)


def test_non_mtp_coord_disabled_inventory_has_exact_actual_key_family(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    config = _config(test_geometry=True, mtp_present=False, coord_enabled=False)
    expected = _write_checkpoint(model_dir, config=config)

    contract, inventory = validate_mk1_checkpoint(
        model_dir,
        allow_test_only_reduced_geometry=True,
    )

    assert contract == expected
    assert contract.mtp.present is False
    assert contract.coord_tokens.enabled is False
    assert inventory.keys == frozenset(contract.expected_tensor_shapes())
    assert not any(key.startswith("mtp.") for key in inventory.keys)


def test_validated_inventory_rejects_same_shape_shard_replacement(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    inventory = read_and_validate_safetensors_index(model_dir)
    shard = inventory.shards[0]
    replacement = tmp_path / "replacement.safetensors"
    shutil.copy2(shard, replacement)
    replacement.replace(shard)

    with pytest.raises(Mk1CheckpointContractError, match="changed after validation"):
        inventory.verify_shard_identity(shard)


def test_contract_validation_accepts_hf_snapshot_blob_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_checkpoint(source)
    repository = tmp_path / "cache" / "models--org--mk1"
    blobs = repository / "blobs"
    model_dir = repository / "snapshots" / "revision" / "hf_model"
    blobs.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    for index, source_file in enumerate(sorted(source.iterdir())):
        blob = blobs / f"{index:04d}"
        shutil.copy2(source_file, blob)
        (model_dir / source_file.name).symlink_to(blob)

    contract, inventory = validate_mk1_checkpoint(
        model_dir,
        allow_test_only_reduced_geometry=True,
    )

    assert inventory.keys == frozenset(contract.expected_tensor_shapes())
    for shard in inventory.shards:
        assert inventory.shard_identities[shard].resolved_path.parent == blobs.resolve()
        inventory.verify_shard_identity(shard)

    outside = tmp_path / "outside.safetensors"
    shutil.copy2(model_dir / "model-00001-of-00002.safetensors", outside)
    (model_dir / "model-00001-of-00002.safetensors").unlink()
    (model_dir / "model-00001-of-00002.safetensors").symlink_to(outside)
    with pytest.raises(Mk1CheckpointContractError, match="missing or escaping shard"):
        validate_mk1_checkpoint(model_dir, allow_test_only_reduced_geometry=True)


def test_config_tensor_shape_disagreement_fails_before_construction(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    key = "model.action_expert.action_expert.context_k_proj.weight"
    _write_checkpoint(model_dir, shape_override=(key, (8, 10)))

    with pytest.raises(Mk1CheckpointContractError, match=r"context_k_proj.*expected \(8, 8\)"):
        validate_mk1_checkpoint(model_dir, allow_test_only_reduced_geometry=True)


@pytest.mark.parametrize(
    ("kind", "key", "message"),
    [
        ("missing", "mtp.pre_fc_norm_hidden.weight", "key coverage mismatch"),
        ("extra", "optimizer.state", "unexpected=1"),
        ("unindexed", "unindexed.weight", "index/header mismatch"),
        ("duplicate", "lm_head.weight", "duplicate tensor keys"),
    ],
)
def test_missing_unexpected_unindexed_and_duplicate_tensors_fail(
    tmp_path: Path, kind: str, key: str, message: str
) -> None:
    model_dir = tmp_path / kind
    _write_checkpoint(
        model_dir,
        missing_key=key if kind == "missing" else None,
        extra_key=key if kind == "extra" else None,
        unindexed_key=key if kind == "unindexed" else None,
        duplicate_key=key if kind == "duplicate" else None,
    )

    with pytest.raises(Mk1CheckpointContractError, match=message):
        validate_mk1_checkpoint(model_dir, allow_test_only_reduced_geometry=True)


def test_non_bf16_tensor_fails_contract(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    key = "model.vector_embedding.0.weight"
    _write_checkpoint(model_dir, dtype_override=(key, torch.float32))

    with pytest.raises(Mk1CheckpointContractError, match=r"dtype F32, expected BF16"):
        validate_mk1_checkpoint(model_dir, allow_test_only_reduced_geometry=True)


def test_index_rejects_escaping_shard_name(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    first_key = next(iter(index["weight_map"]))
    index["weight_map"][first_key] = "../model-00001-of-00002.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(Mk1CheckpointContractError, match="invalid shard name"):
        read_and_validate_safetensors_index(model_dir)


def test_unindexed_safetensors_file_is_rejected(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    save_file({"extra": torch.zeros((1,), dtype=torch.bfloat16)}, model_dir / "extra.safetensors")

    with pytest.raises(Mk1CheckpointContractError, match="unindexed=.*extra.safetensors"):
        read_and_validate_safetensors_index(model_dir)


def test_policy_state_identity_safetensors_is_allowed_beside_model_shards(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    save_file(
        {"policy_state_identity": torch.zeros((100,), dtype=torch.uint8)},
        model_dir / "policy_state_identity.safetensors",
    )

    inventory = read_and_validate_safetensors_index(model_dir)

    assert len(inventory.shards) == 2
    assert "policy_state_identity" not in inventory.tensors


def test_index_total_size_must_match_headers(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["total_size"] += 1
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(Mk1CheckpointContractError, match="total_size does not match"):
        read_and_validate_safetensors_index(model_dir)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map": {}, "weight_map": {}}',
        encoding="utf-8",
    )

    with pytest.raises(Mk1CheckpointContractError, match="duplicate JSON key"):
        read_and_validate_safetensors_index(model_dir)


def _write_portable_isaac05_checkpoint(
    root: Path,
) -> tuple[Mk1CheckpointContract, dict, torch.Tensor]:
    """Write the shared tiny F32 source fixture; no checkpoint code is needed."""
    from safetensors.torch import load_file

    native = _config(test_geometry=True, mtp_present=False)
    raw = copy.deepcopy(native)
    raw["model_type"] = "isaac_0_5"
    raw["architectures"] = ["Isaac05ForConditionalGeneration"]
    raw["auto_map"] = {
        "AutoConfig": "configuration_isaac05.Isaac05Config",
        "AutoModelForCausalLM": "modeling_isaac05.Isaac05ForConditionalGeneration",
        "AutoProcessor": "processing_isaac05.Isaac05Processor",
    }
    raw["storage_dtype"] = "float32"
    raw["runtime_dtype"] = raw["dtype"] = "bfloat16"
    raw["isaac05_test_only_reduced_geometry"] = raw.pop(MK1_TEST_GEOMETRY_MARKER)
    for name in ("artifact", "coord_tokens", "moe", "vla"):
        raw[f"isaac05_{name}"] = raw.pop(f"genesis_{name}")
    raw["text_config"]["isaac05_moe"] = raw["text_config"].pop("genesis_moe")
    for moe in (raw["isaac05_moe"], raw["text_config"]["isaac05_moe"]):
        del moe["null_expert_semantics"], moe["shared_expert_mode"]
    del raw["isaac05_vla"]["backbone_family"]
    action = raw["isaac05_vla"]["action_expert"]
    del action["clean_at_0"]
    action.update(
        type="dit",
        schema_version=1,
        num_inference_steps=10,
        timestep_sampling_alpha=1.5,
        timestep_sampling_beta=1.0,
        timestep_sampling_scale=0.999,
        timestep_sampling_offset=0.001,
        train_samples_per_chunk=8,
        rtc_max_delay_steps=12,
        rtc_probability=0.5,
        rtc_delay_sampling="poisson",
        rtc_poisson_mean=5.0,
        mask_padded_action_rows=True,
        drop_action_dim_overflow=False,
        k_batched_cross_attn=True,
        k_batched_cross_attn_backend="flash_gqa",
    )
    raw["action_expert"] = copy.deepcopy(action)
    raw["vector_max_states"] = 4
    raw["max_sequence_length"] = 128
    raw["vision_token"] = "<|image_pad|>"
    raw["vision_rescale_factor"] = 1 / 255
    raw["isaac05_fast_tokens"] = {
        "enabled": True,
        "offset": 28,
        "size": 4,
        "tokenizer": "physical-intelligence/fast",
    }
    expected = _write_checkpoint(root, config=raw)
    total_bytes = 0
    tensor_count = 0
    value = torch.tensor(0.12345679, dtype=torch.float32)
    for shard in sorted(root.glob("model-*.safetensors")):
        state = {key: torch.full(tensor.shape, value.item()) for key, tensor in load_file(shard).items()}
        total_bytes += sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        tensor_count += len(state)
        save_file(state, shard)
    raw["isaac05_artifact"].update(tensor_count=tensor_count, tensor_bytes=total_bytes)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["total_size"] = total_bytes
    index_path.write_text(json.dumps(index))
    source_bytes = json.dumps(raw).encode()
    (root / "config.json").write_bytes(source_bytes)
    return expected, raw, value


def test_portable_isaac05_loads_native_fp32_without_checkpoint_python(tmp_path: Path) -> None:
    """The historical isaac05 JSON is input data, never executable model code."""
    from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf

    root = tmp_path / "portable"
    expected, raw, value = _write_portable_isaac05_checkpoint(root)
    source_bytes = (root / "config.json").read_bytes()
    action = raw["action_expert"]

    for field, value_override in (("rtc_probability", 0.75), ("unknown_field", True)):
        invalid = copy.deepcopy(raw)
        invalid["action_expert"][field] = value_override
        invalid["isaac05_vla"]["action_expert"][field] = value_override
        with pytest.raises(Mk1CheckpointContractError):
            _parse_test_config(invalid)
    invalid = copy.deepcopy(raw)
    invalid["isaac05_moe"]["top_k"] += 1
    with pytest.raises(Mk1CheckpointContractError, match="identical"):
        _parse_test_config(invalid)
    invalid = copy.deepcopy(raw)
    invalid["action_expert"]["rtc_probability"] = 0.1
    with pytest.raises(Mk1CheckpointContractError, match="identical"):
        _parse_test_config(invalid)
    invalid = copy.deepcopy(raw)
    invalid["storage_dtype"] = "bfloat16"
    with pytest.raises(Mk1CheckpointContractError, match="storage_dtype"):
        _parse_test_config(invalid)

    with pytest.raises(ValueError, match="training load intent"):
        load_mk1_vla_from_hf(
            root,
            dtype=torch.float32,
            allow_test_only_reduced_geometry=True,
            allowed_storage_dtypes=frozenset({"F32"}),
        )
    invalid = copy.deepcopy(raw)
    invalid["vector_max_states"] += 1
    with pytest.raises(Mk1CheckpointContractError, match="vector_max_states"):
        _parse_test_config(invalid)
    invalid = copy.deepcopy(raw)
    del invalid["action_expert"]["rtc_probability"]
    del invalid["isaac05_vla"]["action_expert"]["rtc_probability"]
    with pytest.raises(
        Mk1CheckpointContractError,
        match=r"isaac05_vla\.action_expert fields .*missing=\['rtc_probability'\]",
    ):
        _parse_test_config(invalid)

    model, config, contract = load_mk1_vla_from_hf(
        root,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
        allowed_storage_dtypes=frozenset({"F32"}),
        load_intent="training",
    )

    assert contract == expected
    assert type(model).__module__ == "lerobot.policies.perceptron_isaac.modeling_mk1_vla"
    assert config.model_type == "qwen3_5_moe"
    assert model.action_expert.expert_type == "dit"
    assert config.action_expert == action
    assert set(model.state_dict()) == set(contract.expected_tensor_shapes())
    assert value.item() != value.bfloat16().float().item()
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(torch.equal(parameter, value.expand_as(parameter)) for parameter in model.parameters())
    assert torch.equal(model.lm_head.weight, value.expand_as(model.lm_head.weight))
    assert (root / "config.json").read_bytes() == source_bytes
    assert not model.training
