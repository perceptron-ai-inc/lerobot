"""Schema-2 acceptance must not weaken the native checkpoint trust boundary.

These tests call current validators/loaders. No runtime importer or checkpoint Python
is needed. The VLA attachment envelope and action-expert schema are separate versions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    Mk1CheckpointContractError,
    read_mk1_config_data,
    validate_mk1_checkpoint,
)
from tests.policies.perceptron_isaac._layerwise_fixtures import (
    declare_native_layerwise,
    layerwise_action_config,
    write_layerwise_checkpoint,
)
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _config,
    _write_portable_isaac05_checkpoint,
)


def test_schema2_head_validator_accepts_complete_layerwise_declaration() -> None:
    from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import _validate_action_expert_contract

    action = layerwise_action_config()
    before = copy.deepcopy(action)
    args = _validate_action_expert_contract(action)

    assert args.num_layers == 2
    # Numeric tests below prove that acceptance is not stripping the new fields.
    assert action == before


def test_native_schema2_contract_preserves_metadata_and_schema1_tensor_geometry(tmp_path: Path) -> None:
    fixture = write_layerwise_checkpoint(tmp_path / "model")
    before = (fixture.root / "config.json").read_bytes()

    normalized, contract = read_mk1_config_data(fixture.root, allow_test_only_reduced_geometry=True)

    assert dict(contract.action_expert.native_metadata) == fixture.raw["action_expert"]
    assert normalized["genesis_vla"]["action_expert"] == fixture.raw["action_expert"]
    assert contract.expected_tensor_shapes() == fixture.schema1_contract.expected_tensor_shapes()
    assert contract.artifact == fixture.schema1_contract.artifact
    assert (fixture.root / "config.json").read_bytes() == before


def test_migrated_schema2_preserves_full_40_layer_production_declaration_without_new_weights() -> None:
    # The archived migrated form versions genesis_vla, not the legacy MolmoAct
    # action block. Native Isaac05 instead versions its duplicated DiT action block.
    raw = _config(test_geometry=False, mtp_present=False, coord_enabled=False)
    schema1 = Mk1CheckpointContract.parse_allowlisted(raw)
    raw["genesis_vla"]["schema_version"] = 2
    raw["genesis_vla"]["action_expert"].update(
        context_source="layer_hidden",
        context_layers=list(range(4, 40)),
        context_hidden_norm="final_norm",
    )
    before = copy.deepcopy(raw)

    contract = Mk1CheckpointContract.parse_allowlisted(raw)

    assert contract.text.num_hidden_layers == 40
    assert contract.action_expert.num_layers == 36
    assert contract.expected_tensor_shapes() == schema1.expected_tensor_shapes()
    assert raw == before


@pytest.mark.parametrize("field", ["context_source", "context_layers", "context_hidden_norm"])
def test_schema2_requires_every_conditioning_field(tmp_path: Path, field: str) -> None:
    fixture = write_layerwise_checkpoint(tmp_path / "model")
    for action in (fixture.raw["action_expert"], fixture.raw["isaac05_vla"]["action_expert"]):
        del action[field]

    with pytest.raises(Mk1CheckpointContractError, match=field):
        Mk1CheckpointContract.parse_allowlisted(fixture.raw, allow_test_only_reduced_geometry=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("context_source", "final_hidden", "context_source|layerwise", id="final-substitution"),
        pytest.param("context_source", "unknown", "context_source", id="unknown-source"),
        pytest.param("context_hidden_norm", "unknown", "context_hidden_norm", id="unknown-norm"),
        pytest.param("context_hidden_norm", "none", "context_hidden_norm", id="raw-last-layer"),
        pytest.param("unimplemented_context_mode", True, "unexpected|fields", id="unknown-field"),
        pytest.param("context_layers", [], "context_layers", id="empty"),
        pytest.param("context_layers", [0], "context_layers", id="short"),
        pytest.param("context_layers", [0, 0], "context_layers", id="duplicate"),
        pytest.param("context_layers", [1, 0], "context_layers", id="descending"),
        pytest.param("context_layers", [-1, 0], "context_layers", id="negative"),
        pytest.param("context_layers", [0, 2], "context_layers", id="out-of-backbone"),
        pytest.param("context_layers", [False, 1], "context_layers", id="boolean"),
        pytest.param("context_layers", [0, 1.0], "context_layers", id="float"),
        pytest.param("context_layers", "0,1", "context_layers", id="string"),
        pytest.param("context_layers", None, "context_layers", id="null"),
        pytest.param("context_layers", (0, 1), "context_layers", id="not-json-list"),
    ],
)
def test_native_schema2_rejects_malformed_conditioning(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    fixture = write_layerwise_checkpoint(tmp_path / "model")
    for action in (fixture.raw["action_expert"], fixture.raw["isaac05_vla"]["action_expert"]):
        action[field] = value

    with pytest.raises(Mk1CheckpointContractError, match=message):
        Mk1CheckpointContract.parse_allowlisted(fixture.raw, allow_test_only_reduced_geometry=True)


@pytest.mark.parametrize("schema", [0, 3, True, 2.0, "2", None])
def test_native_action_schema_version_remains_strict(tmp_path: Path, schema: object) -> None:
    fixture = write_layerwise_checkpoint(tmp_path / "model")
    for action in (fixture.raw["action_expert"], fixture.raw["isaac05_vla"]["action_expert"]):
        action["schema_version"] = schema

    with pytest.raises(Mk1CheckpointContractError, match="schema_version"):
        Mk1CheckpointContract.parse_allowlisted(fixture.raw, allow_test_only_reduced_geometry=True)


@pytest.mark.parametrize("copy_name", ["root", "nested"])
def test_native_schema2_still_cross_checks_both_action_copies(tmp_path: Path, copy_name: str) -> None:
    fixture = write_layerwise_checkpoint(tmp_path / "model")
    action = (
        fixture.raw["action_expert"] if copy_name == "root" else fixture.raw["isaac05_vla"]["action_expert"]
    )
    action["schema_version"] = 1
    for field in ("context_source", "context_layers", "context_hidden_norm"):
        del action[field]
    # Individually valid schema-1/schema-2 copies must never normalize into equality.
    with pytest.raises(Mk1CheckpointContractError, match="identical"):
        Mk1CheckpointContract.parse_allowlisted(fixture.raw, allow_test_only_reduced_geometry=True)


def test_native_schema1_control_has_no_layerwise_relabeling(tmp_path: Path) -> None:
    root = tmp_path / "model"
    expected, raw, _ = _write_portable_isaac05_checkpoint(root)
    before = (root / "config.json").read_bytes()

    normalized, contract = read_mk1_config_data(root, allow_test_only_reduced_geometry=True)

    assert contract == expected
    assert normalized["genesis_vla"]["action_expert"] == raw["action_expert"]
    assert raw["action_expert"]["schema_version"] == 1
    assert "context_layers" not in dict(contract.action_expert.native_metadata)
    assert (root / "config.json").read_bytes() == before


@pytest.mark.parametrize(
    ("dtype", "intent"),
    [(torch.float32, "training"), (torch.bfloat16, "inference")],
    ids=["lossless-f32", "bf16-inference"],
)
def test_schema2_raw_native_loader_preserves_values_signed_zero_and_source_bytes(
    tmp_path: Path, dtype: torch.dtype, intent: Literal["training", "inference"]
) -> None:
    from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf

    fixture = write_layerwise_checkpoint(tmp_path / "model")
    key = "lm_head.weight"
    expected = fixture.source_value.expand(fixture.schema1_contract.expected_tensor_shapes()[key]).clone()
    expected.reshape(-1)[:2] = torch.tensor([-0.0, 0.0])
    for shard in fixture.root.glob("model-*.safetensors"):
        state = load_file(shard)
        if key in state:
            state[key] = expected
            save_file(state, shard)
    before = {path.name: path.read_bytes() for path in fixture.root.iterdir()}

    model, config, contract = load_mk1_vla_from_hf(
        fixture.root,
        dtype=dtype,
        device="cpu",
        allow_test_only_reduced_geometry=True,
        allowed_storage_dtypes=frozenset({"F32"}),
        load_intent=intent,
    )

    assert config.action_expert == fixture.raw["action_expert"]
    assert dict(contract.action_expert.native_metadata) == fixture.raw["action_expert"]
    assert set(model.state_dict()) == set(fixture.schema1_contract.expected_tensor_shapes())
    actual = model.lm_head.weight.detach()
    assert actual.dtype == dtype
    assert torch.equal(actual.view(torch.uint8), expected.to(dtype).view(torch.uint8))
    assert torch.signbit(actual.reshape(-1)[0]) and not torch.signbit(actual.reshape(-1)[1])
    assert all(
        parameter.dtype == dtype and parameter.device.type == "cpu" for parameter in model.parameters()
    )
    assert all(buffer.device.type != "meta" for buffer in model.buffers())
    assert {path.name: path.read_bytes() for path in fixture.root.iterdir()} == before


def test_schema2_raw_native_loader_still_requires_training_intent_for_f32(tmp_path: Path) -> None:
    from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf

    fixture = write_layerwise_checkpoint(tmp_path / "model")
    with pytest.raises(ValueError, match="training load intent"):
        load_mk1_vla_from_hf(
            fixture.root,
            dtype=torch.float32,
            allowed_storage_dtypes=frozenset({"F32"}),
            allow_test_only_reduced_geometry=True,
        )


@pytest.mark.parametrize("defect", ["shape", "dtype", "tokenizer"])
def test_schema2_raw_validation_precedes_native_construction(tmp_path: Path, defect: str) -> None:
    from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf

    fixture = write_layerwise_checkpoint(tmp_path / "model")
    if defect == "tokenizer":
        path = fixture.root / "tokenizer.json"
        payload = json.loads(path.read_text())
        payload["added_tokens"][2]["id"] = 6
        path.write_text(json.dumps(payload))
        message = "image_pad.*expected 4"
    else:
        shard = next(fixture.root.glob("model-*.safetensors"))
        state = load_file(shard)
        key = next(iter(state))
        state[key] = state[key].reshape(-1)[:1] if defect == "shape" else state[key].bfloat16()
        save_file(state, shard)
        message = "total_size|shape|dtype"
    with patch(
        "lerobot.policies.perceptron_isaac.modeling_mk1_vla.Mk1Qwen36VLAForActionGeneration",
        side_effect=AssertionError("invalid source reached native construction"),
    ) as constructor:
        with pytest.raises(ValueError, match=message):
            load_mk1_vla_from_hf(
                fixture.root,
                allowed_storage_dtypes=frozenset({"F32"}),
                allow_test_only_reduced_geometry=True,
            )
        constructor.assert_not_called()


def test_public_portable_schema2_loads_in_tree_without_rewriting_source(tmp_path: Path) -> None:
    from lerobot.policies.perceptron_isaac.modeling_mk1_vla import Mk1Qwen36VLAForActionGeneration
    from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
    from tests.policies.perceptron_isaac.test_portable_model_repo import _native_portable_package

    policy_path, config, _ = _native_portable_package(tmp_path)
    source = policy_path.parent / "config.json"
    raw = json.loads(source.read_text())
    declare_native_layerwise(raw)
    source.write_text(json.dumps(raw))
    before = {path: path.read_bytes() for path in policy_path.parent.rglob("*") if path.is_file()}

    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)

    assert isinstance(policy._isaac_model, Mk1Qwen36VLAForActionGeneration)
    assert policy._isaac_model.config.action_expert == raw["action_expert"]
    assert (
        policy.config.artifact_kind == "neutral_debug"
    )  # Tiny-fixture permission only, not a deployment claim.
    assert {path: path.read_bytes() for path in policy_path.parent.rglob("*") if path.is_file()} == before


def test_legacy_production_schema1_geometry_remains_allowlisted() -> None:
    raw: dict[str, Any] = _config(test_geometry=False, mtp_present=False, coord_enabled=False)
    before = copy.deepcopy(raw)
    contract = Mk1CheckpointContract.parse_allowlisted(raw)
    assert contract.text.num_hidden_layers == 40
    assert contract.action_expert.num_layers == 36
    assert contract.action_expert.native_metadata == ()
    assert raw == before


def test_schema2_inventory_does_not_add_parameters(tmp_path: Path) -> None:
    fixture = write_layerwise_checkpoint(tmp_path / "model")
    contract, inventory = validate_mk1_checkpoint(
        fixture.root,
        allow_test_only_reduced_geometry=True,
        allowed_storage_dtypes=frozenset({"F32"}),
    )
    assert inventory.keys == frozenset(fixture.schema1_contract.expected_tensor_shapes())
    assert contract.expected_tensor_shapes() == fixture.schema1_contract.expected_tensor_shapes()
