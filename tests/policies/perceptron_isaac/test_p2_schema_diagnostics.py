"""P2.15: accurate eager schema refusal without misleading conversion advice."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal, NoReturn

import pytest

pytest.importorskip("transformers", reason="requires the parent-owned ISAAC CPU environment")

from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    Mk1CheckpointContractError,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import _validate_action_expert_contract
from lerobot.policies.perceptron_isaac.trained_package import finalize_canonical_trained_package
from tests.policies.perceptron_isaac.test_action_expert_contract import _dit_contract
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _source_package
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _write_portable_isaac05_checkpoint
from tests.policies.perceptron_isaac.test_portable_model_repo import _native_portable_package


def _assert_unsupported_layerwise(message: str) -> None:
    lowered = message.lower()
    assert "unsupported" in lowered or "not supported" in lowered
    assert "per-layer" in lowered or "layerwise" in lowered or "layer_hidden" in lowered
    assert "re-export" not in lowered, "the current converter emits the same unsupported schema"


@pytest.mark.parametrize("schema", [1, 2])
def test_hf_validator_names_unsupported_layerwise_semantics(schema: int) -> None:
    action = _dit_contract(
        schema_version=schema,
        context_source="layer_hidden",
        context_layers=[0],
        context_hidden_norm="final_norm",
    )
    with pytest.raises(ValueError) as caught:
        _validate_action_expert_contract(action)
    _assert_unsupported_layerwise(str(caught.value))
    if schema == 2:
        assert "2" in str(caught.value)


@pytest.mark.parametrize("schema", [1, 2])
def test_mk1_validator_names_unsupported_layerwise_semantics(tmp_path: Path, schema: int) -> None:
    _, raw, _ = _write_portable_isaac05_checkpoint(tmp_path / "model")
    raw["action_expert"].update(
        schema_version=schema,
        context_source="layer_hidden",
        context_layers=[0, 1],
        context_hidden_norm="final_norm",
    )
    raw["isaac05_vla"]["action_expert"] = copy.deepcopy(raw["action_expert"])
    with pytest.raises(Mk1CheckpointContractError) as caught:
        Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
    _assert_unsupported_layerwise(str(caught.value))
    if schema == 2:
        assert "2" in str(caught.value)


@pytest.mark.parametrize("lazy", [True, False], ids=["fp32_lazy", "bf16_eager"])
def test_public_load_refuses_schema2_before_model_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lazy: bool
) -> None:
    import lerobot.policies.perceptron_isaac.modeling_mk1_vla as native

    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_storage_fp32 = lazy
    source_path = policy_path.parent / "config.json"
    raw = json.loads(source_path.read_text())
    raw["action_expert"].update(
        schema_version=2,
        context_source="layer_hidden",
        context_layers=[0, 1],
        context_hidden_norm="final_norm",
    )
    raw["isaac05_vla"]["action_expert"] = copy.deepcopy(raw["action_expert"])
    source_path.write_text(json.dumps(raw))
    original_bytes = source_path.read_bytes()

    def forbid_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("unsupported schema reached model allocation")

    monkeypatch.setattr(native, "Mk1Qwen36VLAForActionGeneration", forbid_model)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    _assert_unsupported_layerwise(str(caught.value))
    assert "2" in str(caught.value)
    assert source_path.read_bytes() == original_bytes


def test_schema1_control_is_accepted_by_both_real_validators(tmp_path: Path) -> None:
    _validate_action_expert_contract(_dit_contract())
    _, raw, _ = _write_portable_isaac05_checkpoint(tmp_path / "model")
    contract = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
    assert contract.action_expert.num_layers == 2


def test_schema1_unknown_field_is_still_refused() -> None:
    with pytest.raises(ValueError, match="unexpected fields"):
        _validate_action_expert_contract(_dit_contract(unimplemented_attention_mode="new"))


def test_schema2_without_layer_fields_is_not_treated_as_convertible_schema1() -> None:
    with pytest.raises(ValueError) as caught:
        _validate_action_expert_contract(_dit_contract(schema_version=2))
    _assert_unsupported_layerwise(str(caught.value))
    assert "2" in str(caught.value)


Corruption = Literal["policy-count", "manifest", "source-config", "unlisted"]


def _authenticated_layerwise_package(root: Path, *, schema: int, lazy: bool) -> None:
    config = _source_package(root)
    config.train_storage_fp32 = lazy
    path = root / "hf_model/config.json"
    raw = json.loads(path.read_text())
    raw["action_expert"].update(
        schema_version=schema,
        context_source="layer_hidden",
        context_layers=[0, 1],
        context_hidden_norm="final_norm",
    )
    raw["isaac05_vla"]["action_expert"] = copy.deepcopy(raw["action_expert"])
    path.write_text(json.dumps(raw))
    # Bind the unsupported semantics honestly before testing diagnostic precedence.
    finalize_canonical_trained_package(
        config, root, mk1_source_model_import_sha256=config.mk1_source_model_import_sha256
    )
    assert config.mk1_trained_package_manifest_sha256 == file_sha256(
        root / "mk1_trained_package_manifest.json"
    )


def _forbid_construction(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("authenticated package refusal reached policy/native construction")


def _construction_tripwires(monkeypatch: pytest.MonkeyPatch) -> None:
    import lerobot.policies.perceptron_isaac.modeling_mk1_vla as native

    monkeypatch.setattr(PerceptronIsaacPolicy, "__init__", _forbid_construction)
    monkeypatch.setattr(native, "Mk1Qwen36VLAForActionGeneration", _forbid_construction)


@pytest.mark.parametrize("schema", [1, 2], ids=["schema1-per-layer", "schema2-per-layer"])
@pytest.mark.parametrize("lazy", [True, False], ids=["fp32-lazy", "bf16-eager"])
def test_authenticated_layerwise_refusal_precedes_policy_and_native_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: int, lazy: bool
) -> None:
    root = tmp_path / "package"
    _authenticated_layerwise_package(root, schema=schema, lazy=lazy)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    _construction_tripwires(monkeypatch)

    with pytest.raises(ValueError) as caught:
        PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)

    _assert_unsupported_layerwise(str(caught.value))
    assert f"schema_version={schema}" in str(caught.value)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("schema", [1, 2], ids=["schema1-per-layer", "schema2-per-layer"])
@pytest.mark.parametrize("lazy", [True, False], ids=["fp32-lazy", "bf16-eager"])
@pytest.mark.parametrize("corruption", ["policy-count", "manifest", "source-config", "unlisted"])
def test_authenticated_integrity_error_wins_over_layerwise_semantics_before_both_constructors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: int, lazy: bool, corruption: Corruption
) -> None:
    root = tmp_path / "package"
    _authenticated_layerwise_package(root, schema=schema, lazy=lazy)
    if corruption == "policy-count":
        path = root / "config.json"
        raw = json.loads(path.read_text())
        raw["trained_steps"] = 100_001
        path.write_text(json.dumps(raw))
        message = "canonical config digest mismatch"
    elif corruption == "manifest":
        path = root / "mk1_trained_package_manifest.json"
        path.write_text(path.read_text() + " ")
        message = "manifest digest mismatch"
    elif corruption == "source-config":
        path = root / "hf_model/config.json"
        path.write_text(path.read_text() + " ")
        message = "digest mismatch for hf_model/config.json"
    else:
        (root / "unlisted.json").write_text("{}\n")
        message = "file inventory disagrees"
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    _construction_tripwires(monkeypatch)

    with pytest.raises(RuntimeError, match=message) as caught:
        PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)

    assert "per-layer" not in str(caught.value)
    assert "layer_hidden" not in str(caught.value)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
