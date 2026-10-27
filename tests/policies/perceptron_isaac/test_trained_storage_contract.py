"""Trained payload storage is independent of its unchanged source export profile.

These tests exercise the public policy save/finalize/move/reload path and authenticated
canonical v3 storage declarations. Only reduced geometry and FSDP2 topology are simulated; the
native loader, parameter freezing, state gathering, serialization and validation run.
No process group, worker, accelerator preparation or checkpoint Python is needed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers", reason="requires the existing perceptron_isaac test environment")

from lerobot.distributed import checkpoint, utils as distributed_utils
from lerobot.policies.perceptron_isaac import mk1_checkpoint_contract, modeling_mk1_vla
from lerobot.policies.perceptron_isaac.checkpoint_integrity import canonical_json_sha256, file_sha256
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mharmony_native import IsaacMharmonyRenderMetadata
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    SafetensorsInventory,
    SafetensorsShardIdentity,
)
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    Mk1Qwen36VLAConfig,
    Mk1Qwen36VLAForActionGeneration,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.trained_package import (
    VerifiedTrainedTensorStorage,
    finalize_canonical_trained_package,
)
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _config, _source_package
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _config as _mk1_config,
    _write_checkpoint,
    _write_portable_isaac05_checkpoint,
)
from tests.policies.perceptron_isaac.test_portable_model_repo import (
    _install_distributed_state,
    _StubAcceleratorState,
    _StubFsdpPlugin,
)

SourceFamily = Literal["isaac05", "legacy_mk1"]
StorageProfile = Literal["all_bf16", "all_f32", "mixed"]
SaveDefect = Literal["unsupported_dtype", "missing_key", "extra_key", "wrong_shape", "mixed_prefix"]
LoadIntent = Literal["inference", "training"]


@dataclass(frozen=True)
class LoadCall:
    dtype: torch.dtype
    storage_dtypes: frozenset[str]
    intent: LoadIntent


@pytest.fixture
def reduced_loader(monkeypatch: pytest.MonkeyPatch) -> list[LoadCall]:
    """Permit fixture geometry only; never overwrite the policy's dtype/storage/intent decision."""
    real_validate = mk1_checkpoint_contract.validate_mk1_checkpoint
    real_load = modeling_mk1_vla.load_mk1_vla_from_hf
    calls: list[LoadCall] = []

    def validate(
        model_dir: str | Path,
        *,
        allow_test_only_reduced_geometry: bool = False,
        allowed_storage_dtypes: frozenset[str] = frozenset({"BF16"}),
    ) -> tuple[Mk1CheckpointContract, SafetensorsInventory]:
        return real_validate(
            model_dir,
            allow_test_only_reduced_geometry=True,
            allowed_storage_dtypes=allowed_storage_dtypes,
        )

    def load(
        model_dir: str | Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        runtime_device: str | torch.device | None = None,
        allow_test_only_reduced_geometry: bool = False,
        attention_backend: Literal["torch_sdpa_v1"] = "torch_sdpa_v1",
        route_reduction: Literal["stable_token_segment_sum_v1"] = "stable_token_segment_sum_v1",
        allowed_storage_dtypes: frozenset[str] = frozenset({"BF16"}),
        load_intent: LoadIntent = "inference",
        verified_tensor_storage: VerifiedTrainedTensorStorage | None = None,
    ) -> tuple[Mk1Qwen36VLAForActionGeneration, Mk1Qwen36VLAConfig, Mk1CheckpointContract]:
        assert torch.device(device).type == "cpu"
        assert runtime_device is None or torch.device(runtime_device).type == "cpu"
        calls.append(LoadCall(dtype, allowed_storage_dtypes, load_intent))
        return real_load(
            model_dir,
            dtype=dtype,
            device=device,
            runtime_device=runtime_device,
            allow_test_only_reduced_geometry=True,
            attention_backend=attention_backend,
            route_reduction=route_reduction,
            allowed_storage_dtypes=allowed_storage_dtypes,
            load_intent=load_intent,
            verified_tensor_storage=verified_tensor_storage,
        )

    monkeypatch.setattr(mk1_checkpoint_contract, "validate_mk1_checkpoint", validate)
    monkeypatch.setattr(modeling_mk1_vla, "load_mk1_vla_from_hf", load)
    return calls


def _render_metadata(config: PerceptronIsaacConfig, path: Path) -> None:
    metadata = IsaacMharmonyRenderMetadata.from_config(config).to_json_dict()
    metadata["mharmony_reserved_token_groups"] = [{"name": "coord", "offset": 24, "size": 4}]
    path.write_text(json.dumps(metadata) + "\n")


def _source_policy(root: Path, family: SourceFamily, *, mtp_present: bool = False) -> PerceptronIsaacPolicy:
    config = _source_package(root)
    if family == "legacy_mk1":
        shutil.rmtree(root / "hf_model")
        _write_checkpoint(root / "hf_model", config=_mk1_config(test_geometry=True, mtp_present=mtp_present))
        source_identity = root / "hf_model" / "source_mk1_model_import.json"
        source_identity.write_text('{"fixture": "native-source-identity"}\n')
        config.mk1_source_model_import_sha256 = file_sha256(source_identity)
    # The shared lifecycle fixture overrides this default; restore it to exercise real freezing.
    config.freeze_input_embeddings = PerceptronIsaacConfig.freeze_input_embeddings
    assert config.freeze_input_embeddings is True
    assert config.train_storage_fp32 is True
    _render_metadata(config, root / "native_render_metadata.json")
    finalize_canonical_trained_package(
        config, root, mk1_source_model_import_sha256=config.mk1_source_model_import_sha256
    )
    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert policy._isaac_model is None
    policy._load_backbone()
    assert policy._isaac_model is not None
    assert {p.dtype for p in policy._isaac_model.parameters()} == {torch.bfloat16}

    # Storage-only isolation: explicit private retention avoids the separate public retention gap.
    # It retains sidecars, never weights, and does not replace any integrity validator.
    policy._retain_verified_runtime_assets(root)
    policy._configure_training_parameters()
    embedding_ids = {id(p) for p in policy._isaac_model.get_input_embeddings().parameters()}
    assert embedding_ids
    for parameter in policy._isaac_model.parameters():
        assert parameter.requires_grad is (id(parameter) not in embedding_ids)
        assert parameter.dtype == (torch.float32 if parameter.requires_grad else torch.bfloat16)
    return policy


def _set_trained_values(policy: PerceptronIsaacPolicy, profile: StorageProfile) -> dict[str, torch.Tensor]:
    model = policy._isaac_model
    assert model is not None
    if profile == "all_f32":
        model.float()
    elif profile == "all_bf16":
        model.bfloat16()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.2500001192092896 if parameter.dtype == torch.float32 else 0.123046875)
            if parameter.numel() >= 3:
                parameter.view(-1)[:2].copy_(torch.tensor([-0.0, 0.0], dtype=parameter.dtype))
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    expected_dtypes = {
        "all_bf16": {torch.bfloat16},
        "all_f32": {torch.float32},
        "mixed": {torch.bfloat16, torch.float32},
    }[profile]
    assert {tensor.dtype for tensor in state.values()} == expected_dtypes
    for tensor in state.values():
        if tensor.dtype == torch.float32:
            assert not torch.equal(tensor.view(torch.int32), tensor.bfloat16().float().view(torch.int32))
    return state


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.detach().contiguous().view(torch.uint8), expected.detach().contiguous().view(torch.uint8)
    )


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _saved_state(root: Path) -> dict[str, torch.Tensor]:
    model_root = root / "hf_model"
    index = json.loads((model_root / "model.safetensors.index.json").read_text())
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(model_root / shard)
        assert not state.keys() & tensors.keys()
        assert all(index["weight_map"][name] == shard for name in tensors)
        state.update(tensors)
    assert state.keys() == index["weight_map"].keys()
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in state.values())
    return state


def _live_fsdp2(monkeypatch: pytest.MonkeyPatch) -> None:
    from accelerate.utils.dataclasses import DistributedType

    _install_distributed_state(
        monkeypatch,
        world_size=2,
        state=_StubAcceleratorState(
            distributed_type=DistributedType.FSDP,
            fsdp_plugin=_StubFsdpPlugin(fsdp_version=2, sharding_strategy=None, reshard_after_forward=True),
        ),
    )


@pytest.mark.parametrize("family", ["isaac05", "legacy_mk1"])
@pytest.mark.parametrize("profile", ["all_bf16", "all_f32", "mixed"])
def test_public_trained_save_move_resume_preserves_storage_and_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reduced_loader: list[LoadCall],
    family: SourceFamily,
    profile: StorageProfile,
) -> None:
    source = tmp_path / "source"
    policy = _source_policy(source, family)
    source_files = _files(source)
    source_model_config = (source / "hf_model" / "config.json").read_bytes()
    source_identity = policy.config.mk1_source_model_import_sha256
    expected = _set_trained_values(policy, profile)
    saved = tmp_path / "saved"

    policy.save_pretrained(saved, finalize_package=False)
    policy.finalize_pretrained_package(saved)

    assert not any(PerceptronIsaacPolicy._is_checkpoint_weight_name(p.name) for p in saved.iterdir())
    actual = _saved_state(saved)
    manifest = json.loads((saved / "mk1_trained_package_manifest.json").read_text())
    assert manifest["schema"] == "perceptron_isaac_mk1_canonical_trained_package_manifest_v3"
    assert manifest["metadata"] == {
        "tensor_storage_dtypes": sorted(
            {"F32" if t.dtype == torch.float32 else "BF16" for t in expected.values()}
        )
    }
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        _assert_bits(actual[name], tensor)
    assert (saved / "hf_model" / "config.json").read_bytes() == source_model_config
    assert (saved / "hf_model" / "source_mk1_model_import.json").read_bytes() == source_files[
        "hf_model/source_mk1_model_import.json"
    ]
    assert _files(source) == source_files

    moved = tmp_path / "moved"
    saved.rename(moved)
    source.rename(tmp_path / "retired-source")
    before_reload = _files(moved)
    reduced_loader.clear()
    _live_fsdp2(monkeypatch)
    reloaded = PerceptronIsaacPolicy.from_pretrained(moved, local_files_only=True, strict=True)
    assert reloaded._isaac_model is None, "FP32-storage resume must not eagerly load BF16 first"
    assert reduced_loader == []
    assert reloaded.config.freeze_input_embeddings is True
    assert reloaded.config.mk1_source_model_import_sha256 == source_identity
    groups = reloaded.get_optim_params()

    storage_dtypes = frozenset("F32" if t.dtype == torch.float32 else "BF16" for t in expected.values())
    runtime_dtype = torch.float32 if "F32" in storage_dtypes else torch.bfloat16
    intent: LoadIntent = "training" if runtime_dtype == torch.float32 else "inference"
    assert reduced_loader == [LoadCall(runtime_dtype, storage_dtypes, intent)]
    model = reloaded._isaac_model
    assert model is not None
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        # This is the pre-shard model. Under FSDP2, BF16-only master promotion
        # happens during real accelerator preparation, not in this topology spy.
        _assert_bits(actual[name], tensor.to(runtime_dtype))
    embedding_ids = {id(p) for p in model.get_input_embeddings().parameters()}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert embedding_ids.isdisjoint(trainable_ids)
    assert trainable_ids
    optimized_ids = [id(p) for group in groups for p in group["params"]]
    assert len(optimized_ids) == len(set(optimized_ids))
    assert set(optimized_ids) == trainable_ids

    # Existing inference remains BF16; that must not redefine the on-disk or resume precision.
    reduced_loader.clear()
    inference_config = PerceptronIsaacConfig.from_pretrained(moved)
    inference_config.train_storage_fp32 = False
    inference = PerceptronIsaacPolicy.from_pretrained(
        moved, config=inference_config, local_files_only=True, strict=True
    )
    assert reduced_loader == [LoadCall(torch.bfloat16, storage_dtypes, "inference")]
    assert inference._isaac_model is not None
    inference_state = inference._isaac_model.state_dict()
    assert inference_state.keys() == expected.keys()
    for name, tensor in expected.items():
        _assert_bits(inference_state[name], tensor.bfloat16())
    assert _files(moved) == before_reload, "Loading/training setup must not write the saved package"


@pytest.mark.parametrize("family", ["isaac05", "legacy_mk1"])
@pytest.mark.parametrize(
    "defect", ["unsupported_dtype", "missing_key", "extra_key", "wrong_shape", "mixed_prefix"]
)
def test_invalid_gathered_save_fails_before_destination_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reduced_loader: list[LoadCall],
    family: SourceFamily,
    defect: SaveDefect,
) -> None:
    policy = _source_policy(tmp_path / "source", family)
    state = dict(policy.state_dict())
    key = "_isaac_model.lm_head.weight"
    if defect == "unsupported_dtype":
        state[key] = state[key].to(torch.float16)
        error = "dtype|storage"
    elif defect == "missing_key":
        del state[key]
        error = "missing|inventory"
    elif defect == "extra_key":
        state["_isaac_model.unexpected.weight"] = torch.zeros(1)
        error = "unexpected|extra|inventory"
    elif defect == "mixed_prefix":
        state[f"module.{key}"] = state.pop(key)
        error = "only native backbone|prefix"
    else:
        state[key] = state[key].flatten()[:1]
        error = "shape"
    destination = tmp_path / "destination"
    (destination / "hf_model").mkdir(parents=True)
    (destination / "sentinel").write_bytes(b"untouched")
    (destination / "config.json").write_bytes(b"existing config")
    (destination / "hf_model" / "model-00001-of-00001.safetensors").write_bytes(b"existing weights")
    before = _files(destination)
    order: list[str] = []

    def gather(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        assert module is policy
        assert _files(destination) == before
        order.append("gather")
        return state

    def main_rank() -> bool:
        assert _files(destination) == before
        order.append("rank")
        return True

    monkeypatch.setattr(checkpoint, "full_model_state_dict", gather)
    monkeypatch.setattr(distributed_utils, "is_main_process", main_rank)
    with pytest.raises((RuntimeError, ValueError), match=error):
        policy.save_pretrained(destination)
    assert order == ["gather", "rank"]
    assert _files(destination) == before


def test_runtime_save_omits_only_source_mtp_tensors(tmp_path: Path, reduced_loader: list[LoadCall]) -> None:
    policy = _source_policy(tmp_path / "source", "legacy_mk1", mtp_present=True)
    expected = _set_trained_values(policy, "mixed")
    assert not any(name.startswith("mtp.") for name in expected)
    destination = tmp_path / "saved"
    policy.save_pretrained(destination)
    actual = _saved_state(destination)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        _assert_bits(actual[name], value)
    config = json.loads((destination / "hf_model" / "config.json").read_text())
    assert config["genesis_vla"]["mtp"]["present"] is False
    assert config["text_config"]["mtp_num_hidden_layers"] == 0


def test_non_main_save_gathers_then_returns_without_validating_empty_state_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reduced_loader: list[LoadCall]
) -> None:
    policy = _source_policy(tmp_path / "source", "isaac05")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"untouched")
    before = _files(destination)
    order: list[str] = []

    def gather(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        assert module is policy
        order.append("gather")
        return {}

    def main_rank() -> bool:
        order.append("rank")
        return False

    monkeypatch.setattr(checkpoint, "full_model_state_dict", gather)
    monkeypatch.setattr(distributed_utils, "is_main_process", main_rank)
    policy.save_pretrained(destination)
    assert order[0] == "gather" and order[1:] and set(order[1:]) == {"rank"}
    assert _files(destination) == before


@pytest.mark.parametrize("family", ["isaac05", "legacy_mk1"])
@pytest.mark.parametrize("tamper", ["weight_bytes", "unlisted_file", "missing_file", "source_identity"])
def test_public_trained_reload_authenticates_the_entire_payload_before_loading(
    tmp_path: Path,
    reduced_loader: list[LoadCall],
    family: SourceFamily,
    tamper: Literal["weight_bytes", "unlisted_file", "missing_file", "source_identity"],
) -> None:
    policy = _source_policy(tmp_path / "source", family)
    _set_trained_values(policy, "mixed")
    saved = tmp_path / "saved"
    policy.save_pretrained(saved)
    if tamper == "weight_bytes":
        shard = next((saved / "hf_model").glob("model-*.safetensors"))
        payload = bytearray(shard.read_bytes())
        payload[-1] ^= 1  # Change a tensor value, not its header, dtype, shape or index size.
        shard.write_bytes(payload)
    elif tamper == "unlisted_file":
        (saved / "hf_model" / "unlisted.json").write_text("{}\n")
    elif tamper == "missing_file":
        (saved / "hf_model" / "tokenizer.json").unlink()
    else:
        (saved / "hf_model" / "source_mk1_model_import.json").write_text('{"tampered": true}\n')
    before = _files(saved)
    reduced_loader.clear()

    with pytest.raises((RuntimeError, ValueError), match="digest|inventory|missing|manifest"):
        PerceptronIsaacPolicy.from_pretrained(saved, local_files_only=True)

    assert reduced_loader == [], "Authentication must fail before native model materialization"
    assert _files(saved) == before


@pytest.mark.parametrize("family", ["isaac05", "legacy_mk1"])
def test_original_source_still_rejects_wrong_family_storage_dtype(
    tmp_path: Path, reduced_loader: list[LoadCall], family: SourceFamily
) -> None:
    """Without a trained outer payload declaration, source profile rules remain strict."""
    model_root = tmp_path / "original"
    if family == "isaac05":
        _write_portable_isaac05_checkpoint(model_root)
        wrong_dtype = torch.bfloat16
    else:
        _write_checkpoint(model_root, config=_mk1_config(test_geometry=True, mtp_present=False))
        wrong_dtype = torch.float32
    # Update actual index sizes, but keep the original F32 source export profile unchanged.
    total_bytes = 0
    for shard in model_root.glob("model-*.safetensors"):
        state = {key: tensor.to(wrong_dtype) for key, tensor in load_file(shard).items()}
        total_bytes += sum(t.numel() * t.element_size() for t in state.values())
        save_file(state, shard)
    index_path = model_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["total_size"] = total_bytes
    index_path.write_text(json.dumps(index))
    config = _config(
        hf_model_path=str(model_root),
        train_storage_fp32=False,
        trained_steps=100_000,
        vector_max_states=4,
        max_action_dim=4,
        max_action_horizon=3,
    )
    before = _files(model_root)
    with pytest.raises(RuntimeError, match="dtype"):
        PerceptronIsaacPolicy(config)
    assert reduced_loader == [], "Original wrong-family source must fail before model materialization"
    assert _files(model_root) == before


def _bind_manifest(root: Path, manifest: dict) -> None:
    """Authenticate a deliberately edited fixture, without relaxing production checks."""
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["mk1_trained_package_manifest_sha256"] = None
    manifest["config_sha256"] = canonical_json_sha256(config)
    manifest_path = root / "mk1_trained_package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    config["mk1_trained_package_manifest_sha256"] = file_sha256(manifest_path)
    config_path.write_text(json.dumps(config) + "\n")


def _storage_package(root: Path, profile: StorageProfile) -> None:
    config = _source_package(root)
    _render_metadata(config, root / "native_render_metadata.json")
    total_size = 0
    for shard in (root / "hf_model").glob("model-*.safetensors"):
        state = load_file(shard)
        for name, value in state.items():
            dtype = torch.bfloat16 if profile == "all_bf16" else torch.float32
            if profile == "mixed" and name == "model.language_model.embed_tokens.weight":
                dtype = torch.bfloat16
            state[name] = value.to(dtype)
        total_size += sum(t.numel() * t.element_size() for t in state.values())
        save_file(state, shard)
    index_path = root / "hf_model" / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["total_size"] = total_size
    index_path.write_text(json.dumps(index))
    actual_dtypes = {tensor.dtype for tensor in _saved_state(root).values()}
    assert (
        actual_dtypes
        == {
            "all_bf16": {torch.bfloat16},
            "all_f32": {torch.float32},
            "mixed": {torch.bfloat16, torch.float32},
        }[profile]
    )
    manifest_path = root / "mk1_trained_package_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "perceptron_isaac_mk1_canonical_trained_package_manifest_v3"
    manifest["metadata"] = {
        "tensor_storage_dtypes": {
            "all_bf16": ["BF16"],
            "all_f32": ["F32"],
            "mixed": ["BF16", "F32"],
        }[profile]
    }
    manifest["files_sha256"] = {name: file_sha256(root / name) for name in manifest["files_sha256"]}
    _bind_manifest(root, manifest)


@pytest.mark.parametrize("profile", ["all_bf16", "all_f32", "mixed"])
def test_v3_storage_declaration_accepts_exact_actual_header_set(
    tmp_path: Path, reduced_loader: list[LoadCall], profile: StorageProfile
) -> None:
    root = tmp_path / "package"
    _storage_package(root, profile)
    manifest = json.loads((root / "mk1_trained_package_manifest.json").read_text())
    expected = frozenset(manifest["metadata"]["tensor_storage_dtypes"])
    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert policy._isaac_model is None
    policy._load_backbone()
    assert reduced_loader == [LoadCall(torch.bfloat16, expected, "inference")]


@pytest.mark.parametrize(
    "declaration",
    [None, [], "F32", ["FP16"], ["BF16", "BF16"], ["F32", "BF16"], ["F32", 1], {}, [True]],
)
def test_v3_storage_declaration_rejects_malformed_values_before_loading(
    tmp_path: Path, reduced_loader: list[LoadCall], declaration: object
) -> None:
    root = tmp_path / "package"
    _storage_package(root, "all_f32")
    manifest = json.loads((root / "mk1_trained_package_manifest.json").read_text())
    manifest["metadata"]["tensor_storage_dtypes"] = declaration
    _bind_manifest(root, manifest)
    with pytest.raises(RuntimeError, match="tensor_storage_dtypes|storage declaration"):
        PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert reduced_loader == []


@pytest.mark.parametrize("defect", ["missing_metadata", "missing_dtypes", "extra_metadata", "null_metadata"])
def test_v3_storage_metadata_is_required_and_exact(
    tmp_path: Path, reduced_loader: list[LoadCall], defect: str
) -> None:
    root = tmp_path / "package"
    _storage_package(root, "all_f32")
    manifest = json.loads((root / "mk1_trained_package_manifest.json").read_text())
    if defect == "missing_metadata":
        del manifest["metadata"]
    elif defect == "missing_dtypes":
        manifest["metadata"] = {}
    elif defect == "null_metadata":
        manifest["metadata"] = None
    else:
        manifest["metadata"]["extra"] = True
    _bind_manifest(root, manifest)
    with pytest.raises(RuntimeError, match="schema|metadata|storage declaration"):
        PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert reduced_loader == []


@pytest.mark.parametrize(
    ("profile", "declaration"),
    [("all_f32", ["BF16"]), ("all_f32", ["BF16", "F32"]), ("mixed", ["F32"])],
)
def test_v3_storage_declaration_must_equal_not_merely_allow_header_set(
    tmp_path: Path, reduced_loader: list[LoadCall], profile: StorageProfile, declaration: list[str]
) -> None:
    root = tmp_path / "package"
    _storage_package(root, profile)
    manifest = json.loads((root / "mk1_trained_package_manifest.json").read_text())
    manifest["metadata"]["tensor_storage_dtypes"] = declaration
    _bind_manifest(root, manifest)
    with pytest.raises(RuntimeError, match="storage.*header|header.*storage"):
        PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert reduced_loader == []


def test_v3_storage_metadata_tamper_requires_manifest_digest_binding(
    tmp_path: Path, reduced_loader: list[LoadCall]
) -> None:
    root = tmp_path / "package"
    _storage_package(root, "all_f32")
    manifest_path = root / "mk1_trained_package_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["tensor_storage_dtypes"] = ["BF16", "F32"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert reduced_loader == []


def test_v3_rebound_manifest_cannot_change_source_import_identity(
    tmp_path: Path, reduced_loader: list[LoadCall]
) -> None:
    root = tmp_path / "package"
    _storage_package(root, "all_f32")
    identity_path = root / "hf_model" / "source_mk1_model_import.json"
    identity_path.write_text('{"different_source": true}\n')
    manifest = json.loads((root / "mk1_trained_package_manifest.json").read_text())
    manifest["files_sha256"]["hf_model/source_mk1_model_import.json"] = file_sha256(identity_path)
    _bind_manifest(root, manifest)
    with pytest.raises(RuntimeError, match="source model-import identity.*mismatch"):
        PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert reduced_loader == []


@pytest.mark.parametrize("profile", ["all_f32", "mixed", "all_bf16"])
def test_v2_missing_declaration_retains_original_source_family_rules(
    tmp_path: Path, reduced_loader: list[LoadCall], profile: StorageProfile
) -> None:
    root = tmp_path / "package"
    _storage_package(root, profile)
    manifest = json.loads((root / "mk1_trained_package_manifest.json").read_text())
    manifest["schema"] = "perceptron_isaac_mk1_canonical_trained_package_manifest_v2"
    del manifest["metadata"]
    _bind_manifest(root, manifest)
    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    if profile == "all_f32":
        policy._load_backbone()
        assert reduced_loader == [LoadCall(torch.bfloat16, frozenset({"F32"}), "inference")]
    else:
        with pytest.raises(RuntimeError, match="dtype"):
            policy._load_backbone()
        assert reduced_loader == []


@pytest.mark.parametrize("read_boundary", ["config", "tokenizer"])
def test_lazy_trained_load_rechecks_metadata_after_loader_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reduced_loader: list[LoadCall], read_boundary: str
) -> None:
    root = tmp_path / "package"
    _storage_package(root, "all_f32")
    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert policy._isaac_model is None
    read_name = "read_mk1_config_data" if read_boundary == "config" else "validate_mk1_tokenizer"
    real_read = (
        mk1_checkpoint_contract.read_mk1_config_data
        if read_boundary == "config"
        else mk1_checkpoint_contract.validate_mk1_tokenizer
    )
    metadata = root / "hf_model" / ("config.json" if read_boundary == "config" else "tokenizer.json")
    reads: list[str] = []
    constructions: list[str] = []
    real_build = modeling_mk1_vla._build_config

    def read_then_change(*args: Any, **kwargs: Any) -> Any:
        result = real_read(*args, **kwargs)
        reads.append(read_boundary)
        metadata.write_bytes(metadata.read_bytes() + b"\n")
        return result

    def observe_build(*args: Any, **kwargs: Any) -> Any:
        constructions.append("build")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(modeling_mk1_vla, read_name, read_then_change)
    monkeypatch.setattr(modeling_mk1_vla, "_build_config", observe_build)
    with pytest.raises((RuntimeError, ValueError), match="changed|identity|digest"):
        policy._load_backbone()
    assert reads == [read_boundary]
    assert constructions == [], "Authenticated metadata must be stable before model construction"


def _mock_shard_timestamp_precision(
    monkeypatch: pytest.MonkeyPatch, precision: Literal["seconds", "nanoseconds"]
) -> None:
    real_capture = mk1_checkpoint_contract._capture_shard_identity
    stamp = 1_000_000_000 if precision == "seconds" else 1_000_000_001

    def capture(shard: Path) -> SafetensorsShardIdentity:
        identity = real_capture(shard)
        return replace(identity, mtime_ns=stamp, ctime_ns=stamp)

    monkeypatch.setattr(mk1_checkpoint_contract, "_capture_shard_identity", capture)


@pytest.mark.parametrize("precision", ["seconds", "nanoseconds"])
def test_lazy_trained_load_only_rehashes_second_resolution_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reduced_loader: list[LoadCall],
    precision: Literal["seconds", "nanoseconds"],
) -> None:
    from lerobot.policies.perceptron_isaac import trained_package

    _mock_shard_timestamp_precision(monkeypatch, precision)
    root = tmp_path / "package"
    _storage_package(root, "all_f32")
    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert policy._isaac_model is None
    hashed_shards: list[Path] = []

    def record_hash(path: str | Path) -> str:
        if Path(path).suffix == ".safetensors":
            hashed_shards.append(Path(path))
        return file_sha256(path)

    monkeypatch.setattr(trained_package, "file_sha256", record_hash)
    policy._load_backbone()
    assert policy._isaac_model is not None
    assert reduced_loader == [LoadCall(torch.bfloat16, frozenset({"F32"}), "inference")]
    expected = sorted((root / "hf_model").glob("model-*.safetensors")) if precision == "seconds" else []
    assert sorted(hashed_shards) == expected, "Only coarse shards need one extra payload-hash pass"


@pytest.mark.parametrize("mutation", ["payload", "replacement", "model-config", "index"])
def test_lazy_trained_load_rejects_a_changed_authenticated_shard_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reduced_loader: list[LoadCall], mutation: str
) -> None:
    _mock_shard_timestamp_precision(monkeypatch, "seconds")
    constructions: list[str] = []
    real_build = modeling_mk1_vla._build_config

    def observe_build(*args: Any, **kwargs: Any) -> Any:
        constructions.append("build")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(modeling_mk1_vla, "_build_config", observe_build)
    root = tmp_path / "package"
    _storage_package(root, "all_f32")
    policy = PerceptronIsaacPolicy.from_pretrained(root, local_files_only=True)
    assert policy._isaac_model is None
    assert reduced_loader == []
    shard = next((root / "hf_model").glob("model-*.safetensors"))
    payload = bytearray(shard.read_bytes())
    if mutation == "payload":
        payload[-1] ^= 1  # Keep the safetensors header and physical dtype unchanged.
        shard.write_bytes(payload)
    elif mutation == "replacement":
        replacement = tmp_path / "replacement.safetensors"
        replacement.write_bytes(payload)
        replacement.replace(shard)
    else:
        filename = "config.json" if mutation == "model-config" else "model.safetensors.index.json"
        metadata = root / "hf_model" / filename
        metadata.write_bytes(metadata.read_bytes() + b"\n")
    with pytest.raises((RuntimeError, ValueError), match="changed|identity|digest"):
        policy._load_backbone()
    assert constructions == [], "Changed authenticated weights must fail before model allocation"


def test_raw_mixed_training_requires_verified_outer_storage(
    tmp_path: Path, reduced_loader: list[LoadCall]
) -> None:
    root = tmp_path / "package"
    _storage_package(root, "mixed")
    with pytest.raises(ValueError, match="training load requires"):
        modeling_mk1_vla.load_mk1_vla_from_hf(
            root / "hf_model",
            dtype=torch.float32,
            allowed_storage_dtypes=frozenset({"BF16", "F32"}),
            load_intent="training",
        )
