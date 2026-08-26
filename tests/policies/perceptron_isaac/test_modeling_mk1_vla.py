from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import lerobot.policies.perceptron_isaac.modeling_mk1_vla as modeling_mk1_vla
import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as modeling_qwen35_vla
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    Mk1CheckpointContractError,
    finalize_mk1_runtime_checkpoint_layout,
    validate_mk1_checkpoint,
)
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    QUALIFIED_TORCH_CUDA_ABI,
    QUALIFIED_TRANSFORMERS_VERSION,
    TORCH_SDPA_ATTENTION_BACKEND,
    load_mk1_vla_from_hf,
)
from lerobot.policies.perceptron_isaac.modeling_qwen36_moe import (
    DETERMINISTIC_ROUTE_REDUCTION,
    GenesisQwen36Attention,
    GenesisQwen36TextRotaryEmbedding,
    GenesisQwen36VisionAttention,
)
from lerobot.policies.perceptron_isaac.tensor_stream import (
    ALL_TYPES,
    Event,
    TensorStream,
    TextType,
    VectorType,
    create_stream,
)
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _config as _mk1_config,
    _write_checkpoint,
)


def _action_stream() -> TensorStream:
    events = [
        Event.from_text_tokens(
            torch.tensor([3, 4]),
            time=(0.0, 0.0),
            type=TextType.text,
            role="user",
        ),
        Event(
            data=torch.tensor([[0.25, -0.5, 0.75, 0.0]]),
            time=(0.0, 0.0),
            type=VectorType.vector,
            role="user",
            dims_virtual=[1],
            dims_real=[1],
            idx_range=(0, 1),
        ),
        Event.from_text_tokens(
            torch.tensor([0]),
            time=(0.0, 0.0),
            type=TextType.action_c,
            role="assistant",
        ),
        Event.from_text_tokens(
            torch.tensor([2]),
            time=(0.0, 0.0),
            type=TextType.control,
            role="assistant",
        ),
    ]
    return TensorStream([create_stream(events, ALL_TYPES, schedule=False)])


def _replace_with_same_bytes(path: Path, replacement: Path) -> None:
    shutil.copy2(path, replacement)
    replacement.replace(path)


def _reduced_mk1_contract(raw: dict | None = None) -> Mk1CheckpointContract:
    return Mk1CheckpointContract.parse_allowlisted(
        raw if raw is not None else _mk1_config(test_geometry=True),
        allow_test_only_reduced_geometry=True,
    )


def _production_mk1_contract(raw: dict | None = None) -> Mk1CheckpointContract:
    return replace(_reduced_mk1_contract(raw), test_only_reduced_geometry=False)


def _qualify_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    capability: tuple[int, int] | None = None,
    device_name: str | None = None,
) -> None:
    monkeypatch.setattr(modeling_mk1_vla.torch, "__version__", QUALIFIED_TORCH_CUDA_ABI)
    monkeypatch.setattr(modeling_mk1_vla.torch.version, "cuda", "12.8")
    if capability is not None:
        monkeypatch.setattr(
            modeling_mk1_vla.torch.cuda,
            "get_device_capability",
            lambda device: capability,
        )
    if device_name is not None:
        monkeypatch.setattr(
            modeling_mk1_vla.torch.cuda,
            "get_device_name",
            lambda device: device_name,
        )


def test_mk1_reuses_pr3154_vla_shell_descriptors_verbatim() -> None:
    for name in ("_embed_text", "_embed_vector", "_embed_vision", "embed_stream", "forward"):
        assert (
            modeling_mk1_vla.Mk1Qwen36VLAModel.__dict__[name]
            is modeling_qwen35_vla.Qwen35VLAModel.__dict__[name]
        )
    for name in ("get_input_embeddings", "action_expert", "sample_action", "train_forward"):
        assert (
            modeling_mk1_vla.Mk1Qwen36VLAForActionGeneration.__dict__[name]
            is modeling_qwen35_vla.Qwen35VLAForActionGeneration.__dict__[name]
        )


def test_mk1_loader_constructs_only_from_checkpoint_and_leaves_no_meta_tensors(tmp_path) -> None:
    model_dir = tmp_path / "mk1"
    expected_contract = _write_checkpoint(model_dir)

    model, config, contract = load_mk1_vla_from_hf(
        model_dir,
        dtype=torch.bfloat16,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )

    assert contract == expected_contract
    assert config.vector_max_states == contract.vector_encoder.max_states
    assert config.action_expert["action_horizon"] == contract.action_expert.action_horizon
    assert model.model.action_expert.action_horizon == contract.action_expert.action_horizon
    assert model.model.action_expert.action_dim == contract.action_expert.action_dim
    assert model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert not any(parameter.device.type == "meta" for parameter in model.parameters())
    assert not any(buffer.device.type == "meta" for buffer in model.buffers())
    assert not any(key.startswith("mtp.") for key in model.state_dict())
    assert config.text_config._attn_implementation == "sdpa"
    assert config.vision_config._attn_implementation == "sdpa"
    assert config.text_config._mk1_route_reduction == DETERMINISTIC_ROUTE_REDUCTION
    assert isinstance(model.model.language_model.rotary_emb, GenesisQwen36TextRotaryEmbedding)
    assert model.model.language_model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert model.model.visual.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert not any("rotary_emb.inv_freq" in key for key in model.state_dict())
    assert all(layer.mlp.deterministic_route_reduction for layer in model.model.language_model.layers)


def test_mk1_runtime_checkpoint_layout_real_loader_survives_move(tmp_path) -> None:
    source = tmp_path / "source-mk1"
    source_contract = _write_checkpoint(source)
    assert source_contract.mtp.present

    model, _, _ = load_mk1_vla_from_hf(
        source,
        dtype=torch.bfloat16,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )
    runtime_state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    assert runtime_state
    assert not any(key.startswith("mtp.") for key in runtime_state)

    runtime = tmp_path / "runtime-mk1"
    runtime.mkdir()
    shutil.copy2(source / "config.json", runtime / "config.json")
    shutil.copy2(source / "tokenizer.json", runtime / "tokenizer.json")
    save_file(runtime_state, runtime / "model.safetensors")

    finalize_mk1_runtime_checkpoint_layout(runtime)

    assert not (runtime / "model.safetensors").exists()
    index_path = runtime / "model.safetensors.index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text())
    assert set(index["weight_map"]) == set(runtime_state)
    assert set(index["weight_map"].values()) == {"model-00001-of-00001.safetensors"}
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in runtime_state.values()
    )

    runtime_config = json.loads((runtime / "config.json").read_text())
    assert runtime_config["genesis_vla"]["mtp"] == {
        "present": False,
        "physical_layers": 0,
        "rollout_steps": 0,
        "action_runtime": "exclude",
    }
    assert runtime_config["text_config"]["mtp_num_hidden_layers"] == 0

    moved = tmp_path / "moved-runtime-mk1"
    runtime.rename(moved)
    source.rename(tmp_path / "retired-source-mk1")

    moved_contract, inventory = validate_mk1_checkpoint(
        moved,
        allow_test_only_reduced_geometry=True,
    )
    assert not moved_contract.mtp.present
    assert set(inventory.tensors) == set(runtime_state)

    reloaded, _, loaded_contract = load_mk1_vla_from_hf(
        moved,
        dtype=torch.bfloat16,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )
    assert loaded_contract == moved_contract
    reloaded_state = reloaded.state_dict()
    assert set(reloaded_state) == set(runtime_state)
    for key, expected in runtime_state.items():
        torch.testing.assert_close(reloaded_state[key], expected)


def test_mk1_config_normalizes_real_export_omissions_to_qualified_runtime_defaults() -> None:
    raw = _mk1_config(test_geometry=True)
    common = {
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": True,
        "chunk_size_feed_forward": 0,
        "is_encoder_decoder": False,
    }
    for key in common:
        assert key not in raw
        assert key not in raw["text_config"]
        assert key not in raw["vision_config"]
    for key in ("output_router_logits", "norm_topk_prob"):
        assert key not in raw["text_config"]

    contract = Mk1CheckpointContract.parse_allowlisted(
        raw,
        allow_test_only_reduced_geometry=True,
    )
    config = modeling_mk1_vla._build_config(raw, contract)

    for key, expected in common.items():
        assert getattr(config, key) == expected
        assert getattr(config.text_config, key) == expected
        assert getattr(config.vision_config, key) == expected
    assert config.text_config.use_cache is True
    assert config.text_config.output_router_logits is False
    assert config.text_config.norm_topk_prob is True
    # Loading may add runtime defaults to its private copy, never the strict
    # checkpoint JSON object that was validated and authenticated.
    assert "output_attentions" not in raw
    assert "norm_topk_prob" not in raw["text_config"]


def test_mk1_loader_can_cast_checkpoint_storage_for_training(tmp_path) -> None:
    model_dir = tmp_path / "mk1"
    _write_checkpoint(model_dir)

    model, _, _ = load_mk1_vla_from_hf(
        model_dir,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )

    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_mk1_loader_applies_explicit_runtime_sdpa_to_text_and_vision(tmp_path) -> None:
    model_dir = tmp_path / "mk1"
    _write_checkpoint(model_dir)

    model, config, _ = load_mk1_vla_from_hf(
        model_dir,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
        attention_backend=TORCH_SDPA_ATTENTION_BACKEND,
    )

    assert config._attn_implementation == "sdpa"
    assert config.text_config._attn_implementation == "sdpa"
    assert config.vision_config._attn_implementation == "sdpa"
    assert model.model.language_model.config._attn_implementation == "sdpa"
    assert model.model.visual.config._attn_implementation == "sdpa"
    assert all(
        isinstance(layer.self_attn, GenesisQwen36Attention)
        for layer in model.model.language_model.layers
        if layer.layer_type == "full_attention"
    )
    assert all(isinstance(block.attn, GenesisQwen36VisionAttention) for block in model.model.visual.blocks)


def test_mk1_loader_rejects_unqualified_transformers_abi_before_checkpoint_read(
    tmp_path, monkeypatch
) -> None:
    reads = 0

    def unexpected_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("checkpoint must not be read before the Transformers ABI gate")

    monkeypatch.setattr(modeling_mk1_vla.transformers, "__version__", "5.5.5")
    monkeypatch.setattr(modeling_mk1_vla, "read_mk1_config_data", unexpected_read)

    with pytest.raises(RuntimeError, match=rf"requires.*{QUALIFIED_TRANSFORMERS_VERSION}.*5\.5\.5"):
        load_mk1_vla_from_hf(tmp_path / "missing")

    assert reads == 0


@pytest.mark.parametrize(
    ("torch_version", "cuda_version"),
    [
        ("2.11.0+cu128", "12.8"),
        ("2.10.0+cu129", "12.9"),
        ("2.10.0+cu129", "12.8"),
        ("2.10.0", None),
    ],
)
def test_trained_cuda_loader_rejects_unqualified_torch_abi_before_safetensor_read_or_allocation(
    tmp_path, monkeypatch, torch_version, cuda_version
) -> None:
    raw = _mk1_config(test_geometry=True)
    full_contract = _production_mk1_contract(raw)
    safetensor_reads = []
    allocations = []

    def unexpected_safetensor_read(*args, **kwargs):
        safetensor_reads.append((args, kwargs))
        raise AssertionError("safetensors must not be read before the qualified Torch ABI gate")

    def unexpected_allocation(*args, **kwargs):
        allocations.append((args, kwargs))
        raise AssertionError("model allocation must not run before the qualified Torch ABI gate")

    monkeypatch.setattr(
        modeling_mk1_vla,
        "read_mk1_config_data",
        lambda *args, **kwargs: (raw, full_contract),
    )
    monkeypatch.setattr(
        modeling_mk1_vla,
        "read_and_validate_safetensors_index",
        unexpected_safetensor_read,
    )
    monkeypatch.setattr(
        modeling_mk1_vla,
        "Mk1Qwen36VLAForActionGeneration",
        unexpected_allocation,
    )
    monkeypatch.setattr(modeling_mk1_vla.torch, "__version__", torch_version)
    monkeypatch.setattr(modeling_mk1_vla.torch.version, "cuda", cuda_version)

    with pytest.raises(RuntimeError, match="requires.*parity-qualified") as error:
        load_mk1_vla_from_hf(tmp_path / "config-is-mocked", device="cuda")

    assert QUALIFIED_TORCH_CUDA_ABI in str(error.value)
    assert safetensor_reads == []
    assert allocations == []


def test_qualified_torch_cuda_gate_exempts_only_nonproduction_execution(monkeypatch) -> None:
    reduced_contract = _reduced_mk1_contract()
    full_contract = _production_mk1_contract()
    neutral_contract = replace(
        full_contract,
        artifact=replace(
            full_contract.artifact,
            artifact_kind="neutral_debug",
            trained_steps=0,
        ),
    )
    monkeypatch.setattr(modeling_mk1_vla.torch, "__version__", "2.11.0+cu128")
    monkeypatch.setattr(modeling_mk1_vla.torch.version, "cuda", "12.8")

    modeling_mk1_vla._require_qualified_torch_cuda_abi(reduced_contract, "cuda")
    modeling_mk1_vla._require_qualified_torch_cuda_abi(neutral_contract, "cuda")
    modeling_mk1_vla._require_qualified_torch_cuda_abi(full_contract, "cpu")


def test_qualified_torch_cuda_gate_accepts_exact_parity_abi(monkeypatch) -> None:
    contract = _production_mk1_contract()
    _qualify_cuda_runtime(
        monkeypatch,
        capability=(9, 0),
        device_name="NVIDIA H100 80GB HBM3",
    )

    modeling_mk1_vla._require_qualified_torch_cuda_abi(contract, "cuda:0")


@pytest.mark.parametrize(
    ("capability", "device_name", "message"),
    [
        pytest.param(
            (8, 0),
            "NVIDIA H100 80GB HBM3",
            r"requires.*Hopper SM90.*8\.0",
            id="unqualified_device",
        ),
        pytest.param(
            (9, 0),
            "NVIDIA H200",
            r"requires.*NVIDIA H100.*NVIDIA H200",
            id="unqualified_hopper_family",
        ),
    ],
)
def test_qualified_torch_cuda_gate_rejects_unqualified_hardware(
    monkeypatch,
    capability: tuple[int, int],
    device_name: str,
    message: str,
) -> None:
    contract = _production_mk1_contract()
    _qualify_cuda_runtime(
        monkeypatch,
        capability=capability,
        device_name=device_name,
    )

    with pytest.raises(RuntimeError, match=message):
        modeling_mk1_vla._require_qualified_torch_cuda_abi(contract, "cuda:0")


def test_mk1_loader_rejects_unknown_runtime_attention_backend(tmp_path) -> None:
    model_dir = tmp_path / "mk1"
    _write_checkpoint(model_dir)

    with pytest.raises(ValueError, match="Unsupported MK1 attention backend"):
        load_mk1_vla_from_hf(
            model_dir,
            dtype=torch.float32,
            device="cpu",
            allow_test_only_reduced_geometry=True,
            attention_backend="flash_attention_3",
        )


def test_mk1_loader_builds_from_the_same_config_object_it_validated(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "mk1"
    _write_checkpoint(model_dir)
    reads = 0
    original = modeling_mk1_vla.read_mk1_config_data

    def counted_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(modeling_mk1_vla, "read_mk1_config_data", counted_read)
    modeling_mk1_vla.load_mk1_vla_from_hf(
        model_dir,
        dtype=torch.bfloat16,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )

    assert reads == 1


def test_mk1_loader_rejects_shard_replaced_after_header_validation(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "mk1"
    _write_checkpoint(model_dir)
    original_read = modeling_mk1_vla.read_and_validate_safetensors_index
    stream_opens = []

    def read_then_replace(*args, **kwargs):
        inventory = original_read(*args, **kwargs)
        shard = inventory.shards[0]
        _replace_with_same_bytes(shard, tmp_path / "replacement.safetensors")
        return inventory

    original_safe_open = modeling_mk1_vla.safe_open

    def opening_spy(*args, **kwargs):
        stream_opens.append(args[0])
        return original_safe_open(*args, **kwargs)

    monkeypatch.setattr(modeling_mk1_vla, "read_and_validate_safetensors_index", read_then_replace)
    monkeypatch.setattr(modeling_mk1_vla, "safe_open", opening_spy)

    with pytest.raises(Mk1CheckpointContractError, match="changed after validation"):
        load_mk1_vla_from_hf(
            model_dir,
            dtype=torch.bfloat16,
            device="cpu",
            allow_test_only_reduced_geometry=True,
        )

    assert stream_opens == []


def test_mk1_loader_rejects_shard_replaced_during_stream(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "mk1"
    _write_checkpoint(model_dir)
    original_safe_open = modeling_mk1_vla.safe_open
    replaced = False

    class ReplacingHandle:
        def __init__(self, handle, shard: Path) -> None:
            self.handle = handle
            self.shard = shard

        def keys(self):
            return self.handle.keys()

        def get_tensor(self, key):
            nonlocal replaced
            value = self.handle.get_tensor(key)
            if not replaced:
                _replace_with_same_bytes(self.shard, tmp_path / "replacement.safetensors")
                replaced = True
            return value

    class ReplacingContext:
        def __init__(self, shard, *args, **kwargs) -> None:
            self.shard = Path(shard)
            self.context = original_safe_open(shard, *args, **kwargs)

        def __enter__(self):
            return ReplacingHandle(self.context.__enter__(), self.shard)

        def __exit__(self, *args):
            return self.context.__exit__(*args)

    monkeypatch.setattr(modeling_mk1_vla, "safe_open", ReplacingContext)

    with pytest.raises(Mk1CheckpointContractError, match="changed after validation"):
        load_mk1_vla_from_hf(
            model_dir,
            dtype=torch.bfloat16,
            device="cpu",
            allow_test_only_reduced_geometry=True,
        )

    assert replaced


def test_mk1_loader_accepts_hf_snapshot_blob_symlinks(tmp_path) -> None:
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

    model, _, _ = load_mk1_vla_from_hf(
        model_dir,
        dtype=torch.bfloat16,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_mk1_loader_rejects_tokenizer_mutation_before_model_allocation(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "mk1"
    _write_checkpoint(model_dir)
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    tokenizer["added_tokens"][2]["id"] = 6
    tokenizer_path.write_text(json.dumps(tokenizer))
    allocations = []

    def allocation_spy(*args, **kwargs):
        allocations.append((args, kwargs))
        raise AssertionError("model allocation must not run for an invalid tokenizer")

    monkeypatch.setattr(modeling_mk1_vla, "Mk1Qwen36VLAForActionGeneration", allocation_spy)

    with pytest.raises(ValueError, match="image_pad.*expected 4"):
        load_mk1_vla_from_hf(
            model_dir,
            dtype=torch.bfloat16,
            device="cpu",
            allow_test_only_reduced_geometry=True,
        )

    assert allocations == []


def test_reduced_checkpoint_runs_existing_vla_action_path_deterministically(tmp_path) -> None:
    model_dir = tmp_path / "mk1"
    contract = _write_checkpoint(model_dir)
    model, _, _ = load_mk1_vla_from_hf(
        model_dir,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )
    stream = _action_stream()

    torch.manual_seed(123)
    first = model.sample_action(
        stream,
        num_steps=1,
        action_dim=2,
        num_action_steps=2,
    )
    torch.manual_seed(123)
    second = model.sample_action(
        stream,
        num_steps=1,
        action_dim=2,
        num_action_steps=2,
    )

    assert first.shape == (1, 2, 2)
    assert contract.action_expert.action_dim == 4
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
