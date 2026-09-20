from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

AutoModelForCausalLM = pytest.importorskip("transformers").AutoModelForCausalLM


def _write_portable_repo(path: Path) -> Path:
    path.mkdir()
    config = {
        "model_type": "isaac_0_5",
        "architectures": ["Isaac05ForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "configuration_isaac05.Isaac05Config",
            "AutoModelForCausalLM": "modeling_isaac05.Isaac05ForConditionalGeneration",
            "AutoProcessor": "processing_isaac05.Isaac05Processor",
        },
    }
    (path / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    return path


@pytest.fixture(autouse=True)
def _fail_legacy_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    import lerobot.policies.perceptron_isaac.modeling_qwen35_vla as legacy_model

    monkeypatch.setattr(
        legacy_model,
        "load_qwen35_vla_from_hf",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy loader called")),
    )


def _policy(model_path: Path, *, device: str = "cpu") -> PerceptronIsaacPolicy:
    config = PerceptronIsaacConfig(
        device=device,
        hf_model_path=str(model_path),
        n_obs_steps=1,
        chunk_size=2,
        n_action_steps=2,
        action_dim=2,
        proprio_dim=2,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    return PerceptronIsaacPolicy(config)


def _native_portable_package(tmp_path: Path) -> tuple[Path, PerceptronIsaacConfig, torch.Tensor]:
    import hashlib

    from lerobot.policies.perceptron_isaac.mharmony_native import IsaacMharmonyRenderMetadata
    from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
        _write_portable_isaac05_checkpoint,
    )

    model_path = tmp_path / "model"
    _, raw, value = _write_portable_isaac05_checkpoint(model_path)
    # Public tiny geometry keeps the existing neutral-debug opt-in, never a production bypass.
    raw["isaac05_artifact"].update(artifact_kind="neutral_debug", trained_steps=0)
    (model_path / "config.json").write_text(json.dumps(raw))
    policy_path = model_path / "lerobot_policy"
    policy_path.mkdir()
    adapter = policy_path / "isaac_deployment_adapter.json"
    adapter.write_text("{}\n")
    config = _policy(model_path).config
    config.hf_model_path = ".."
    config.artifact_kind = "neutral_debug"
    config.trained_steps = 0
    config.vector_max_states = 4
    config.max_action_dim = 4
    config.max_action_horizon = 3
    config.apply_offset_norm = False
    config.train_storage_fp32 = False
    config.deployment_adapter_sha256 = hashlib.sha256(adapter.read_bytes()).hexdigest()
    metadata = IsaacMharmonyRenderMetadata.from_config(config).to_json_dict()
    metadata["mharmony_reserved_token_groups"] = [{"name": "coord", "offset": 24, "size": 4}]
    (policy_path / "render.json").write_text(json.dumps(metadata))
    config.native_render_metadata_path = "render.json"
    return policy_path, config, value


def _forbid_checkpoint_code(*args: object, **kwargs: object) -> None:
    raise AssertionError("FORBIDDEN_AUTOCLASS_BOUNDARY: checkpoint code must not execute")


def test_policy_loads_portable_repo_through_native_mk1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers import AutoConfig, AutoProcessor, dynamic_module_utils

    from lerobot.policies.perceptron_isaac.modeling_mk1_vla import Mk1Qwen36VLAForActionGeneration
    from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import DiTActionExpertHead

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _forbid_checkpoint_code)
    monkeypatch.setattr(AutoConfig, "from_pretrained", _forbid_checkpoint_code)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", _forbid_checkpoint_code)
    monkeypatch.setattr(dynamic_module_utils, "get_class_from_dynamic_module", _forbid_checkpoint_code)
    policy_path, config, source_value = _native_portable_package(tmp_path)
    # What this test owns is that loading a portable repo never touches CUDA. Asserting an
    # uninitialized process would instead assert a global property it does not own: in a full
    # run a sibling test that legitimately exercises CUDA (test_modeling_qwen36_moe.py's [cuda]
    # cases) boots the context first. So measure a delta across the load instead.
    cuda_initialized_before = torch.cuda.is_initialized()
    cuda_allocated_before = torch.cuda.memory_allocated() if cuda_initialized_before else 0
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    model = policy._isaac_model
    assert isinstance(model, Mk1Qwen36VLAForActionGeneration)
    assert isinstance(model.action_expert, DiTActionExpertHead)
    assert model.action_expert.args.rtc_max_delay_steps == 12
    assert model.action_expert.args.rtc_probability == 0.5
    assert model.config.action_expert["type"] == "dit"
    assert policy.config.action_expert_type == "molmoact"
    assert (policy.config.artifact_kind, policy.config.trained_steps) == ("neutral_debug", 0)
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(buffer.device.type != "meta" for buffer in model.buffers())
    assert all(
        torch.equal(parameter, source_value.bfloat16().expand_as(parameter))
        for parameter in model.parameters()
    )
    assert source_value.item() != source_value.bfloat16().float().item()
    assert not policy.training and not model.training
    assert torch.cuda.is_initialized() == cuda_initialized_before
    if cuda_initialized_before:
        assert torch.cuda.memory_allocated() == cuda_allocated_before
    assert not list(policy_path.parent.glob("*.py"))


@pytest.mark.parametrize("defect", ["adapter", "path", "identity", "metadata", "dtype"])
def test_native_portable_public_load_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    import lerobot.policies.perceptron_isaac.modeling_mk1_vla as native

    policy_path, config, _ = _native_portable_package(tmp_path)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _forbid_checkpoint_code)
    monkeypatch.setattr(native, "Mk1Qwen36VLAForActionGeneration", _forbid_checkpoint_code)
    expected = ""
    if defect == "adapter":
        config.deployment_adapter_sha256 = "0" * 64
        expected = "digest mismatch"
    elif defect == "path":
        config.hf_model_path = "../.."
        expected = "escapes the package root"
    elif defect == "identity":
        raw_path = policy_path.parent / "config.json"
        raw = json.loads(raw_path.read_text())
        raw["isaac05_artifact"].update(artifact_kind="trained_policy", trained_steps=100000)
        raw_path.write_text(json.dumps(raw))
        expected = "artifact identity mismatch"
    elif defect == "metadata":
        raw_path = policy_path.parent / "config.json"
        raw = json.loads(raw_path.read_text())
        raw["action_expert"]["rtc_probability"] = 0.1
        raw_path.write_text(json.dumps(raw))
        expected = "identical"
    else:
        from safetensors.torch import load_file, save_file

        shard = next(policy_path.parent.glob("model-*.safetensors"))
        state = load_file(shard)
        key = next(iter(state))
        state[key] = state[key].bfloat16()
        save_file(state, shard)
        expected = "total_size|dtype"
    with pytest.raises((RuntimeError, ValueError), match=expected):
        PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)


def test_policy_allows_direct_parent_portable_model_repo(tmp_path: Path) -> None:
    model_path = _write_portable_repo(tmp_path / "artifact")
    policy_path = model_path / "lerobot_policy"
    policy_path.mkdir()
    config = PerceptronIsaacConfig(hf_model_path="..")

    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, policy_path)

    assert config.hf_model_path == str(model_path)


def test_policy_allows_portable_parent_fast_processor(tmp_path: Path) -> None:
    model_path = _write_portable_repo(tmp_path / "artifact")
    fast_processor_path = model_path / "fast_processor_pinned"
    fast_processor_path.mkdir()
    policy_path = model_path / "lerobot_policy"
    policy_path.mkdir()
    config = PerceptronIsaacConfig(
        hf_model_path="..",
        fast_processor_path="../fast_processor_pinned",
        fast_processor_tree_sha256="0" * 64,
    )

    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, policy_path)

    assert config.fast_processor_path == str(fast_processor_path)


def test_portable_package_verifies_adapter_without_legacy_manifest(tmp_path: Path) -> None:
    model_path = _write_portable_repo(tmp_path / "artifact")
    policy_path = model_path / "lerobot_policy"
    policy_path.mkdir()
    adapter_path = policy_path / "isaac_deployment_adapter.json"
    adapter_path.write_text("{}\n")
    import hashlib

    config = PerceptronIsaacConfig(
        hf_model_path="..",
        deployment_adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
    )
    PerceptronIsaacPolicy._resolve_checkpoint_local_paths(config, policy_path)

    PerceptronIsaacPolicy._verify_packaged_contract_digests(config, policy_path)


def test_policy_refuses_unsharded_dense_portable_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense full-parameter Isaac-0.5 training is refused for the capacity reason, not a dtype one."""
    model_path = _write_portable_repo(tmp_path / "model")
    called = False

    def load_auto(*args: object, **kwargs: object) -> nn.Module:
        nonlocal called
        called = True
        return nn.Linear(2, 2)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_auto)
    policy = _policy(model_path)
    assert not policy.config.train_expert_only and not policy.config.use_peft

    with pytest.raises(RuntimeError, match="dense full-parameter training is unsupported") as refusal:
        policy._load_backbone(training=True)
    message = str(refusal.value)
    # The retired message blamed an FP32/BF16 dispatcher conflict that no longer exists.
    assert "grouped_mm" not in message
    assert "DDP replicates" in message
    assert "UNVERIFIED" in message
    assert "train_expert_only=true or use_peft=true" in message
    assert not called


def test_policy_admits_expert_only_portable_training(tmp_path: Path) -> None:
    """A bounded trainable set trains: FP32 storage for the expert, BF16 frozen MoE experts."""
    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_expert_only = True
    config.train_storage_fp32 = True
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)

    groups = policy.get_optim_params()

    parameters = dict(policy._isaac_model.named_parameters())
    trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith("model.action_expert.") for name in trainable)
    assert all(
        parameter.dtype is torch.float32 for group in groups for parameter in group["params"]
    )
    assert {id(parameter) for group in groups for parameter in group["params"]} == {
        id(parameters[name]) for name in trainable
    }
    fused_expert_names = [name for name in parameters if ".mlp.experts." in name]
    assert fused_expert_names
    # The 32.2B fused MoE experts stay frozen in BF16, so training keeps the production
    # grouped_mm dispatch instead of the FP32 eager expert loop.
    assert all(parameters[name].dtype is torch.bfloat16 for name in fused_expert_names)
    assert not any(parameters[name].requires_grad for name in fused_expert_names)


def test_policy_refuses_dense_training_at_optimizer_construction(tmp_path: Path) -> None:
    """The refusal fires at get_optim_params, the site LeRobot reaches before the first forward."""
    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_expert_only = False
    config.use_peft = False
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)

    with pytest.raises(RuntimeError, match="dense full-parameter training is unsupported"):
        policy.get_optim_params()


@dataclass(frozen=True)
class _StubFsdpPlugin:
    """Stands in for accelerate's FullyShardedDataParallelPlugin sharding surface."""

    fsdp_version: int
    sharding_strategy: object
    reshard_after_forward: object


@dataclass(frozen=True)
class _StubAcceleratorState:
    distributed_type: object
    fsdp_plugin: object | None


def _install_distributed_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int,
    state: object | None,
) -> None:
    """Make `detect_full_shard_topology` observe a chosen live topology.

    `state=None` means accelerate's state reports itself uninitialised.
    """
    import accelerate.state as accelerate_state

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(accelerate_state, "is_initialized", lambda: state is not None)
    monkeypatch.setattr(accelerate_state, "AcceleratorState", lambda *a, **k: state)


def test_policy_admits_dense_training_under_fsdp_full_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dense full-parameter training is admitted when the live state is FSDP FULL_SHARD."""
    from accelerate.utils.dataclasses import DistributedType
    from torch.distributed.fsdp import ShardingStrategy

    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_expert_only = False
    config.use_peft = False
    config.train_storage_fp32 = False
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    _install_distributed_state(
        monkeypatch,
        world_size=16,
        state=_StubAcceleratorState(
            distributed_type=DistributedType.FSDP,
            fsdp_plugin=_StubFsdpPlugin(
                fsdp_version=1,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                reshard_after_forward=True,
            ),
        ),
    )

    groups = policy.get_optim_params()

    trainable = {
        name for name, parameter in policy._isaac_model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert any(".mlp.experts." in name for name in trainable)
    assert {id(parameter) for group in groups for parameter in group["params"]}


def test_policy_refuses_dense_training_under_ddp_multi_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DDP replicates parameters, so the capacity refusal stands even at world_size 16."""
    from accelerate.utils.dataclasses import DistributedType

    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_expert_only = False
    config.use_peft = False
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    _install_distributed_state(
        monkeypatch,
        world_size=16,
        state=_StubAcceleratorState(distributed_type=DistributedType.MULTI_GPU, fsdp_plugin=None),
    )

    with pytest.raises(RuntimeError, match="dense full-parameter training is unsupported"):
        policy.get_optim_params()


def test_policy_refuses_dense_training_when_shard_strategy_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed: an FSDP state whose sharding surface is not FULL_SHARD is refused."""
    from accelerate.utils.dataclasses import DistributedType

    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_expert_only = False
    config.use_peft = False
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    _install_distributed_state(
        monkeypatch,
        world_size=16,
        state=_StubAcceleratorState(
            distributed_type=DistributedType.FSDP,
            fsdp_plugin=_StubFsdpPlugin(
                fsdp_version=1,
                sharding_strategy="SHARD_GRAD_OP",
                reshard_after_forward=False,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="dense full-parameter training is unsupported"):
        policy.get_optim_params()
