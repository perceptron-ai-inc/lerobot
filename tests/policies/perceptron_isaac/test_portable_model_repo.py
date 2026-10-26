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


def test_native_policy_loads_with_default_offset_norm(tmp_path: Path) -> None:
    policy_path, config, source_value = _native_portable_package(tmp_path)
    config.save_pretrained(policy_path)
    config_path = policy_path / "config.json"
    payload = json.loads(config_path.read_text())
    # Exercise the schema default, not the fixture's explicit legacy false value.
    payload.pop("apply_offset_norm", None)
    config_path.write_text(json.dumps(payload))

    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, local_files_only=True)

    assert policy._isaac_model is not None
    assert all(
        torch.equal(parameter, source_value.bfloat16().expand_as(parameter))
        for parameter in policy._isaac_model.parameters()
    )


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
    # Sharded dense training is verified on 16 H100s, so the refusal must point at the
    # working route -- upstream LeRobot's own FSDP2 flags -- instead of calling it unverified.
    assert "UNVERIFIED" not in message
    assert "--parallelism.dp_shard" in message
    assert "--accelerator.mixed_precision=bf16" in message
    assert "accelerate FSDP config" not in message
    assert "train_expert_only=true or use_peft=true" not in message
    assert not called


@pytest.mark.parametrize("mode", ["train_expert_only", "use_peft"])
def test_policy_rejects_mutated_deferred_training_modes(tmp_path: Path, mode: str) -> None:
    _, config, _ = _native_portable_package(tmp_path)
    if mode == "train_expert_only":
        config.train_expert_only = True
    else:
        config.use_peft = True
    config.train_storage_fp32 = True
    policy = PerceptronIsaacPolicy(config)
    with pytest.raises(RuntimeError, match="supports only dense training"):
        policy.get_optim_params()
    with pytest.raises(RuntimeError, match="supports only dense training"):
        policy({})
    assert policy._isaac_model is None


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
    # Accelerate upcasts the sharded FSDP flat parameters to FP32 only when its own mixed
    # precision is on, so the FP32-master decision reads this field off the live state.
    mixed_precision: str = "bf16"


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


def test_policy_refuses_dense_training_under_fsdp1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FSDP1 must fail before optimizer collection, not silently take an unqualified path."""
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

    with pytest.raises(RuntimeError, match="FSDP1 training is unsupported"):
        policy.get_optim_params()


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


def _full_shard_state(sharding_strategy: object, mixed_precision: str) -> _StubAcceleratorState:
    """A live accelerate state that reports FSDP FULL_SHARD at a chosen mixed precision."""
    from accelerate.utils.dataclasses import DistributedType

    return _StubAcceleratorState(
        distributed_type=DistributedType.FSDP,
        fsdp_plugin=_StubFsdpPlugin(
            fsdp_version=2,
            sharding_strategy=sharding_strategy,
            reshard_after_forward=True,
        ),
        mixed_precision=mixed_precision,
    )


def test_native_fp32_masters_keep_source_bits_before_fsdp2_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native F32 parameters preserve source values before FSDP2 shards their storage."""
    from torch.distributed.fsdp import ShardingStrategy

    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_expert_only = False
    config.use_peft = False
    config.train_storage_fp32 = True
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    _install_distributed_state(
        monkeypatch,
        world_size=16,
        state=_full_shard_state(ShardingStrategy.FULL_SHARD, "bf16"),
    )

    groups = policy.get_optim_params()

    assert groups
    assert policy._isaac_model is not None
    trainable = [
        (name, parameter)
        for name, parameter in policy._isaac_model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all(parameter.dtype is torch.float32 for _, parameter in trainable)
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 0.12345679104328156)) for _, parameter in trainable
    )


def test_fp32_masters_refuse_full_shard_without_accelerate_mixed_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse master-promotion deferral when accelerate mixed precision is disabled."""
    from torch.distributed.fsdp import ShardingStrategy

    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_expert_only = False
    config.use_peft = False
    config.train_storage_fp32 = True
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    _install_distributed_state(
        monkeypatch,
        world_size=16,
        state=_full_shard_state(ShardingStrategy.FULL_SHARD, "no"),
    )

    with pytest.raises(RuntimeError, match="needs accelerate mixed precision enabled"):
        policy.get_optim_params()


def test_fp32_masters_are_materialized_eagerly_without_sharding(tmp_path: Path) -> None:
    """Unsharded training keeps the existing eager promotion: there is no shard to defer to."""
    policy_path, config, _ = _native_portable_package(tmp_path)
    from tests.policies.perceptron_isaac.test_qwen35_training_lifecycle import reduced_native_policy

    policy = reduced_native_policy(tmp_path / "in-memory")
    groups = policy.get_optim_params()
    assert groups
    assert all(parameter.dtype is torch.float32 for group in groups for parameter in group["params"])


def _record_backbone_load_device(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the device `_load_backbone` asks for; really load the tiny model on the CPU host."""
    import lerobot.policies.perceptron_isaac.modeling_mk1_vla as mk1

    real_loader = mk1.load_mk1_vla_from_hf
    devices: list[str] = []

    def recording_loader(*args: object, device: str, **kwargs: object):
        devices.append(str(device))
        return real_loader(*args, device="cpu", **kwargs)

    monkeypatch.setattr(mk1, "load_mk1_vla_from_hf", recording_loader)
    return devices


def test_fsdp2_dense_training_builds_the_backbone_in_host_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under live FSDP2 the backbone is built in host RAM, never on the run device.

    The backbone loads lazily in get_optim_params, after make_policy returned. accelerate's
    fsdp2_prepare_model then upcasts every trainable parameter to FP32 on the still-unsharded
    model -- 133.1 GiB for Isaac-0.5, beyond any 80 GB device -- and fully_shard moves each unit
    onto its rank's GPU as it shards it.
    """
    from accelerate.utils.dataclasses import DistributedType

    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_expert_only = False
    config.use_peft = False
    config.train_storage_fp32 = True
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    assert policy._isaac_model is None, "the backbone must still be deferred after make_policy"
    policy.config.device = "cuda"
    devices = _record_backbone_load_device(monkeypatch)
    _install_distributed_state(
        monkeypatch,
        world_size=16,
        state=_StubAcceleratorState(
            distributed_type=DistributedType.FSDP,
            fsdp_plugin=_StubFsdpPlugin(fsdp_version=2, sharding_strategy=None, reshard_after_forward=True),
        ),
    )

    policy.get_optim_params()

    assert devices == ["cpu"]
    assert policy.config.device == "cuda", "the run device is unchanged for forward and checkpoints"


def test_unsharded_inference_builds_the_backbone_on_the_run_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path, config, _ = _native_portable_package(tmp_path)
    config.train_storage_fp32 = True
    policy = PerceptronIsaacPolicy.from_pretrained(policy_path, config=config, local_files_only=True)
    policy.config.device = "cuda"
    devices = _record_backbone_load_device(monkeypatch)

    policy._load_backbone(training=False)

    assert devices == ["cuda"]
