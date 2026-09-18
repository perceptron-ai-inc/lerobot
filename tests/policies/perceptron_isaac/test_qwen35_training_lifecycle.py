"""Synthetic, reduced real Qwen3.5 VLA; no pretrained weights or robot data."""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import load_file, save_file

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

pytest.importorskip("transformers")

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
    Qwen35VLAConfig,
    Qwen35VLAForActionGeneration,
)
from tests.policies.perceptron_isaac.test_modeling_training import _config


def reduced_qwen35_model(
    *, vocab_size: int = 32, action_dim: int = 2, proprio_dim: int = 2
) -> Qwen35VLAForActionGeneration:
    """Build real attention, vision, vector encoder and action expert with tiny geometry."""
    return Qwen35VLAForActionGeneration(
        Qwen35VLAConfig(
            text_config={
                "vocab_size": vocab_size,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "layer_types": ["full_attention"],
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [1, 1, 2],
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 32,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 16,
            },
            vector_max_states=proprio_dim,
            action_expert={
                "type": "molmoact",
                "action_dim": action_dim,
                "action_horizon": 2,
                "num_layers": 1,
                "hidden_dim": 16,
                "num_heads": 2,
                "mlp_ratio": 1.0,
                "num_inference_steps": 2,
                "timestep_embed_dim": 8,
                "ffn_multiple_of": 8,
            },
        )
    )


def reduced_qwen35_policy(
    root: Path, *, vocab_size: int = 32, action_dim: int = 2, proprio_dim: int = 2
) -> PerceptronIsaacPolicy:
    torch.manual_seed(17)
    model = reduced_qwen35_model(vocab_size=vocab_size, action_dim=action_dim, proprio_dim=proprio_dim)
    model.config.save_pretrained(root / "hf_model")
    save_file(model.state_dict(), root / "hf_model" / "model.safetensors")
    config = _config()
    config.hf_model_path = str(root / "hf_model")
    config.vector_max_states = proprio_dim
    config.action_dim = action_dim
    config.proprio_dim = proprio_dim
    config.input_features["observation.state"] = replace(
        config.input_features["observation.state"], shape=(proprio_dim,)
    )
    config.output_features["action"] = replace(config.output_features["action"], shape=(action_dim,))
    config.apply_offset_norm = False
    config.push_to_hub = False
    config.pretrained_path = root
    return PerceptronIsaacPolicy(config)


def test_custom_optimizer_materializes_lazy_qwen35(tmp_path: Path) -> None:
    policy = reduced_qwen35_policy(tmp_path)
    assert list(policy.parameters()) == []
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/qwen35"),
        policy=policy.config,
        use_policy_training_preset=False,
        optimizer=AdamWConfig(lr=0.002),
    )
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    assert scheduler is None
    assert policy._training_parameters_configured
    assert optimizer.param_groups[0]["lr"] == 0.002
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in policy.parameters()
    }


@pytest.mark.parametrize("train_expert_only", [False, True])
def test_peft_optimizer_preserves_serializable_trainable_set(tmp_path: Path, train_expert_only: bool) -> None:
    pytest.importorskip("peft")
    from peft import get_peft_model_state_dict

    policy = reduced_qwen35_policy(tmp_path)
    policy.config.train_expert_only = train_expert_only
    peft_policy = policy.wrap_with_peft(peft_cli_overrides={"r": 2})
    frozen = {name: p.detach().clone() for name, p in peft_policy.named_parameters() if not p.requires_grad}
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/qwen35"),
        policy=policy.config,
        optimizer=AdamWConfig(lr=0.002),
    )
    optimizer, _ = make_optimizer_and_scheduler(cfg, peft_policy)
    trainable = {name: p for name, p in peft_policy.named_parameters() if p.requires_grad}
    assert all("lora_" in name or "modules_to_save.default" in name for name in trainable), sorted(trainable)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in trainable.values()
    }
    # Every optimizer-owned value must be retained by real PEFT serialization.
    sum(p.square().sum() for p in trainable.values()).backward()
    optimizer.step()
    print(f"OPTIMIZER_UPDATE serialization train_expert_only={train_expert_only}")
    for name, parameter in peft_policy.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    adapter = get_peft_model_state_dict(peft_policy)
    for name, parameter in trainable.items():
        key = name.replace(".modules_to_save.default", "").replace(".default", "")
        torch.testing.assert_close(adapter[key], parameter)


def synthetic_qwen35_dataset(
    root: Path, *, action_dim: int = 2, proprio_dim: int = 2, camera_order: tuple[str, ...] = ("image",)
) -> "LeRobotDataset":
    """Write/read two synthetic image frames through native LeRobot dataset serialization."""
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    action_names = ["a", "b"] if action_dim == 2 else [f"action_{i}" for i in range(action_dim)]
    state_names = ["a", "b"] if proprio_dim == 2 else [f"state_{i}" for i in range(proprio_dim)]
    dataset = LeRobotDataset.create(
        "synthetic/qwen35",
        root=root,
        fps=20,
        use_videos=False,
        features={
            "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
            "observation.state": {"dtype": "float32", "shape": (proprio_dim,), "names": state_names},
            **{
                f"observation.images.{camera}": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                }
                for camera in camera_order
            },
        },
    )
    for frame_index in range(2):
        action = np.arange(1 + frame_index * action_dim, 1 + (frame_index + 1) * action_dim, dtype=np.float32)
        dataset.add_frame(
            {
                "action": np.array(action, dtype=np.float32),
                "observation.state": np.zeros(proprio_dim, dtype=np.float32),
                **{
                    f"observation.images.{camera}": np.zeros((256, 256, 3), dtype=np.uint8)
                    for camera in camera_order
                },
                "task": "synthetic contract only",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset("synthetic/qwen35", root=root, video_backend="pyav")


@pytest.mark.parametrize("use_peft", [False, True])
@torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False)
def test_native_qwen35_update_save_reload_resume(tmp_path: Path, monkeypatch, use_peft: bool) -> None:
    """Real native loss/update and disk checkpoint callers; synthetic stream, not mharmony rendering."""
    import json
    import random
    import sys

    import numpy as np
    from accelerate import Accelerator

    from lerobot.common.train_utils import load_training_state, save_checkpoint
    from lerobot.configs import PreTrainedConfig
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    from lerobot.policies.factory import make_policy
    from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
    from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
        ISAAC_FLOW_NOISE_KEY,
        ISAAC_FLOW_TIMESTEPS_KEY,
    )
    from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
    from lerobot.utils.random_utils import serialize_rng_state
    from tests.policies.perceptron_isaac._fixtures import flow_stream

    dataset = synthetic_qwen35_dataset(tmp_path / "dataset")
    torch.testing.assert_close(dataset[0]["action"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(dataset[1]["action"], torch.tensor([3.0, 4.0]))
    source = tmp_path / "base"
    initial = reduced_qwen35_policy(tmp_path / "seed")
    initial.get_optim_params()
    # Explicitly inert metadata, not an imported/qualified deployment contract.
    adapter = tmp_path / "seed" / "isaac_deployment_adapter.json"
    adapter.write_text(json.dumps({"fixture": "synthetic reduced Qwen; not deployable"}))
    initial.config.hf_model_path = "hf_model"
    initial.config.save_pretrained(tmp_path / "seed")
    PerceptronIsaacPolicy.resolve_checkpoint_config_paths(initial.config, tmp_path / "seed")
    initial.save_pretrained(source, finalize=False)
    initial.config.deployment_adapter_sha256 = file_sha256(adapter)
    initial.finalize_pretrained_package(source)
    config = PreTrainedConfig.from_pretrained(source)
    config.pretrained_path = source
    policy = make_policy(config, ds_meta=dataset.meta)
    if use_peft:
        policy = policy.wrap_with_peft(peft_cli_overrides={"r": 2})
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/qwen35", root=str(tmp_path / "dataset")),
        policy=policy.config,
        optimizer=AdamWConfig(lr=0.002),
        scheduler=CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=0, num_decay_steps=4, peak_lr=0.002, decay_lr=0.0001
        ),
        steps=4,
        output_dir=tmp_path / "run",
    )
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    accelerator = Accelerator(cpu=True)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    tracker = MetricsTracker(
        1, 2, 1, {key: AverageMeter(key) for key in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb")}
    )
    batch = {
        PERCEPTRON_ISAAC_STREAM_KEY: flow_stream(),
        ISAAC_FLOW_TIMESTEPS_KEY: torch.tensor([[0.5]]),
        ISAAC_FLOW_NOISE_KEY: torch.zeros(1, 1, 2, 2),
    }
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    update_policy(tracker, policy, batch, optimizer, 10.0, accelerator, scheduler)
    print(f"OPTIMIZER_UPDATE native use_peft={use_peft}")
    assert torch.isfinite(torch.tensor(tracker.loss.val))
    assert tracker.grad_norm.val > 0
    assert any(not torch.equal(p, before[name]) for name, p in policy.named_parameters())
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, 1, cfg, accelerator.unwrap_model(policy), optimizer, scheduler)
    expected = {name: p.detach().clone() for name, p in policy.named_parameters()}
    moved = tmp_path / "moved"
    checkpoint.rename(moved)
    source.rename(tmp_path / "retired-base")
    assert not source.exists()
    pretrained = moved / "pretrained_model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-train",
            f"--config_path={pretrained / 'train_config.json'}",
            "--resume=true",
            "--policy.device=cpu",
        ],
    )
    resumed_cfg = TrainPipelineConfig.from_pretrained(pretrained, cli_args=["--resume=true"])
    resumed_cfg.validate()
    resumed = make_policy(resumed_cfg.policy, ds_meta=dataset.meta)
    resumed_optimizer, resumed_scheduler = make_optimizer_and_scheduler(resumed_cfg, resumed)
    saved_rng = load_file(moved / "training_state/rng_state.safetensors")
    assert saved_rng
    random.random()
    np.random.random()
    torch.rand(1)
    if "torch_cuda_rng_state" in saved_rng:
        torch.rand(1, device="cuda")
    if "torch_mps_rng_state" in saved_rng:
        torch.rand(1, device="mps")
    perturbed_rng = serialize_rng_state()
    assert perturbed_rng.keys() == saved_rng.keys()
    for prefix in ("py_", "np_", "torch_rng", "torch_cuda", "torch_mps"):
        keys = [key for key in saved_rng if key.startswith(prefix)]
        if keys:
            assert any(not torch.equal(perturbed_rng[key], saved_rng[key]) for key in keys), prefix
    step, resumed_optimizer, resumed_scheduler = load_training_state(
        moved, resumed_optimizer, resumed_scheduler
    )
    # Inspect the native supported representation before any next RNG consumer.
    restored_rng = serialize_rng_state()
    assert restored_rng.keys() == saved_rng.keys()
    for name, value in saved_rng.items():
        torch.testing.assert_close(restored_rng[name], value, rtol=0, atol=0, msg=name)
    assert step == 1
    assert resumed_scheduler is not None
    assert scheduler is not None
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    restored_parameters = dict(resumed.named_parameters())
    assert expected and restored_parameters.keys() == expected.keys()
    for name, parameter in restored_parameters.items():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0, msg=name)
    original_state = optimizer.state_dict()
    restored_state = resumed_optimizer.state_dict()
    assert restored_state["param_groups"] == original_state["param_groups"]
    assert original_state["state"] and restored_state["state"].keys() == original_state["state"].keys()
    for index, state in original_state["state"].items():
        assert state and restored_state["state"][index].keys() == state.keys()
        for key, value in state.items():
            torch.testing.assert_close(restored_state["state"][index][key], value, rtol=0, atol=0)
    policy.eval()
    resumed.eval()
    with torch.no_grad(), accelerator.autocast():
        torch.testing.assert_close(resumed.forward(batch)[0], policy.forward(batch)[0], rtol=0, atol=0)
    update_policy(tracker, policy, batch, optimizer, 10.0, accelerator, scheduler)
    print(f"OPTIMIZER_UPDATE native use_peft={use_peft}")
    resumed, resumed_optimizer = accelerator.prepare(resumed, resumed_optimizer)
    update_policy(tracker, resumed, batch, resumed_optimizer, 10.0, accelerator, resumed_scheduler)
    print(f"OPTIMIZER_UPDATE native use_peft={use_peft}")
    for (name, parameter), (restored_name, restored) in zip(
        policy.named_parameters(), resumed.named_parameters(), strict=True
    ):
        assert name == restored_name
        torch.testing.assert_close(restored, parameter, rtol=0, atol=0)
