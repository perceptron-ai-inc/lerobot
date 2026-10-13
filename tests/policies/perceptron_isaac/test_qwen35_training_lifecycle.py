"""Synthetic, reduced real Qwen3.5 VLA; no pretrained weights or robot data."""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file

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
