"""Reduced native MK1 training fixtures; no pretrained weights or robot data."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    Mk1Qwen36VLAConfig,
    Mk1Qwen36VLAForActionGeneration,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config as _mk1_config
from tests.policies.perceptron_isaac.test_modeling_training import _config


def reduced_native_model(
    *, vocab_size: int = 32, action_dim: int = 2, proprio_dim: int = 2
) -> Mk1Qwen36VLAForActionGeneration:
    """Build real native attention, vision, vector encoder and expert with tiny geometry."""
    raw = _mk1_config(test_geometry=True, mtp_present=False, coord_enabled=False)
    raw["text_config"]["vocab_size"] = vocab_size
    raw["vision_config"]["patch_size"] = 16
    expert = dict(raw["genesis_vla"]["action_expert"], action_dim=action_dim, action_horizon=2)
    config = Mk1Qwen36VLAConfig(vector_max_states=proprio_dim, action_expert=expert, **raw)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    return Mk1Qwen36VLAForActionGeneration(config)


def reduced_native_policy(
    root: Path, *, vocab_size: int = 32, action_dim: int = 2, proprio_dim: int = 2
) -> PerceptronIsaacPolicy:
    torch.manual_seed(17)
    model = reduced_native_model(vocab_size=vocab_size, action_dim=action_dim, proprio_dim=proprio_dim)
    root.mkdir(parents=True, exist_ok=True)
    config = _config()
    config.vector_max_states = proprio_dim
    config.action_dim = action_dim
    config.proprio_dim = proprio_dim
    config.input_features["observation.state"] = replace(
        config.input_features["observation.state"], shape=(proprio_dim,)
    )
    config.output_features["action"] = replace(config.output_features["action"], shape=(action_dim,))
    config.apply_offset_norm = False
    config.push_to_hub = False
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = model
    return policy


def test_dense_optimizer_preserves_serializable_trainable_set(tmp_path: Path) -> None:
    policy = reduced_native_policy(tmp_path)
    policy.config.freeze_input_embeddings = True
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/native-mk1"),
        policy=policy.config,
        optimizer=AdamWConfig(lr=0.002),
    )
    optimizer, _ = make_optimizer_and_scheduler(cfg, policy)
    trainable = {name: p for name, p in policy.named_parameters() if p.requires_grad}
    frozen = {name: p.detach().clone() for name, p in policy.named_parameters() if not p.requires_grad}
    before = {name: p.detach().clone() for name, p in trainable.items()}
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in trainable.values()
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = policy({PERCEPTRON_ISAAC_STREAM_KEY: flow_stream()})
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in trainable.values())
    optimizer.step()
    assert any(not torch.equal(parameter, before[name]) for name, parameter in trainable.items())
    for name, parameter in policy.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    save_file(policy.state_dict(), tmp_path / "trained.safetensors")
    saved = load_file(tmp_path / "trained.safetensors")
    for name, parameter in trainable.items():
        torch.testing.assert_close(saved[name], parameter, rtol=0, atol=0)


def test_peft_wrap_refuses_before_loading_backbone(tmp_path: Path) -> None:
    policy = reduced_native_policy(tmp_path)
    with pytest.raises(RuntimeError, match="PEFT training is unsupported"):
        policy.wrap_with_peft()
