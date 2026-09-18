"""Real reduced-Qwen sampling through native evaluator/checkpoint callers; pre-rendered inputs only."""

import json
from pathlib import Path

import gymnasium as gym
import pytest
import torch

# lerobot.scripts.lerobot_eval imports lerobot.datasets, which require_package("datasets").
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.perceptron_isaac.checkpoint_integrity import file_sha256
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_STREAM_KEY,
    make_perceptron_isaac_pre_post_processors,
)
from lerobot.scripts.lerobot_eval import eval_policy
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _write_stats
from tests.policies.perceptron_isaac.test_qwen35_training_lifecycle import reduced_qwen35_policy
from tests.test_eval_recording import TwoStepEnv


def reduced_qwen35_inference_package(root: Path) -> Path:
    seed = root / "seed"
    initial = reduced_qwen35_policy(seed)
    initial.get_optim_params()
    _write_stats(seed / "isaac_stats.json")
    initial.config.native_stats_path = "isaac_stats.json"
    initial.config.hf_model_path = "hf_model"
    initial.config.num_settle_steps = 0
    initial.config.num_inference_steps = 2
    initial.config.flow_seed_base = 19
    initial.config.clip_action_pose = False
    initial.config.gripper_binary_to_signed = False
    initial.config.save_pretrained(seed)
    adapter = seed / "isaac_deployment_adapter.json"
    adapter.write_text(json.dumps({"fixture": "synthetic reduced model; no production qualification"}))
    PerceptronIsaacPolicy.resolve_checkpoint_config_paths(initial.config, seed)
    package = root / "package"
    initial.save_pretrained(package, finalize=False)
    pre, post = make_perceptron_isaac_pre_post_processors(initial.config)
    pre.save_pretrained(package)
    post.save_pretrained(package)
    initial.config.deployment_adapter_sha256 = file_sha256(adapter)
    initial.finalize_pretrained_package(package)
    moved = root / "moved-package"
    package.rename(moved)
    return moved


def test_native_qwen35_eval_output_is_unchanged_by_recording(tmp_path):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    package = reduced_qwen35_inference_package(tmp_path)
    results = []
    for recording in (False, True):
        config = PreTrainedConfig.from_pretrained(package)
        policy = PerceptronIsaacPolicy.from_pretrained(package, config=config)
        _, postprocessor = make_pre_post_processors(config, pretrained_path=package)

        # Synthetic stream injection is the fixture boundary, not a mock model/normalizer/sampler.
        def prerendered(batch):
            return {**batch, PERCEPTRON_ISAAC_STREAM_KEY: flow_stream()}

        def identity(value):
            return value

        env = gym.vector.SyncVectorEnv([TwoStepEnv])
        try:
            result = eval_policy(
                env,
                policy,
                identity,
                identity,
                prerendered,
                postprocessor,
                n_episodes=1,
                return_episode_data=True,
                recording_dir=tmp_path / "recording" if recording else None,
                env_features={
                    "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                    "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
                },
            )
            results.append(result["episodes"]["action"])
        finally:
            env.close()
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    assert torch.isfinite(results[0]).all()
    dataset = LeRobotDataset("eval_recording", root=tmp_path / "recording", video_backend="pyav")
    actual = torch.stack([dataset[index]["action"] for index in range(2)])
    torch.testing.assert_close(actual, results[0][:2], rtol=0, atol=0)
