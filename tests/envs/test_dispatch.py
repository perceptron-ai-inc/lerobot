"""Tests for the benchmark dispatch refactor (create_envs / get_env_processors on EnvConfig)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import gymnasium as gym
import pytest
import torch
from gymnasium.envs.registration import register, registry as gym_registry

from lerobot.configs.types import PolicyFeature
from lerobot.envs.configs import EnvConfig, LiberoEnv
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.processor import LiberoProcessorStep
from lerobot.utils.constants import OBS_PREFIX, OBS_STATE

logger = logging.getLogger(__name__)


def test_registry_all_types():
    """make_env_config should resolve every registered EnvConfig subclass via the registry."""
    known = list(EnvConfig.get_known_choices().keys())
    assert len(known) >= 6
    for t in known:
        cfg = make_env_config(t)
        if not isinstance(cfg, EnvConfig):
            continue
        assert cfg.type == t


def test_unknown_type():
    with pytest.raises(ValueError, match="not registered"):
        make_env_config("nonexistent")


def test_libero_fps_controls_simulator_frequency():
    cfg = LiberoEnv(fps=17)

    assert cfg.gym_kwargs["control_freq"] == 17


def test_libero_rejects_nonpositive_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        LiberoEnv(fps=0)


def test_libero_adopts_strict_checkpoint_timing_geometry_and_camera_order():
    policy = type(
        "PolicyContract",
        (),
        {
            "target_fps": 30.0,
            "strict_environment_feature_contract": True,
            "action_dim": 7,
            "proprio_dim": 8,
            "image_size": (256, 256),
            "camera_order": ("image", "wrist_image"),
        },
    )()
    cfg = LiberoEnv()

    cfg.reconcile_policy_contract(policy)

    assert cfg.fps == 30
    assert (cfg.observation_height, cfg.observation_width) == (256, 256)
    assert cfg.camera_name_mapping == {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "wrist_image",
    }
    assert cfg.features_map["pixels/robot0_eye_in_hand_image"] == "observation.images.wrist_image"
    assert cfg.gym_kwargs["control_freq"] == 30


def test_libero_rejects_explicit_checkpoint_contract_mismatch():
    policy = type(
        "PolicyContract",
        (),
        {
            "target_fps": 30.0,
            "strict_environment_feature_contract": True,
            "action_dim": 7,
            "proprio_dim": 8,
            "image_size": (256, 256),
            "camera_order": ("image", "wrist_image"),
        },
    )()

    with pytest.raises(ValueError, match="Explicit env.fps"):
        LiberoEnv(fps=20).reconcile_policy_contract(policy, explicit_fields={"fps"})


def test_eval_override_names_include_nested_yaml_parent():
    from lerobot.configs.parser import get_override_field_names

    assert get_override_field_names(["--camera_name_mapping.agentview_image=image"]) == {
        "camera_name_mapping",
        "camera_name_mapping.agentview_image",
    }


def test_train_reconciles_strict_libero_checkpoint_contract(tmp_path, monkeypatch):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import (
        PerceptronIsaacConfig,
    )

    monkeypatch.setattr("sys.argv", ["pytest"])
    policy = PerceptronIsaacConfig(repo_id="synthetic/libero")
    policy.strict_environment_feature_contract = True
    policy.image_size = (256, 256)
    policy.target_fps = 20.0
    policy.camera_order = ("image", "wrist_image")
    policy.action_dim = 7
    policy.proprio_dim = 8
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="synthetic/train"),
        env=LiberoEnv(),
        policy=policy,
        output_dir=tmp_path / "train",
    )

    config.validate()

    assert config.env is not None
    assert (config.env.observation_height, config.env.observation_width) == (256, 256)
    assert config.env.fps == 20


def test_identity_processors():
    """Base class get_env_processors() returns identity pipelines."""
    cfg = make_env_config("aloha")
    pre, post = cfg.get_env_processors()
    assert len(pre.steps) == 0 and len(post.steps) == 0


def test_delegation():
    """make_env() should call cfg.create_envs(), not use if/elif dispatch."""
    sentinel = {"delegated": {0: "marker"}}
    fake = type(
        "Fake",
        (),
        {
            "hub_path": None,
            "create_envs": lambda self, n_envs, use_async_envs=False: sentinel,
        },
    )()
    result = make_env(fake, n_envs=1)
    assert result is sentinel


def test_processors_delegation():
    """make_env_pre_post_processors delegates to cfg.get_env_processors()."""
    cfg = make_env_config("aloha")
    pre, post = make_env_pre_post_processors(cfg, policy_cfg=None)
    assert len(pre.steps) == 0


def test_libero_processors_are_policy_agnostic():
    cfg = LiberoEnv()
    pre, post = make_env_pre_post_processors(cfg, policy_cfg=object())

    assert isinstance(pre.steps[0], LiberoProcessorStep)
    assert len(post.steps) == 0


def test_libero_processor_flattens_state_to_raw_8_dim():
    step = LiberoProcessorStep()
    observation = {
        OBS_PREFIX + "robot_state": {
            "eef": {
                "pos": torch.tensor([[1.0, 2.0, 3.0]]),
                "quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            },
            "gripper": {"qpos": torch.tensor([[4.0, 5.0]])},
        }
    }

    state = step.observation(observation)[OBS_STATE]
    assert state.shape == (1, 8)
    assert torch.allclose(state, torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0]]))


def test_base_create_envs():
    """Base class create_envs() should build a single-task VectorEnv via gym.make()."""
    gym_id = "_dispatch_test/CartPole-v99"
    if gym_id not in gym_registry:
        register(id=gym_id, entry_point="gymnasium.envs.classic_control:CartPoleEnv")

    @EnvConfig.register_subclass("_dispatch_base_test")
    @dataclass
    class _Env(EnvConfig):
        task: str = "CartPole-v99"
        fps: int = 10
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def package_name(self):
            return "_dispatch_test"

        @property
        def gym_id(self):
            return gym_id

        @property
        def gym_kwargs(self):
            return {}

    try:
        envs = _Env().create_envs(n_envs=2)
        assert "_dispatch_base_test" in envs
        env = envs["_dispatch_base_test"][0]
        assert isinstance(env, gym.vector.VectorEnv)
        assert env.num_envs == 2
        env.close()
    finally:
        if gym_id in gym_registry:
            del gym_registry[gym_id]


def test_custom_create_envs_override():
    """A custom EnvConfig subclass can override create_envs()."""
    mock_vec = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])

    @EnvConfig.register_subclass("_dispatch_custom_test")
    @dataclass
    class _Env(EnvConfig):
        task: str = "x"
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def gym_kwargs(self):
            return {}

        def create_envs(self, n_envs, use_async_envs=False):
            return {"custom_suite": {0: mock_vec}}

    try:
        result = make_env(_Env(), n_envs=1)
        assert "custom_suite" in result
    finally:
        mock_vec.close()


def test_custom_get_env_processors_override():
    """A custom EnvConfig subclass can override get_env_processors()."""
    from lerobot.processor.pipeline import DataProcessorPipeline

    @EnvConfig.register_subclass("_dispatch_proc_test")
    @dataclass
    class _Env(EnvConfig):
        task: str = "x"
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def gym_kwargs(self):
            return {}

        def get_env_processors(self):
            return DataProcessorPipeline(steps=[]), DataProcessorPipeline(steps=[])

    pre, post = _Env().get_env_processors()
    assert isinstance(pre, DataProcessorPipeline)
