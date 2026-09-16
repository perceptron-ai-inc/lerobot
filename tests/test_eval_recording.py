"""Synthetic LIBERO-shaped observations, real Gym vector calls and native dataset writers."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.scripts.lerobot_eval import _build_raw_frame, _env_features_to_dataset_features, eval_policy


def libero_observation():
    return {
        "pixels": {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "image2": np.full((256, 256, 3), 17, dtype=np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.array([1.0, 2.0, 3.0]),
                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "mat": np.eye(3),
            },
            "gripper": {"qpos": np.array([0.1, 0.2]), "qvel": np.array([0.3, 0.4])},
            "joints": {"pos": np.arange(7, dtype=np.float64), "vel": np.arange(7, dtype=np.float64)},
        },
    }


def batch_arrays(observation):
    return {
        key: batch_arrays(value) if isinstance(value, dict) else value[None]
        for key, value in observation.items()
    }


def test_libero_recording_schema_is_slash_free():
    config = LiberoConfig(observation_height=256, observation_width=256)
    features = _env_features_to_dataset_features(config.features)
    assert all("/" not in name for name in features)


def test_libero_recording_frame_preserves_all_declared_nested_state_and_cameras():
    config = LiberoConfig(observation_height=256, observation_width=256)
    frame = _build_raw_frame(
        batch_arrays(libero_observation()),
        0,
        np.zeros(7),
        1.0,
        True,
        True,
        "synthetic",
        config.features,
        config.features_map,
    )
    assert len(frame) == len(config.features) + 4
    assert all("/" not in name for name in frame)


class TwoStepEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    task = "synthetic"
    task_description = "synthetic episode"
    _max_episode_steps = 2

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(-100.0, 100.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return {"agent_pos": np.array([0.0, 1.0], dtype=np.float32)}, {}

    def step(self, action):
        self.steps += 1
        return (
            {"agent_pos": np.array([self.steps, 1.0], dtype=np.float32)},
            1.0,
            self.steps == 2,
            False,
            {"is_success": self.steps == 2},
        )


class ConstantPolicy(PreTrainedPolicy):
    config_class = ACTConfig
    name = "recording_fixture"

    def __init__(self):
        super().__init__(ACTConfig(device="cpu", push_to_hub=False))
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.resets = 0

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        return self.bias.unsqueeze(0) + 0.0

    def predict_action_chunk(self, batch):
        return self.select_action(batch).unsqueeze(1)

    def forward(self, batch):
        return self.bias.square().mean(), {}

    def get_optim_params(self):
        return self.parameters()


def test_eval_recording_keeps_one_writer_across_two_episodes(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy = ConstantPolicy()
    env = gym.vector.SyncVectorEnv([TwoStepEnv])
    features = {
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }

    def identity(value):
        return value

    try:
        result = eval_policy(
            env,
            policy,
            identity,
            identity,
            identity,
            identity,
            n_episodes=2,
            recording_dir=tmp_path / "recording",
            env_features=features,
        )
    finally:
        env.close()
    dataset = LeRobotDataset("eval_recording", root=tmp_path / "recording", video_backend="pyav")
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 4
    assert dataset.fps == 20
    assert policy.resets == 2
    torch.testing.assert_close(dataset[1]["timestamp"], torch.tensor(0.05))
    assert len(result["per_episode"]) == 2


def test_libero_instance_metadata_uses_its_control_frequency_without_simulator():
    from lerobot.envs.libero import LiberoEnv

    task = SimpleNamespace(
        name="synthetic", language="synthetic", problem_folder="unused", bddl_file="unused.bddl"
    )
    suite = SimpleNamespace(get_task=lambda task_id: task)
    slow = LiberoEnv(suite, 0, "libero_spatial", init_states=False, control_freq=20)
    fast = LiberoEnv(suite, 0, "libero_spatial", init_states=False, control_freq=30)
    assert slow._env is None and fast._env is None
    assert slow.metadata["render_fps"] == 20
    assert fast.metadata["render_fps"] == 30
    assert slow.metadata is not fast.metadata


class LiberoShapedEnv(TwoStepEnv):
    """No simulator: preserve LIBERO's actual nested raw-observation contract."""

    def __init__(self):
        super().__init__()

        def space_for(value):
            if isinstance(value, dict):
                return gym.spaces.Dict({key: space_for(item) for key, item in value.items()})
            return (
                gym.spaces.Box(0, 255, shape=value.shape, dtype=np.uint8)
                if value.dtype == np.uint8
                else gym.spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=value.dtype)
            )

        self.observation_space = space_for(libero_observation())
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return libero_observation(), {}

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return libero_observation(), reward, terminated, truncated, info


def test_libero_shaped_eval_records_two_native_video_episodes(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.lerobot_eval import eval_policy_all

    config = LiberoConfig(observation_height=256, observation_width=256)
    policy = ConstantPolicy()
    policy.bias = torch.nn.Parameter(torch.zeros(7))

    def identity(value):
        return value

    env = gym.vector.SyncVectorEnv([LiberoShapedEnv])
    result = eval_policy_all(
        {"synthetic": {0: env}},
        policy,
        identity,
        identity,
        identity,
        identity,
        n_episodes=2,
        recording_dir=tmp_path,
        env_features=config.features,
        env_features_map=config.features_map,
    )
    dataset = LeRobotDataset("eval_recording", root=tmp_path / "synthetic_0", video_backend="pyav")
    assert dataset.num_episodes == 2 and dataset.num_frames == 4
    assert dataset.fps == 20
    assert policy.resets == 2
    assert result["overall"]["n_episodes"] == 2
    assert all("/" not in key for key in dataset.features)
    assert set(config.features_map.values()) <= dataset.features.keys()
    for camera in ("observation.images.image", "observation.images.image2"):
        video_info = dataset.meta.features[camera]["info"]
        assert video_info["video.fps"] == 20
        assert (video_info["video.height"], video_info["video.width"], video_info["video.channels"]) == (
            256,
            256,
            3,
        )
    first = dataset[0]
    torch.testing.assert_close(first["observation.state.eef_pos"], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(first["observation.state.eef_mat"], torch.eye(3))
    torch.testing.assert_close(first["action"], torch.zeros(7))
    assert first["observation.images.image"].shape == (3, 256, 256)
    assert first["observation.images.image2"].shape == (3, 256, 256)
    torch.testing.assert_close(
        first["observation.images.image2"], torch.full((3, 256, 256), 17 / 255), rtol=0, atol=2 / 255
    )
    torch.testing.assert_close(dataset[1]["timestamp"], torch.tensor(0.05))
    torch.testing.assert_close(dataset[2]["timestamp"], torch.tensor(0.0))
    assert bool(dataset[1]["next.done"]) and bool(dataset[3]["next.done"])


def test_recording_frame_honors_custom_camera_aliases():
    config = LiberoConfig(
        observation_height=256,
        observation_width=256,
        camera_name_mapping={"agentview_image": "front", "robot0_eye_in_hand_image": "wrist"},
    )
    raw = libero_observation()
    raw["pixels"] = {"front": raw["pixels"]["image"], "wrist": raw["pixels"]["image2"]}
    frame = _build_raw_frame(
        batch_arrays(raw),
        0,
        np.zeros(7),
        1.0,
        False,
        False,
        "synthetic",
        config.features,
        config.features_map,
    )
    assert frame["observation.images.wrist"].shape == (256, 256, 3)
    assert np.all(frame["observation.images.wrist"] == 17)
    with pytest.raises(ValueError, match="collides"):
        _env_features_to_dataset_features(
            config.features, {"pixels/agentview_image": "same", "pixels/robot0_eye_in_hand_image": "same"}
        )


@pytest.mark.parametrize("failure", ["create", "body", "finalize"])
def test_recording_closes_all_owned_writers_and_never_publishes_failed_eval(tmp_path, monkeypatch, failure):
    from lerobot.scripts import lerobot_eval

    closed = []
    published = []
    created = []

    class Writer:
        num_episodes = 1

        def __init__(self, index):
            self.index = index

        def finalize(self):
            closed.append(self.index)
            if failure == "finalize" and self.index == 1:
                raise ValueError("synthetic failure")

        def push_to_hub(self, **kwargs):
            published.append(self.index)

    def create(**kwargs):
        index = len(created)
        if failure == "create" and index == 1:
            raise ValueError("synthetic failure")
        writer = Writer(index)
        created.append(writer)
        return writer

    monkeypatch.setattr(lerobot_eval.LeRobotDataset, "create", create)
    env = SimpleNamespace(num_envs=2, unwrapped=SimpleNamespace(metadata={"render_fps": 20}))
    with (
        pytest.raises(ValueError, match="synthetic failure"),
        lerobot_eval._eval_recording_datasets(env, tmp_path, {}, None, "synthetic/no-network", False),
    ):
        if failure == "body":
            raise ValueError("synthetic failure")
    assert set(closed) == {writer.index for writer in created}
    assert published == []


def test_recording_frame_uses_the_declared_action_alias():
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    mapping = {"action": "executed_action"}
    schema = _env_features_to_dataset_features(features, mapping)
    frame = _build_raw_frame({}, 0, np.zeros(2), 0.0, False, False, "synthetic", features, mapping)
    assert set(frame) - {"task"} == set(schema)
