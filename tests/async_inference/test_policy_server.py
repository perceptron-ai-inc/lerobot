# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import skip_if_package_missing

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@skip_if_package_missing("grpcio", "grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@skip_if_package_missing("grpcio", "grpc")
def test_load_saved_policy_delegates_peft_loading(monkeypatch):
    from lerobot.async_inference import policy_server as ps_module

    class _PolicyConfig:
        use_peft = True

    policy_class = object()
    load_peft = MagicMock(return_value="peft-policy")
    monkeypatch.setattr(ps_module, "load_peft_policy", load_peft)

    config = _PolicyConfig()
    result = ps_module._load_saved_policy(policy_class, config, "adapter")

    assert result == "peft-policy"
    load_peft.assert_called_once_with(policy_class, config, "adapter")


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


@skip_if_package_missing("grpcio", "grpc")
def test_async_fps_contract_binds_client_server_and_checkpoint():
    from lerobot.async_inference.policy_server import _validate_async_fps_contract

    config = SimpleNamespace(target_fps=20.0)
    _validate_async_fps_contract(config, client_fps=20.0, server_fps=20.0)

    with pytest.raises(ValueError, match="client fps=30.*server fps=20"):
        _validate_async_fps_contract(config, client_fps=30.0, server_fps=20.0)
    with pytest.raises(ValueError, match="server fps=30.*checkpoint target_fps=20"):
        _validate_async_fps_contract(config, client_fps=30.0, server_fps=30.0)
    for client_fps, server_fps, target_fps in (
        (float("nan"), 20.0, 20.0),
        (20.0, float("inf"), 20.0),
        (20.0, 20.0, float("nan")),
        (20.0, 20.0, float("inf")),
    ):
        with pytest.raises(ValueError, match="fps"):
            _validate_async_fps_contract(
                SimpleNamespace(target_fps=target_fps),
                client_fps=client_fps,
                server_fps=server_fps,
            )


@skip_if_package_missing("grpcio", "grpc")
def test_strict_async_contract_validates_ordered_joint_names_and_mapped_cameras():
    from lerobot.async_inference.policy_server import _validate_strict_async_feature_contract

    config = SimpleNamespace(
        strict_hardware_feature_contract=True,
        state_feature_names=["shoulder.pos", "elbow.pos"],
        action_feature_names=["shoulder.pos", "elbow.pos"],
        camera_order=("top", "side"),
    )
    features = {
        "observation.state": {
            "shape": (2,),
            "names": ["shoulder.pos", "elbow.pos"],
        },
        "action": {
            "shape": (2,),
            "names": ["shoulder.pos", "elbow.pos"],
        },
        "observation.images.camera_a": {"shape": (256, 256, 3)},
        "observation.images.camera_b": {"shape": (256, 256, 3)},
    }
    rename_map = {
        "observation.images.camera_a": "observation.images.top",
        "observation.images.camera_b": "observation.images.side",
    }
    _validate_strict_async_feature_contract(config, features, rename_map)

    bad_state = {**features, "observation.state": {**features["observation.state"]}}
    bad_state["observation.state"]["names"] = ["elbow.pos", "shoulder.pos"]
    with pytest.raises(ValueError, match="state order"):
        _validate_strict_async_feature_contract(config, bad_state, rename_map)

    bad_action = {**features, "action": {**features["action"]}}
    bad_action["action"]["names"] = ["elbow.pos", "shoulder.pos"]
    with pytest.raises(ValueError, match="action order"):
        _validate_strict_async_feature_contract(config, bad_action, rename_map)

    with pytest.raises(ValueError, match="camera mapping"):
        _validate_strict_async_feature_contract(config, features, {})


@skip_if_package_missing("grpcio", "grpc")
def test_perceptron_async_chunk_contract_rejects_unsupported_stateful_semantics():
    from lerobot.async_inference.policy_server import _validate_async_chunk_contract

    config = SimpleNamespace(
        type="perceptron_isaac",
        n_obs_steps=1,
        n_action_steps=12,
        num_settle_steps=0,
    )
    _validate_async_chunk_contract(config, actions_per_chunk=12)

    with pytest.raises(ValueError, match="actions_per_chunk=13.*n_action_steps=12"):
        _validate_async_chunk_contract(config, actions_per_chunk=13)
    with pytest.raises(NotImplementedError, match="n_obs_steps=3.*synchronous"):
        _validate_async_chunk_contract(
            SimpleNamespace(**{**vars(config), "n_obs_steps": 3}),
            actions_per_chunk=12,
        )
    with pytest.raises(NotImplementedError, match="num_settle_steps"):
        _validate_async_chunk_contract(
            SimpleNamespace(**{**vars(config), "num_settle_steps": 2}),
            actions_per_chunk=12,
        )


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """The whole chunk is postprocessed once with per-timestep value parity."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.processor import PolicyProcessorPipeline, UnnormalizerProcessorStep
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import ACTION

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk
    raw_actions = torch.arange(batch_size * actions_per_chunk * action_dim, dtype=torch.float32).reshape(
        batch_size, actions_per_chunk, action_dim
    )
    scale = torch.linspace(0.5, 1.0, action_dim)
    offset = torch.arange(action_dim, dtype=torch.float32)
    postprocessor_shapes = []

    def transform(tensor):
        return tensor * scale + offset

    pipeline = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats={ACTION: {"mean": offset, "std": scale}},
            )
        ],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    def postprocessor(tensor):
        postprocessor_shapes.append(tuple(tensor.shape))
        return pipeline(tensor)

    policy_server.postprocessor = postprocessor

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return raw_actions

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert postprocessor_shapes == [(batch_size, actions_per_chunk, action_dim)]
    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))
    expected = torch.stack(
        [transform(raw_actions[:, i, :]) for i in range(actions_per_chunk)], dim=1
    ).squeeze(0)
    torch.testing.assert_close(torch.stack([ta.get_action() for ta in timed_actions]), expected)

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6
