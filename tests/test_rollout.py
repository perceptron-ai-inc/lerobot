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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
import importlib
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    ["lerobot.rollout", "lerobot.rollout.inference", "lerobot.rollout.strategies"],
    ids=["top-level", "inference", "strategies"],
)
def test_public_exports(module_name):
    module = importlib.import_module(module_name)

    for name in module.__all__:
        assert hasattr(module, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"


def test_inference_config_types():
    from lerobot.rollout import (
        RTCInferenceConfig,
        SyncInferenceConfig,
        TemporalEnsembleInferenceConfig,
    )

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None

    temporal = TemporalEnsembleInferenceConfig()
    assert temporal.type == "temporal_ensemble"
    assert temporal.ensemble_m == 0.5
    assert temporal.max_step == 15.0


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


def test_rollout_config_passes_policy_pretrained_revision(monkeypatch):
    from lerobot.configs import PreTrainedConfig, parser
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    captured = {}

    def fake_from_pretrained(cls, pretrained_name_or_path, **kwargs):
        captured["pretrained_name_or_path"] = pretrained_name_or_path
        captured.update(kwargs)
        return SimpleNamespace(device="cpu", pretrained_revision=kwargs["revision"])

    monkeypatch.setattr(parser, "get_yaml_overrides", lambda _: ["--pretrained_revision=yaml-sha"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["lerobot-rollout", "--policy.path=user/policy", "--policy.pretrained_revision=cli-sha"],
    )
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(fake_from_pretrained))

    cfg = RolloutConfig(robot=MockRobotConfig())

    assert captured["pretrained_name_or_path"] == "user/policy"
    assert captured["revision"] == "cli-sha"
    assert captured["cli_overrides"] == [
        "--pretrained_revision=yaml-sha",
        "--pretrained_revision=cli-sha",
    ]
    assert cfg.policy.pretrained_path == "user/policy"
    assert cfg.policy.pretrained_revision == "cli-sha"


def test_rollout_config_adopts_and_enforces_checkpoint_fps(monkeypatch):
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    monkeypatch.setattr(sys, "argv", ["lerobot-rollout"])
    policy = SimpleNamespace(device="cpu", target_fps=20.0)

    cfg = RolloutConfig(robot=MockRobotConfig(), policy=policy)
    assert cfg.fps == 20.0

    with pytest.raises(ValueError, match="checkpoint target_fps"):
        RolloutConfig(robot=MockRobotConfig(), policy=policy, fps=30.0)


def test_rollout_config_prediction_only_is_base_strategy_only(monkeypatch):
    from lerobot.rollout import RolloutConfig, SentryStrategyConfig
    from tests.mocks.mock_robot import MockRobotConfig

    monkeypatch.setattr(sys, "argv", ["lerobot-rollout"])
    policy = SimpleNamespace(device="cpu", target_fps=None)
    dataset = SimpleNamespace(
        repo_id="user/rollout_test",
        fps=30.0,
        single_task="test",
        streaming_encoding=True,
    )

    with pytest.raises(ValueError, match="execute_actions=false"):
        RolloutConfig(
            robot=MockRobotConfig(),
            policy=policy,
            strategy=SentryStrategyConfig(),
            dataset=dataset,
            execute_actions=False,
        )


def test_load_pretrained_policy_passes_revision(monkeypatch):
    import lerobot.rollout.context as rollout_context

    policy_config = SimpleNamespace(
        type="mock",
        use_peft=False,
        pretrained_path="user/policy",
        pretrained_revision="policy-sha",
    )
    policy_class = MagicMock()
    loaded_policy = MagicMock()
    policy_class.from_pretrained.return_value = loaded_policy
    monkeypatch.setattr(rollout_context, "get_policy_class", lambda _: policy_class)

    policy = rollout_context._load_pretrained_policy(policy_config)

    assert policy is loaded_policy
    policy_class.from_pretrained.assert_called_once_with(
        "user/policy",
        config=policy_config,
        revision="policy-sha",
    )


def test_load_pretrained_peft_policy_delegates_to_shared_loader(monkeypatch):
    import lerobot.rollout.context as rollout_context

    policy_config = SimpleNamespace(
        type="mock",
        use_peft=True,
        pretrained_path="user/adapter",
        pretrained_revision="adapter-sha",
    )
    policy_class = MagicMock()
    monkeypatch.setattr(rollout_context, "get_policy_class", lambda _: policy_class)
    adapted_policy = MagicMock()
    load_peft = MagicMock(return_value=adapted_policy)
    monkeypatch.setattr(rollout_context, "load_peft_policy", load_peft)

    policy = rollout_context._load_pretrained_policy(policy_config)

    assert policy is adapted_policy
    load_peft.assert_called_once_with(
        policy_class,
        policy_config,
        "user/adapter",
        adapter_revision="adapter-sha",
    )


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


def test_sync_preflight_generates_and_validates_complete_fresh_chunk():
    from lerobot.rollout import SyncInferenceEngine

    policy = MagicMock()
    policy.config.use_amp = False
    policy.predict_action_chunk.return_value = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    preprocessor = MagicMock(side_effect=lambda value: value)
    postprocessor = MagicMock(side_effect=lambda value: value)
    engine = SyncInferenceEngine(
        policy,
        preprocessor,
        postprocessor,
        {"action": {"names": ["b.pos", "a.pos"]}},
        ["a.pos", "b.pos"],
        "test",
        "cpu",
        "so100_follower",
    )

    actions = engine.preflight({"observation.state": np.zeros(2, dtype=np.float32)})

    assert torch.equal(actions, torch.tensor([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]]))
    policy.predict_action_chunk.assert_called_once()
    policy.select_action.assert_not_called()
    policy.reset.assert_called_once()


def test_create_inference_engine_temporal_ensemble():
    from lerobot.rollout import (
        TemporalEnsembleInferenceConfig,
        TemporalEnsembleInferenceEngine,
        create_inference_engine,
    )

    policy = MagicMock()
    policy.reset.return_value = None
    engine = create_inference_engine(
        TemporalEnsembleInferenceConfig(),
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="so100_follower"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["joint.pos"],
        task="test",
        fps=30.0,
        device="cpu",
        preflight_inferences=2,
    )

    assert isinstance(engine, TemporalEnsembleInferenceEngine)
    assert engine.ready is False


def _perceptron_temporal_policy(*, n_obs_steps=1, n_action_steps=12, num_settle_steps=0):
    policy = MagicMock()
    policy.type = "perceptron_isaac"
    policy.config = SimpleNamespace(
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_settle_steps=num_settle_steps,
        use_amp=False,
    )
    return policy


def _create_temporal_engine_for_policy(policy, config=None):
    from lerobot.rollout import TemporalEnsembleInferenceConfig, create_inference_engine

    return create_inference_engine(
        config or TemporalEnsembleInferenceConfig(),
        policy=policy,
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="so100_follower"),
        hw_features={},
        dataset_features={"action": {"names": ["joint.pos"]}},
        ordered_action_keys=["joint.pos"],
        task="test",
        fps=30.0,
        device="cpu",
        preflight_inferences=0,
    )


def test_temporal_perceptron_rejects_history_it_cannot_sample_at_control_rate():
    with pytest.raises(NotImplementedError, match="n_obs_steps=3.*synchronous"):
        _create_temporal_engine_for_policy(_perceptron_temporal_policy(n_obs_steps=3))


def test_temporal_perceptron_enforces_select_action_chunk_contract():
    from lerobot.rollout import TemporalEnsembleInferenceConfig

    policy = _perceptron_temporal_policy(n_action_steps=12)
    engine = _create_temporal_engine_for_policy(policy)
    assert engine._actions_per_chunk == 12

    with pytest.raises(ValueError, match="actions_per_chunk=13.*n_action_steps=12"):
        _create_temporal_engine_for_policy(
            policy,
            TemporalEnsembleInferenceConfig(actions_per_chunk=13),
        )

    with pytest.raises(NotImplementedError, match="num_settle_steps"):
        _create_temporal_engine_for_policy(_perceptron_temporal_policy(num_settle_steps=1))


def test_temporal_ensemble_skips_latency_stale_prefix_and_combines_active_chunks():
    from lerobot.rollout.inference.temporal_ensemble import (
        TimestampedActionChunk,
        temporal_ensemble_action,
    )

    older = torch.arange(30, dtype=torch.float32).unsqueeze(1)
    newer = (100 + torch.arange(30, dtype=torch.float32)).unsqueeze(1)
    action, diagnostics = temporal_ensemble_action(
        [
            TimestampedActionChunk(older, observation_time=9.5, chunk_id=1),
            TimestampedActionChunk(newer, observation_time=9.8, chunk_id=2),
        ],
        now=10.0,
        action_fps=30.0,
        ensemble_m=0.0,
    )

    # 500 ms and 200 ms old means steps 15 and 5, never stale step zero.
    assert diagnostics == [(1, 15, 0.5), (2, 5, pytest.approx(0.2))]
    assert torch.allclose(action, torch.tensor([(15.0 + 105.0) / 2.0]))


def test_temporal_ensemble_rejects_expired_chunks():
    from lerobot.rollout.inference.temporal_ensemble import (
        TimestampedActionChunk,
        temporal_ensemble_action,
    )

    action, diagnostics = temporal_ensemble_action(
        [TimestampedActionChunk(torch.zeros(30, 2), observation_time=1.0, chunk_id=1)],
        now=2.01,
        action_fps=30.0,
        ensemble_m=0.5,
    )

    assert action is None
    assert diagnostics == []


def test_direction_preserving_action_cap_scales_complete_delta_vector():
    from lerobot.rollout.inference.temporal_ensemble import clip_action_direction_preserving

    current = torch.tensor([0.0, 10.0, -5.0])
    target = torch.tensor([30.0, 20.0, -20.0])
    clipped = clip_action_direction_preserving(target, current, max_step=15.0)

    assert torch.allclose(clipped, torch.tensor([15.0, 15.0, -12.5]))


def test_temporal_engine_discards_complete_warmup_chunks_before_publish(monkeypatch):
    from lerobot.rollout.inference.temporal_ensemble import TemporalEnsembleInferenceEngine

    policy = MagicMock()
    policy.reset.return_value = None
    engine = TemporalEnsembleInferenceEngine(
        policy,
        MagicMock(),
        MagicMock(),
        robot_type="so100_follower",
        hw_features={},
        dataset_features={"action": {"names": ["joint.pos"]}},
        ordered_action_keys=["joint.pos"],
        task="test",
        fps=30.0,
        device="cpu",
        max_step=None,
        warmup_predictions=2,
    )
    calls = 0

    def predict(_obs, *, timestep):
        nonlocal calls
        calls += 1
        if calls == 3:
            engine.pause()
        return torch.full((30, 1), float(calls))

    monkeypatch.setattr(engine, "_predict", predict)
    engine.reset()
    engine.start()
    engine.notify_observation({"joint.pos": 0.0})
    engine.resume()
    deadline = time.monotonic() + 1.0
    while calls < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    engine.stop()

    assert calls == 3
    assert engine.ready is True
    assert len(engine._ring) == 1
    assert engine._ring[0].chunk_id == 1
    assert torch.all(engine._ring[0].actions == 3.0)
    # reset() before start plus a second reset after the final discarded warmup
    # keeps prediction-only state out of the first live chunk.
    assert policy.reset.call_count == 2


def test_temporal_engine_predicts_each_control_observation_once_with_its_timestep(monkeypatch):
    from lerobot.rollout.inference.temporal_ensemble import TemporalEnsembleInferenceEngine

    policy = MagicMock()
    policy.config.use_amp = False
    engine = TemporalEnsembleInferenceEngine(
        policy,
        MagicMock(),
        MagicMock(),
        robot_type="so100_follower",
        hw_features={},
        dataset_features={"action": {"names": ["joint.pos"]}},
        ordered_action_keys=["joint.pos"],
        task="test",
        fps=30.0,
        device="cpu",
        max_step=None,
        warmup_predictions=0,
    )
    predicted_timesteps = []

    def predict(_obs, *, timestep):
        predicted_timesteps.append(timestep)
        return torch.zeros(30, 1)

    monkeypatch.setattr(engine, "_predict", predict)
    engine.reset()
    engine.notify_observation({"joint.pos": 0.0})
    engine._producer_iteration(0)
    _, sleep_s = engine._producer_iteration(0)
    assert sleep_s is not None
    assert predicted_timesteps == [0]

    engine.notify_observation({"joint.pos": 1.0})
    engine._producer_iteration(0)
    assert predicted_timesteps == [0, 1]


def test_temporal_engine_passes_control_timing_to_policy_hook():
    from lerobot.rollout.inference.temporal_ensemble import TemporalEnsembleInferenceEngine

    policy = MagicMock()
    policy.config.use_amp = False
    policy.predict_action_chunk.return_value = torch.zeros(1, 4, 2)
    policy.adapt_async_observation.side_effect = lambda observation, **_: observation
    preprocessor = MagicMock(side_effect=lambda value: value)
    postprocessor = MagicMock(side_effect=lambda value: value)
    engine = TemporalEnsembleInferenceEngine(
        policy,
        preprocessor,
        postprocessor,
        robot_type="so100_follower",
        hw_features={
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["joint.pos"],
            }
        },
        dataset_features={"action": {"names": ["joint.pos", "gripper.pos"]}},
        ordered_action_keys=["joint.pos", "gripper.pos"],
        task="test",
        fps=30.0,
        device="cpu",
        max_step=None,
        warmup_predictions=0,
    )

    engine._predict({"joint.pos": 1.0}, timestep=7)

    policy.adapt_async_observation.assert_called_once()
    assert policy.adapt_async_observation.call_args.kwargs == {"timestep": 7, "fps": 30.0}


def _rollout_context_harness(monkeypatch, robot):
    import lerobot.rollout.context as rollout_context
    from lerobot.rollout import BaseStrategyConfig, SyncInferenceConfig
    from tests.mocks.mock_robot import MockRobotConfig

    policy = MagicMock()
    policy.type = "mock"
    policy.to.return_value = policy
    policy_config = SimpleNamespace(
        type="mock",
        pretrained_path="/checkpoint",
        pretrained_revision=None,
        input_features={},
        action_feature_names=None,
    )
    monkeypatch.setattr(rollout_context, "_load_pretrained_policy", lambda _: policy)
    monkeypatch.setattr(rollout_context, "make_robot_from_config", lambda _: robot)
    cfg = SimpleNamespace(
        inference=SyncInferenceConfig(),
        policy=policy_config,
        use_torch_compile=False,
        torch_compile_backend="inductor",
        torch_compile_mode="default",
        device="cpu",
        robot=MockRobotConfig(n_motors=2),
        teleop=None,
        rename_map={},
        dataset=None,
        strategy=BaseStrategyConfig(),
        resume=False,
        task="test",
        fps=30.0,
        compile_warmup_inferences=0,
    )
    return rollout_context, cfg


def test_build_rollout_context_validates_processors_before_connect(monkeypatch):
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    events = []
    robot = MockRobot(MockRobotConfig(n_motors=2))
    original_connect = robot.connect

    def tracked_connect():
        events.append("connect")
        original_connect()

    robot.connect = tracked_connect
    rollout_context, cfg = _rollout_context_harness(monkeypatch, robot)
    monkeypatch.setattr(
        rollout_context,
        "validate_visual_features_consistency",
        lambda *_: events.append("visuals"),
    )

    def tracked_processors(**kwargs):
        events.append("processors")
        return preprocessor, postprocessor

    def tracked_engine(*args, **kwargs):
        events.append("engine")
        return MagicMock()

    preprocessor = MagicMock(steps=[])
    postprocessor = MagicMock(steps=[])
    monkeypatch.setattr(rollout_context, "make_pre_post_processors", tracked_processors)
    monkeypatch.setattr(rollout_context, "create_inference_engine", tracked_engine)

    ctx = rollout_context.build_rollout_context(cfg, MagicMock())

    assert events == ["visuals", "processors", "engine", "connect"]
    assert ctx.hardware.robot_wrapper.inner is robot
    robot.disconnect()


def test_build_rollout_context_disconnects_after_early_connect_failure(monkeypatch):
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=2))
    # Seed an unresolved camera resolution so build_rollout_context connects early; the
    # cached_property is popped after connect, so the real (camera-less) features return.
    robot.__dict__["observation_features"] = {**robot._motors_ft, "cam": (None, None, 3)}
    rollout_context, cfg = _rollout_context_harness(monkeypatch, robot)
    monkeypatch.setattr(
        rollout_context,
        "make_pre_post_processors",
        lambda **kwargs: (MagicMock(steps=[]), MagicMock(steps=[])),
    )

    def failing_engine(*args, **kwargs):
        raise RuntimeError("engine construction failed")

    monkeypatch.setattr(rollout_context, "create_inference_engine", failing_engine)

    with pytest.raises(RuntimeError, match="engine construction failed"):
        rollout_context.build_rollout_context(cfg, MagicMock())

    assert robot.is_connected is False


def test_strict_action_contract_fails_instead_of_falling_back():
    from lerobot.rollout.context import _resolve_action_key_order

    with pytest.raises(ValueError, match="Strict policy action names"):
        _resolve_action_key_order(["left.pos", "right.pos"], ["a.pos", "b.pos"], strict=True)


def test_strict_hardware_contract_validates_order_and_camera_geometry():
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.rollout.context import _validate_strict_hardware_feature_contract

    policy = SimpleNamespace(
        strict_hardware_feature_contract=True,
        state_feature_names=["left.pos", "right.pos"],
        action_feature_names=["left.pos", "right.pos"],
        input_features={
            "observation.images.top": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 256, 256),
            )
        },
    )
    observations = {"left.pos": float, "right.pos": float, "cam0": (256, 256, 3)}
    actions = {"left.pos": float, "right.pos": float}

    _validate_strict_hardware_feature_contract(
        policy,
        observations,
        actions,
        {"observation.images.cam0": "observation.images.top"},
    )

    with pytest.raises(ValueError, match="expects CHW"):
        _validate_strict_hardware_feature_contract(
            policy,
            {**observations, "cam0": (480, 640, 3)},
            actions,
            {"observation.images.cam0": "observation.images.top"},
        )


def test_prediction_only_dispatch_runs_postprocessor_without_sending(monkeypatch):
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    engine = MagicMock()
    engine.get_action.return_value = torch.tensor([1.0, 2.0])
    robot = MagicMock()
    action_processor = MagicMock(side_effect=lambda value: value[0])
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=SimpleNamespace(execute_actions=False)),
        policy=SimpleNamespace(inference=engine),
        data=SimpleNamespace(dataset_features={}, ordered_action_keys=["a.pos", "b.pos"]),
        processors=SimpleNamespace(robot_action_processor=action_processor),
        hardware=SimpleNamespace(robot_wrapper=robot),
    )
    monkeypatch.setattr(
        "lerobot.rollout.strategies.core.build_dataset_frame",
        lambda *args, **kwargs: {"observation.state": torch.zeros(1)},
    )

    action = send_next_action({}, {}, ctx, ActionInterpolator(multiplier=1))

    assert action == {"a.pos": 1.0, "b.pos": 2.0}
    action_processor.assert_called_once()
    robot.send_action.assert_not_called()


@pytest.mark.parametrize(
    ("return_to_initial_position", "execute_actions", "expected"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_rollout_teardown_context_requires_return_and_execution(
    return_to_initial_position, execute_actions, expected
):
    from lerobot.rollout import BaseStrategy, BaseStrategyConfig

    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._teardown_hardware = MagicMock()
    hardware = MagicMock()
    ctx = SimpleNamespace(
        hardware=hardware,
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                return_to_initial_position=return_to_initial_position,
                execute_actions=execute_actions,
            )
        ),
    )

    strategy._teardown_context(ctx)

    strategy._teardown_hardware.assert_called_once_with(
        hardware,
        return_to_initial_position=expected,
    )


def test_sync_prediction_preflight_never_sends_and_resets(monkeypatch):
    from lerobot.rollout import BaseStrategy, BaseStrategyConfig, SyncInferenceConfig

    engine = MagicMock()
    engine.preflight.return_value = torch.tensor([[0.25], [0.5]])
    robot = MagicMock()
    robot.get_observation.return_value = {"joint.pos": 0.0}
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                interpolation_multiplier=1,
                preflight_inferences=2,
                inference=SyncInferenceConfig(),
            )
        ),
        policy=SimpleNamespace(inference=engine),
        hardware=SimpleNamespace(robot_wrapper=robot),
        processors=SimpleNamespace(robot_observation_processor=lambda obs: obs),
        data=SimpleNamespace(dataset_features={}, ordered_action_keys=["joint.pos"]),
    )
    monkeypatch.setattr(
        "lerobot.rollout.strategies.core.build_dataset_frame",
        lambda *args, **kwargs: {"observation.state": torch.zeros(1)},
    )

    BaseStrategy(BaseStrategyConfig()).setup(ctx)

    assert engine.preflight.call_count == 2
    assert engine.reset.call_count == 2
    robot.send_action.assert_not_called()


def test_cuda_device_process_lock_rejects_competing_owner(tmp_path):
    from lerobot.utils.device_process_lock import _physical_device_key, exclusive_torch_device_lock

    with exclusive_torch_device_lock("cuda:3", lock_dir=tmp_path) as path:
        assert path == tmp_path / f"lerobot-cuda-v2-{_physical_device_key(3)}.lock"
        with (
            pytest.raises(RuntimeError, match="already owned"),
            exclusive_torch_device_lock("cuda:3", lock_dir=tmp_path),
        ):
            pass

    with exclusive_torch_device_lock("cuda:3", lock_dir=tmp_path):
        pass


def test_cuda_device_lock_key_is_stable_across_visible_devices_spellings(monkeypatch):
    from lerobot.utils import device_process_lock

    gpu_uuid = "12345678-1234-1234-1234-123456789abc"
    monkeypatch.setattr(
        device_process_lock.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(uuid=gpu_uuid),
    )
    keys = set()
    for spelling in (None, "1", f"GPU-{gpu_uuid}"):
        if spelling is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", spelling)
        keys.add(device_process_lock._physical_device_key(0))
    assert keys == {f"GPU-{gpu_uuid}"}


def test_cuda_device_lock_file_is_world_writable_despite_umask(tmp_path):
    import os

    from lerobot.utils.device_process_lock import exclusive_torch_device_lock

    prior_umask = os.umask(0o077)
    try:
        with exclusive_torch_device_lock("cuda:3", lock_dir=tmp_path) as path:
            assert (path.stat().st_mode & 0o777) == 0o666
    finally:
        os.umask(prior_umask)


def test_cuda_device_lock_ignores_unopenable_legacy_lock_file(tmp_path):
    from lerobot.utils.device_process_lock import exclusive_torch_device_lock

    legacy = tmp_path / "lerobot-cuda-3.lock"
    legacy.touch(mode=0o000)
    with exclusive_torch_device_lock("cuda:3", lock_dir=tmp_path) as path:
        assert path is not None
        assert path != legacy


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}


def test_temporal_ensemble_underflow_falls_back_to_freshest_chunk():
    """exp(-m*age) underflows float32 to exactly 0 for m*age > ~104; the ensemble must
    fall back to the freshest live chunk instead of dividing by zero into NaN."""
    from lerobot.rollout.inference.temporal_ensemble import (
        TimestampedActionChunk,
        temporal_ensemble_action,
    )

    older = torch.full((60, 2), 1.0)
    newer = torch.full((60, 2), 2.0)
    # ages 1.60 s and 1.55 s: with ensemble_m=70, m*age is 112 and 108.5 -- both past
    # the ~103.9 float32 underflow point, so every live weight is exactly zero.
    action, diagnostics = temporal_ensemble_action(
        [
            TimestampedActionChunk(older, observation_time=8.40, chunk_id=1),
            TimestampedActionChunk(newer, observation_time=8.45, chunk_id=2),
        ],
        now=10.0,
        action_fps=30.0,
        ensemble_m=70.0,
    )

    assert torch.isfinite(action).all()
    assert torch.allclose(action, torch.full((2,), 2.0))  # freshest chunk wins
    assert [d[0] for d in diagnostics] == [2]


def test_temporal_ensemble_normal_weights_stay_finite():
    from lerobot.rollout.inference.temporal_ensemble import (
        TimestampedActionChunk,
        temporal_ensemble_action,
    )

    action, diagnostics = temporal_ensemble_action(
        [
            TimestampedActionChunk(torch.full((60, 2), 1.0), observation_time=9.5, chunk_id=1),
            TimestampedActionChunk(torch.full((60, 2), 2.0), observation_time=9.8, chunk_id=2),
        ],
        now=10.0,
        action_fps=30.0,
        ensemble_m=0.5,
    )
    assert torch.isfinite(action).all()
    assert len(diagnostics) == 2


def test_build_rollout_context_disconnects_when_resolution_stays_unknown(monkeypatch):
    """A camera that reports (None, None, 3) both before and after connect must not
    leave the arm torqued: the still-unresolved raise happens after the early connect
    but before the assembly cleanup handler."""
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=2))
    monkeypatch.setattr(
        MockRobot,
        "observation_features",
        property(lambda self: {**self._motors_ft, "cam": (None, None, 3)}),
    )
    rollout_context, cfg = _rollout_context_harness(monkeypatch, robot)
    monkeypatch.setattr(
        rollout_context,
        "make_pre_post_processors",
        lambda **kwargs: (MagicMock(steps=[]), MagicMock(steps=[])),
    )
    monkeypatch.setattr(rollout_context, "create_inference_engine", lambda *a, **k: MagicMock())

    with pytest.raises(ValueError, match="still unknown after connect"):
        rollout_context.build_rollout_context(cfg, MagicMock())

    assert robot.is_connected is False
