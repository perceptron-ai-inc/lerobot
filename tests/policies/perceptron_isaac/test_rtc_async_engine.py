"""LeRobot's asynchronous RTC engine drives ISAAC's trained-RTC prefix sampler.

These tests run the real ``RTCInferenceEngine`` worker, chunk runtime, ISAAC processors and policy
rendering. Only the model leaf is replaced by a gated sampler that records the prefix it is given,
and the engine clock is frozen as the existing engine tests do, so the sequence is deterministic.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Semaphore
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.inference import RTCInferenceConfig, RTCInferenceEngine
from lerobot.inference.rtc import supports_rtc_inference, validate_local_chunk_policy
from lerobot.policies.chunk import ExecutionMode
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mharmony_native import IsaacMharmonyContentPlan
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    make_perceptron_isaac_pre_post_processors,
)
from lerobot.policies.perceptron_isaac.tensor_stream import TensorStream
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.context import _validate_trained_rtc_rollout_config
from tests.policies.perceptron_isaac.test_batched_online_rollout import (
    _live_batch,
    _ProprioChunkSampler,
    _rollout_harness,
    sdk_plans as _sdk_plans,
)
from tests.policies.perceptron_isaac.test_observation_timestamp_recipe import _package_config
from tests.utils import wait_for

sdk_plans = _sdk_plans

_TASK = "Task: Move the block !"
_DATASET_FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (16, 16, 3), "names": ["h", "w", "c"]},
    "observation.images.wrist_image": {"dtype": "video", "shape": (16, 16, 3), "names": ["h", "w", "c"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
}


class _GatedPrefixSampler(_ProprioChunkSampler):
    """Model leaf: records each requested prefix, waits for the test, then pins the prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.release = Semaphore(0)
        self.prefixes: list[tuple[Tensor | None, int | None]] = []
        self.outputs: list[Tensor] = []

    def sample_action(
        self,
        stream: TensorStream,
        *,
        num_steps: int,
        action_dim: int,
        num_action_steps: int,
        num_flow_samples: int,
        action_prefix: Tensor | None = None,
        prefix_length: int | None = None,
    ) -> Tensor:
        self.prefixes.append((None if action_prefix is None else action_prefix.clone(), prefix_length))
        assert self.release.acquire(timeout=10), "the test must release each model call"
        chunk = super().sample_action(
            stream,
            num_steps=num_steps,
            action_dim=action_dim,
            num_action_steps=num_action_steps,
            num_flow_samples=num_flow_samples,
        )
        if action_prefix is not None:
            assert prefix_length is not None
            chunk[:, :prefix_length] = action_prefix[:, :prefix_length]
        self.outputs.append(chunk.clone())
        return chunk


def _rtc_config(tmp_path: Path, *, budget: int = 2, recipe_emit: bool = False) -> PerceptronIsaacConfig:
    """A timestamp-free package (chunk_size=4) whose action-expert recipe declares ``budget``."""
    config = _package_config(tmp_path, recipe_emit=recipe_emit, n_obs_steps=1)
    package = Path(str(config.pretrained_path))
    (package / "hf_model").mkdir(exist_ok=True)
    (package / "hf_model" / "config.json").write_text(
        json.dumps(
            {
                "action_expert": {
                    "rtc_max_delay_steps": budget,
                    "rtc_probability": 0.5 if budget else None,
                }
            }
        )
    )
    config.hf_model_path = str(package / "hf_model")
    return config


@contextmanager
def _policy(config: PerceptronIsaacConfig) -> Iterator[tuple[PerceptronIsaacPolicy, _GatedPrefixSampler]]:
    with _rollout_harness(config) as harness:
        sampler = _GatedPrefixSampler()
        harness.policy._isaac_model = sampler
        yield harness.policy, sampler


def _engine(
    policy: PerceptronIsaacPolicy,
    config: PerceptronIsaacConfig,
    *,
    mode: str = "trained",
    rtc_enabled: bool = True,
) -> RTCInferenceEngine:
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(config)
    return RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=SimpleNamespace(robot_type="mock", action_features={}, supports_hold=True),
        rtc_config=RTCConfig(enabled=rtc_enabled, mode=mode, execution_horizon=2),
        dataset_features=_DATASET_FEATURES,
        task=_TASK,
        fps=30,
        device="cpu",
        # Request the successor once one of the four predicted actions has executed.
        rtc_queue_threshold=3,
    )


def _observe(engine: RTCInferenceEngine, tick: int) -> None:
    batch = _live_batch(tick, batch_size=1)
    state = batch["observation.state"][0]
    engine.notify_observation(
        {
            "j1.pos": float(state[0]),
            "j2.pos": float(state[1]),
            "image": batch["observation.images.image"][0].permute(1, 2, 0).numpy().copy(),
            "wrist_image": batch["observation.images.wrist_image"][0].permute(1, 2, 0).numpy().copy(),
        }
    )


def _run_one_prediction(engine: RTCInferenceEngine, sampler: _GatedPrefixSampler, tick: int) -> None:
    """Publish one observation and let exactly one model call start and finish."""
    started = len(sampler.prefixes)
    _observe(engine, tick)
    assert wait_for(lambda: len(sampler.prefixes) == started + 1 or engine.failed, timeout=10)
    assert not engine.failed, engine.failure_traceback
    # The worker has copied this observation; withdraw it so no further request starts.
    with engine._obs_lock:
        engine._obs_holder["obs"] = None
    sampler.release.release()
    assert wait_for(lambda: len(sampler.outputs) == started + 1 or engine.failed, timeout=10)
    assert wait_for(lambda: engine._runtime.pending is None or engine.failed, timeout=10)
    assert not engine.failed, engine.failure_traceback


def test_startup_admits_trained_rtc_only_for_a_declared_budget(tmp_path: Path) -> None:
    config = _rtc_config(tmp_path, budget=2)
    inference = RTCInferenceConfig(rtc=RTCConfig(mode="trained", execution_horizon=2), queue_threshold=2)
    # lerobot-rollout checks the budget on the policy config before any weights load.
    assert config.rtc_training_max_delay == 2
    _validate_trained_rtc_rollout_config(config, inference)
    with _policy(config) as (policy, _):
        assert supports_rtc_inference(policy)
        spec = validate_local_chunk_policy(policy)
        assert spec.modes == (ExecutionMode.CHUNK, ExecutionMode.RTC_TRAINED)
        assert (spec.prediction_steps, spec.execution_steps, spec.training_max_delay) == (4, 2, 2)
        assert _engine(policy, config)._runtime.mode is ExecutionMode.RTC_TRAINED
        # Guided RTC is never advertised, so the engine refuses it at startup.
        with pytest.raises(ValueError, match="rtc_guided"):
            _engine(policy, config, mode="guided")


def test_startup_refuses_trained_rtc_without_a_budget(tmp_path: Path) -> None:
    config = _rtc_config(tmp_path, budget=0)
    inference = RTCInferenceConfig(rtc=RTCConfig(mode="trained", execution_horizon=2), queue_threshold=2)
    with pytest.raises(ValueError, match="rtc_training_max_delay > 0"):
        _validate_trained_rtc_rollout_config(config, inference)
    with _policy(config) as (policy, _):
        assert not supports_rtc_inference(policy)
        assert validate_local_chunk_policy(policy).modes == (ExecutionMode.CHUNK,)
        with pytest.raises(ValueError, match="rtc_trained"):
            _engine(policy, config)
        # A direct trained-prefix request is rejected before the model is called.
        with pytest.raises(ValueError, match="prefix budget 0"):
            policy.predict_action_chunk(
                _live_batch(0, batch_size=1), inference_delay=1, prev_chunk_left_over=torch.zeros(2, 2)
            )


def test_plain_async_inference_gives_a_timestamp_recipe_its_episode_time(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    """Like the published LIBERO package: timestamps on, one observation step, no RTC budget."""
    config = _rtc_config(tmp_path, budget=0, recipe_emit=True)
    with _policy(config) as (policy, sampler):
        spec = validate_local_chunk_policy(policy)
        assert spec.modes == (ExecutionMode.CHUNK,)
        assert spec.needs_episode_time
        engine = _engine(policy, config, rtc_enabled=False)
        assert engine._runtime.mode is ExecutionMode.CHUNK
        clock = [time.monotonic()]
        engine._runtime.clock = lambda: clock[0]

        def predict_after(seconds: float, tick: int) -> None:
            clock[0] += seconds
            engine._robot.observation_time = clock[0]
            _run_one_prediction(engine, sampler, tick=tick)

        engine.start()
        try:
            engine.resume()
            predict_after(0.0, tick=0)
            predict_after(0.5, tick=1)
            # A reset starts a new episode, whose first observation is time zero again.
            engine.reset()
            predict_after(0.25, tick=2)
        finally:
            sampler.release.release(20)
            engine.stop()
        assert not engine.failed, engine.failure_traceback
        assert [prefix for prefix, _ in sampler.prefixes] == [None, None, None]
    # Each prompt carries the observation's episode-relative capture time, not a frame count.
    assert [plan.anchor_timestamp_seconds for plan in sdk_plans] == [0.0, 0.5, 0.0]
    for plan in sdk_plans:
        items = [item for item in plan.user_content if item["kind"] == "timestamp"]
        assert len(items) == 1
        assert items[0]["metadata"]["timestamp_seconds"] == plan.anchor_timestamp_seconds


@pytest.mark.parametrize("recipe_emit", [False, True])
def test_chunk_spec_loads_deferred_weights_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe_emit: bool
) -> None:
    """FP32-storage packages defer loading; chunk callers need it done before their deadlines."""
    policy = PerceptronIsaacPolicy(_rtc_config(tmp_path, budget=2, recipe_emit=recipe_emit))
    assert policy._isaac_model is None
    loads: list[int] = []

    def load_backbone() -> None:
        loads.append(1)
        policy._isaac_model = _GatedPrefixSampler()

    monkeypatch.setattr(policy, "_load_backbone", load_backbone)
    for _ in range(2):
        spec = policy.chunk_inference_spec()
        assert spec.training_max_delay == 2
        # Only a recipe that renders episode-time tokens asks its caller for episode time.
        assert spec.needs_episode_time is recipe_emit
    assert len(loads) == 1


def test_engine_continuation_becomes_the_prefix_and_reset_clears_it(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    config = _rtc_config(tmp_path, budget=2)
    with _policy(config) as (policy, sampler):
        engine = _engine(policy, config)
        now = time.monotonic()
        engine._runtime.clock = lambda: now
        engine._robot.observation_time = now
        engine.start()
        try:
            engine.resume()

            # 1. Startup: nothing is executing, so the first prediction has no prefix.
            _run_one_prediction(engine, sampler, tick=0)
            assert sampler.prefixes[0] == (None, None)
            assert engine.action_queue.qsize() == config.chunk_size
            first = sampler.outputs[0][0]

            # 2. The robot executes one action; the remaining rows are the continuation.
            assert engine.get_action(None) is not None
            _run_one_prediction(engine, sampler, tick=1)
            prefix, length = sampler.prefixes[1]
            assert length == 2, "the frozen clock conditions on the full two-step budget"
            assert prefix is not None
            assert prefix.shape == (1, 2, config.action_dim)
            # Rows 1 and 2 of the first chunk execute while this prediction runs.
            torch.testing.assert_close(prefix[0], first[1:3], rtol=0, atol=0)
            torch.testing.assert_close(sampler.outputs[1][0, :2], first[1:3], rtol=0, atol=0)
            assert policy._frame_index == 1

            # 3. Reset drops the continuation and the policy state with it.
            engine.reset()
            _run_one_prediction(engine, sampler, tick=2)
            assert sampler.prefixes[2] == (None, None)
            assert policy._frame_index == 0
            assert engine.action_queue.qsize() == config.chunk_size
        finally:
            sampler.release.release(20)
            engine.stop()
        assert not engine.failed, engine.failure_traceback
    # Every asynchronous prediction rendered a prompt without episode-time items.
    assert len(sdk_plans) == 3
    assert all(item["kind"] != "timestamp" for plan in sdk_plans for item in plan.user_content)


def test_predict_action_chunk_maps_delay_and_continuation_to_the_sampler(tmp_path: Path) -> None:
    config = _rtc_config(tmp_path, budget=2)
    continuation = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
    with _policy(config) as (policy, sampler):
        sampler.release.release(10)
        batch = _live_batch(0, batch_size=1)
        # A zero delay is the ordinary prefix-free call, with or without a continuation.
        policy.predict_action_chunk(batch, inference_delay=0, prev_chunk_left_over=continuation)
        assert sampler.prefixes[-1] == (None, None)
        chunk = policy.predict_action_chunk(batch, inference_delay=2, prev_chunk_left_over=continuation)
        prefix, length = sampler.prefixes[-1]
        assert length == 2
        assert prefix is not None
        torch.testing.assert_close(prefix, continuation[:2].unsqueeze(0), rtol=0, atol=0)
        torch.testing.assert_close(chunk[0, :2], continuation[:2], rtol=0, atol=0)
        with pytest.raises(ValueError, match="requires prev_chunk_left_over"):
            policy.predict_action_chunk(batch, inference_delay=1)
        with pytest.raises(ValueError, match="nonnegative"):
            policy.predict_action_chunk(batch, inference_delay=-1, prev_chunk_left_over=continuation)
        with pytest.raises(ValueError, match="normalized actions"):
            policy.predict_action_chunk(batch, inference_delay=3, prev_chunk_left_over=continuation[:2])
        with pytest.raises(ValueError, match="normalized actions"):
            policy.predict_action_chunk(
                batch, inference_delay=1, prev_chunk_left_over=continuation.unsqueeze(0)
            )
        with pytest.raises(ValueError, match="either an explicit action_prefix"):
            policy.predict_action_chunk(
                batch,
                action_prefix=continuation[:1].unsqueeze(0),
                prefix_length=1,
                inference_delay=1,
                prev_chunk_left_over=continuation,
            )
        with pytest.raises(ValueError, match="sync path"):
            policy.select_action(batch, inference_delay=1, prev_chunk_left_over=continuation)
