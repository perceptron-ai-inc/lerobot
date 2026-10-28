"""P2 rollout regressions: real windowing/packing/queues, isolated SDK and model leaves.

The SDK leaf serializes an already-built real content plan into local TensorStream
records. It does not test Qwen vocabulary IDs, image processing, or model quality.
The sampler leaf derives different chunks from each stream's own proprio, with
independent horizon/feature values. Neither leaf implements batching or history.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import Tensor, nn

from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.isaac_stats import (
    load_isaac_stats,
    normalize_isaac_actions,
    normalize_isaac_proprio,
)
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyContentPlan,
    IsaacMharmonyRenderMetadata,
    IsaacNativeMharmonyRenderer,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_RENDER_META_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacMharmonyPackProcessorStep,
    make_perceptron_isaac_pre_post_processors,
)
from lerobot.policies.perceptron_isaac.tensor_stream import (
    TensorStream,
    TextType,
    VectorType,
)
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.perceptron_isaac._rollout_fixtures import (
    _assert_plan_history,
    _rollout_config,
    _write_rollout_stats,
    sdk_plans as _sdk_plans,
)

sdk_plans = _sdk_plans


def _live_batch(tick: int, batch_size: int = 2) -> dict[str, Any]:
    """Different environments, cameras, times, state channels, and prompt lengths."""
    return {
        "observation.images.image": torch.stack(
            [torch.full((3, 16, 16), 20 + 60 * env + tick, dtype=torch.uint8) for env in range(batch_size)]
        ),
        "observation.images.wrist_image": torch.stack(
            [torch.full((3, 16, 16), 40 + 60 * env + tick, dtype=torch.uint8) for env in range(batch_size)]
        ),
        OBS_STATE: torch.tensor(
            [[0.1 + env * 0.25 + tick * 0.01, -0.2 + env * 0.1] for env in range(batch_size)]
        ),
        "task": ["Task: Move " + "the block " * (env + 1) + "!" for env in range(batch_size)],
    }


def _state_event(stream: TensorStream, index: int) -> Tensor:
    vectors = [event.data for event in stream.streams[index].events if event.type == VectorType.vector]
    assert len(vectors) == 1, "The ordinary prompt contains last-state-only proprio."
    return vectors[0].reshape(-1)


class _ProprioChunkSampler(nn.Module):
    """Model-only boundary: each sample depends on its own actual packed state."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.inputs: list[TensorStream] = []

    def sample_action(
        self,
        stream: TensorStream,
        *,
        num_steps: int,
        action_dim: int,
        num_action_steps: int,
        action_prefix: Tensor | None,
        prefix_length: int | None,
        num_flow_samples: int,
    ) -> Tensor:
        assert num_steps == num_flow_samples == 1
        assert action_prefix is None and prefix_length is None
        self.inputs.append(stream)
        states = torch.stack([_state_event(stream, index)[:action_dim] for index in range(stream.shape[0])])
        offsets = torch.arange(num_action_steps, dtype=states.dtype) * 0.02
        return states[:, None, :] + offsets[None, :, None]


@dataclass(frozen=True)
class _RolloutHarness:
    policy: PerceptronIsaacPolicy
    sampler: _ProprioChunkSampler


@contextmanager
def _rollout_harness(config: PerceptronIsaacConfig) -> Iterator[_RolloutHarness]:
    policy = PerceptronIsaacPolicy(config)
    sampler = _ProprioChunkSampler()
    # The checkpoint loader/model is the only policy boundary replaced. No I/O or model download.
    with patch.object(policy, "_isaac_model", sampler):
        yield _RolloutHarness(policy, sampler)


def _expected_chunk(
    config: PerceptronIsaacConfig,
    batch: dict[str, Any],
    *,
    packed_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    assert config.native_stats_path is not None
    stats = load_isaac_stats(config.native_stats_path)
    state = np.stack([normalize_isaac_proprio(row.numpy(), stats.proprio) for row in batch[OBS_STATE]])
    # The saved pack contract is BF16; the narrow sampler has an F32 parameter.
    packed_state = torch.from_numpy(state).to(packed_dtype).float()
    return packed_state[:, None, :] + torch.arange(config.chunk_size)[None, :, None] * 0.02


def test_native_renderer_leaf_already_collates_distinct_samples(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    """Control: leaf collation works; the missing behavior is above this boundary."""
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    renderer = IsaacNativeMharmonyRenderer(
        metadata=IsaacMharmonyRenderMetadata.from_config(config), stats=load_isaac_stats(stats_path)
    )
    batch = _live_batch(0)
    streams = []
    for env in range(2):
        streams.append(
            renderer.build(
                observation_window=[
                    {
                        "images": {
                            "image": batch["observation.images.image"][env].numpy(),
                            "wrist_image": batch["observation.images.wrist_image"][env].numpy(),
                        },
                        "proprio": batch[OBS_STATE][env].numpy(),
                    }
                ],
                prompt="move " + "the block " * env + "the block",
                anchor_timestamp_seconds=0.1,
                device="cpu",
                dtype=torch.float32,
            )
        )
    packed = renderer.collate(streams)
    assert packed.shape[0] == 2
    assert not torch.equal(_state_event(packed, 0), _state_event(packed, 1))
    assert any(event.type == TextType.padding for event in packed.streams[0].events)
    for env, plan in enumerate(sdk_plans):
        _assert_plan_history(plan, env, [0, 0, 0])


def test_online_preprocessor_defers_batched_single_steps_to_policy_history(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    preprocessor, _ = make_perceptron_isaac_pre_post_processors(_rollout_config(stats_path))
    batch = _live_batch(0)
    processed = preprocessor(batch)
    assert PERCEPTRON_ISAAC_STREAM_KEY not in processed
    assert processed[PERCEPTRON_ISAAC_RENDER_META_KEY]["render_skipped"] is True
    assert processed["task"] == ["move the block", "move the block the block"]
    torch.testing.assert_close(processed[OBS_STATE], batch[OBS_STATE])


def test_explicit_batched_windows_render_distinct_prompts_states_and_images(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    step = PerceptronIsaacMharmonyPackProcessorStep.from_config(
        config, native_stats_path=str(stats_path), device="cpu"
    )
    batches = [_live_batch(tick) for tick in range(3)]
    batch = dict(batches[-1])
    for key in (OBS_STATE, "observation.images.image", "observation.images.wrist_image"):
        batch[key] = torch.stack([item[key] for item in batches], dim=1)
    batch["timestamp"] = torch.tensor([0.2, 0.35])
    output = transition_to_batch(step(batch_to_transition(batch)))
    packed = output[PERCEPTRON_ISAAC_STREAM_KEY]
    assert packed.shape[0] == 2
    assert output[PERCEPTRON_ISAAC_RENDER_META_KEY]["anchor_timestamp_seconds"] == pytest.approx([0.2, 0.35])
    for env, plan in enumerate(sdk_plans):
        _assert_plan_history(plan, env, [0, 1, 2])
        torch.testing.assert_close(
            _state_event(packed, env).float(), _expected_chunk(config, batches[-1])[env, 0]
        )
    assert len(sdk_plans) == 2


@pytest.mark.parametrize("batch_size", [1, 2], ids=["single_control", "batched"])
@pytest.mark.parametrize("n_obs_steps", [1, 3], ids=["nobs1", "nobs3"])
def test_online_select_action_keeps_per_environment_history_and_consumes_chunk_axis(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan], batch_size: int, n_obs_steps: int
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path, n_obs_steps=n_obs_steps)
    with _rollout_harness(config) as harness:
        for tick in range(5):
            batch = _live_batch(tick, batch_size)
            action = harness.policy.select_action(batch)
            anchor_tick = tick - tick % 2
            expected = _expected_chunk(config, _live_batch(anchor_tick, batch_size))[:, tick % 2]
            assert action.shape == (batch_size, 2)
            torch.testing.assert_close(action, expected)
            assert len(harness.sampler.inputs) == tick // 2 + 1
        assert len(sdk_plans) == batch_size * 3
        for refill, tick in enumerate((0, 2, 4)):
            ticks = [max(0, tick - delta) for delta in reversed(range(n_obs_steps))]
            for env in range(batch_size):
                plan = sdk_plans[refill * batch_size + env]
                _assert_plan_history(plan, env, ticks)
                assert plan.anchor_timestamp_seconds == pytest.approx(max(tick, n_obs_steps - 1) / 20.0)
        # One action remains queued. Global reset must discard it and all old observations.
        harness.policy.reset()
        action = harness.policy.select_action(_live_batch(9, batch_size))
        torch.testing.assert_close(action, _expected_chunk(config, _live_batch(9, batch_size))[:, 0])
        for env in range(batch_size):
            plan = sdk_plans[-batch_size + env]
            _assert_plan_history(plan, env, [9] * n_obs_steps)
            assert plan.anchor_timestamp_seconds == pytest.approx((n_obs_steps - 1) / 20.0)


def test_online_history_owns_snapshots_when_vector_env_reuses_buffers(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    batch = _live_batch(0)
    with _rollout_harness(config) as harness:
        for tick in range(3):
            current = _live_batch(tick)
            for key in (OBS_STATE, "observation.images.image", "observation.images.wrist_image"):
                batch[key].copy_(current[key])
            harness.policy.select_action(batch)
        for env in range(2):
            _assert_plan_history(sdk_plans[2 + env], env, [0, 1, 2])


def test_online_batch_size_change_requires_global_reset(tmp_path: Path, sdk_plans: list) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    with _rollout_harness(config) as harness:
        harness.policy.select_action(_live_batch(0))
        with pytest.raises(ValueError, match=r"batch size changed.*reset"):
            harness.policy.select_action(_live_batch(1, batch_size=1))
        harness.policy.reset()
        action = harness.policy.select_action(_live_batch(2, batch_size=1))
        assert action.shape == (1, 2)
        torch.testing.assert_close(action, _expected_chunk(config, _live_batch(2, batch_size=1))[:, 0])


def test_predict_action_chunk_supports_raw_batched_online_observations(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    with _rollout_harness(config) as harness:
        for tick in range(2):
            batch = _live_batch(tick)
            chunk = harness.policy.predict_action_chunk(batch)
            assert chunk.shape == (2, 4, 2)
            torch.testing.assert_close(chunk, _expected_chunk(config, batch))
        for env in range(2):
            _assert_plan_history(sdk_plans[2 + env], env, [0, 0, 1])


def test_settle_actions_preserve_batch_axis_without_advancing_history(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path, settle=2)
    stats = load_isaac_stats(stats_path)
    with _rollout_harness(config) as harness:
        for tick in range(2):
            action = harness.policy.select_action(_live_batch(tick))
            idle = np.tile([0.0, config.settle_gripper], (2, 1)).astype(np.float32)
            assert action.shape == (2, 2)
            torch.testing.assert_close(action, torch.from_numpy(normalize_isaac_actions(idle, stats.action)))
            assert not sdk_plans and not harness.sampler.inputs
        action = harness.policy.select_action(_live_batch(2))
        torch.testing.assert_close(action, _expected_chunk(config, _live_batch(2))[:, 0])
        for env, plan in enumerate(sdk_plans):
            _assert_plan_history(plan, env, [2, 2, 2])
            assert plan.anchor_timestamp_seconds == pytest.approx(0.1)


def _prepacked_rollout_batch(config: PerceptronIsaacConfig, batch_size: int) -> dict[str, Any]:
    assert config.native_stats_path is not None
    renderer = IsaacNativeMharmonyRenderer(
        metadata=IsaacMharmonyRenderMetadata.from_config(config),
        stats=load_isaac_stats(config.native_stats_path),
    )
    batch = _live_batch(0, batch_size)
    streams = [
        renderer.build(
            observation_window=[
                {
                    "images": {
                        "image": np.zeros((16, 16, 3), np.uint8),
                        "wrist_image": np.zeros((16, 16, 3), np.uint8),
                    },
                    "proprio": batch[OBS_STATE][env].numpy(),
                }
            ],
            prompt="move",
            device="cpu",
            dtype=torch.float32,
        )
        for env in range(batch_size)
    ]
    return {PERCEPTRON_ISAAC_STREAM_KEY: renderer.collate(streams)}


def test_prepacked_policy_queue_control_preserves_environment_and_horizon_axes(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    """Control: genuine B,H,D output drains H,B,D as B,D, without new rendering."""
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    batch = _live_batch(0)
    packed = _prepacked_rollout_batch(config, 2)
    with _rollout_harness(config) as harness:
        for tick in range(4):
            selected = harness.policy.select_action(packed)
            torch.testing.assert_close(
                selected, _expected_chunk(config, batch, packed_dtype=torch.float32)[:, tick % 2]
            )
            assert len(harness.sampler.inputs) == tick // 2 + 1
    assert len(sdk_plans) == 2


@pytest.mark.parametrize("initial_size,next_size", [(1, 2), (2, 1)])
@pytest.mark.parametrize(
    "first_packed,second_packed",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["raw-raw", "raw-packed", "packed-raw", "packed-packed"],
)
@pytest.mark.parametrize("settle", [0, 2], ids=["queued", "settling"])
def test_changed_rollout_batch_is_rejected_before_state_advances(
    tmp_path: Path,
    sdk_plans: list[IsaacMharmonyContentPlan],
    initial_size: int,
    next_size: int,
    first_packed: bool,
    second_packed: bool,
    settle: int,
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path, settle=settle)
    first = _prepacked_rollout_batch(config, initial_size) if first_packed else _live_batch(0, initial_size)
    second = _prepacked_rollout_batch(config, next_size) if second_packed else _live_batch(1, next_size)
    with _rollout_harness(config) as harness:
        policy = harness.policy
        assert policy.select_action(first).shape == (initial_size, config.action_dim)
        clocks = (policy._frame_index, policy._settle_index, policy._flow_seed_index)
        queued = tuple(action.clone() for action in policy._queues[ACTION])
        sample_calls = len(harness.sampler.inputs)
        with pytest.raises(ValueError, match=r"batch size changed.*reset"):
            policy.select_action(second)
        assert (policy._frame_index, policy._settle_index, policy._flow_seed_index) == clocks
        assert len(harness.sampler.inputs) == sample_calls
        assert len(policy._queues[ACTION]) == len(queued)
        for actual, expected in zip(policy._queues[ACTION], queued, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        policy.reset()
        assert policy.select_action(second).shape == (next_size, config.action_dim)


@pytest.mark.parametrize("initial_size,next_size", [(1, 2), (2, 1)])
def test_raw_chunk_batch_change_is_rejected_before_model_clock_advances(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan], initial_size: int, next_size: int
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    with _rollout_harness(config) as harness:
        policy = harness.policy
        policy.predict_action_chunk(_live_batch(0, initial_size))
        frame = policy._frame_index
        with pytest.raises(ValueError, match=r"batch size changed.*reset"):
            policy.predict_action_chunk(_live_batch(1, next_size))
        assert policy._frame_index == frame
        assert len(harness.sampler.inputs) == 1
        assert policy._online_state_queue[-1].shape[0] == initial_size
        policy.reset()
        chunk = policy.predict_action_chunk(_live_batch(2, next_size))
        assert chunk.shape == (next_size, config.chunk_size, config.action_dim)


@pytest.mark.parametrize("initial_size,next_size", [(1, 2), (2, 1)])
@pytest.mark.parametrize("initial_chunk", [False, True], ids=["queued", "direct"])
def test_prepacked_chunk_batch_change_is_rejected_before_sampling(
    tmp_path: Path,
    sdk_plans: list[IsaacMharmonyContentPlan],
    initial_size: int,
    next_size: int,
    initial_chunk: bool,
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    first = _prepacked_rollout_batch(config, initial_size)
    second = _prepacked_rollout_batch(config, next_size)
    with _rollout_harness(config) as harness:
        policy = harness.policy
        if initial_chunk:
            policy.predict_action_chunk(first)
        else:
            policy.select_action(first)
        clocks = (policy._frame_index, policy._settle_index, policy._flow_seed_index)
        queued = tuple(action.clone() for action in policy._queues[ACTION])
        sample_calls = len(harness.sampler.inputs)
        with pytest.raises(ValueError, match=r"batch size changed.*reset"):
            policy.predict_action_chunk(second)
        assert (policy._frame_index, policy._settle_index, policy._flow_seed_index) == clocks
        assert len(harness.sampler.inputs) == sample_calls
        assert len(policy._queues[ACTION]) == len(queued)
        for actual, expected in zip(policy._queues[ACTION], queued, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        policy.reset()
        chunk = policy.predict_action_chunk(second)
        assert chunk.shape == (next_size, config.chunk_size, config.action_dim)


class _DistinctEpisodeEnv(gym.Env):
    """Real vector-env lifecycle: one env ends early, the other keeps observing."""

    def __init__(self, env_index: int, episode_length: int) -> None:
        self.env_index = env_index
        self.episode_length = episode_length
        self._max_episode_steps = 5
        self.task = "move"
        self.task_description = "Task: Move " + "the block " * (env_index + 1) + "!"
        self.tick = 0
        self.episode = 0
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Dict(
                    {
                        camera: gym.spaces.Box(0, 255, (16, 16, 3), np.uint8)
                        for camera in ("image", "wrist_image")
                    }
                ),
                "agent_pos": gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def _observation(self) -> dict[str, Any]:
        batch = _live_batch(self.tick + self.episode * 10)
        return {
            "pixels": {
                camera: batch[f"observation.images.{camera}"][self.env_index].permute(1, 2, 0).numpy()
                for camera in ("image", "wrist_image")
            },
            "agent_pos": batch[OBS_STATE][self.env_index].numpy(),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.episode = 0 if seed is None else seed
        self.tick = 0
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert action.shape == (2,)
        self.tick += 1
        return self._observation(), 0.0, self.tick >= self.episode_length, False, {}


def test_upstream_batched_rollout_freezes_finished_env_and_resets_next_episode_group(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    """Exercise the maintained rollout(), not a new per-environment reset protocol."""
    from lerobot.envs.utils import FreezeAfterEpisodeEnd
    from lerobot.scripts.lerobot_eval import rollout

    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)
    env = gym.vector.SyncVectorEnv(
        [
            lambda: FreezeAfterEpisodeEnd(_DistinctEpisodeEnv(0, 2)),
            lambda: FreezeAfterEpisodeEnd(_DistinctEpisodeEnv(1, 5)),
        ],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(config)

    def run(policy: PerceptronIsaacPolicy, seed: int) -> dict[str, Any]:
        return rollout(
            env,
            policy,
            env_preprocessor=PolicyProcessorPipeline(steps=[]),
            env_postprocessor=PolicyProcessorPipeline(steps=[]),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=[seed, seed],
        )

    try:
        with _rollout_harness(config) as harness:
            first = run(harness.policy, 0)
            assert first[ACTION].shape == (2, 5, 2)
            torch.testing.assert_close(
                first["done"],
                torch.tensor([[False, True, True, True, True], [False, False, False, False, True]]),
            )
            assert len(harness.sampler.inputs) == 3
            # The completed environment stays terminal. The other history advances independently.
            _assert_plan_history(sdk_plans[4], 0, [2, 2, 2])
            _assert_plan_history(sdk_plans[5], 1, [2, 3, 4])
            assert not torch.equal(first[ACTION][0, :2], first[ACTION][1, :2])
            # Five steps leave half a chunk pending. rollout() must clear it before seed 1.
            reused = run(harness.policy, 1)
            _assert_plan_history(sdk_plans[6], 0, [10, 10, 10])
            _assert_plan_history(sdk_plans[7], 1, [10, 10, 10])
        with _rollout_harness(config) as fresh:
            fresh_result = run(fresh.policy, 1)
        torch.testing.assert_close(reused[ACTION], fresh_result[ACTION], rtol=0, atol=0)
        torch.testing.assert_close(reused["done"], fresh_result["done"])
    finally:
        env.close()
