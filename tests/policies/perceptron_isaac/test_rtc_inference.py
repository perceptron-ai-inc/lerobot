"""Prefix-pinned RTC inference for the native DiT head and the policy's chunk refills."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

pytest.importorskip("transformers", reason="transformers is required (install lerobot[perceptron_isaac])")

from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mharmony_native import IsaacMharmonyContentPlan
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import build_action_expert_head
from lerobot.policies.perceptron_isaac.tensor_stream import TensorStream
from tests.policies.perceptron_isaac._rollout_fixtures import _rollout_config, _write_rollout_stats
from tests.policies.perceptron_isaac.test_action_expert_contract import _dit_contract
from tests.policies.perceptron_isaac.test_batched_online_rollout import (
    _live_batch,
    _ProprioChunkSampler,
    _rollout_harness,
    sdk_plans as _sdk_plans,
)

sdk_plans = _sdk_plans


def _rtc_head(budget: int = 3):
    torch.manual_seed(5)
    head = build_action_expert_head(
        _dit_contract(rtc_max_delay_steps=budget, rtc_probability=0.5), vlm_dim=16
    ).eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.uniform_(-0.2, 0.2)
    return head


@pytest.mark.parametrize("num_flow_samples", [1, 3])
def test_per_row_prefix_lengths_are_pinned_for_every_flow_candidate(num_flow_samples: int) -> None:
    head = _rtc_head()
    context = torch.randn(2, 3, 16)
    prefix = torch.stack([torch.full((2, 4), 0.3), torch.full((2, 4), -0.7)])
    lengths = torch.tensor([1, 2])
    actions = head.sample(
        context, action_prefix=prefix, prefix_length=lengths, num_flow_samples=num_flow_samples
    )
    assert actions.shape == (2, 6, 4)
    torch.testing.assert_close(actions[0, :1], prefix[0, :1], rtol=0, atol=0)
    torch.testing.assert_close(actions[1, :2], prefix[1, :2], rtol=0, atol=0)
    # Row 0 has only one pinned row; its second row is generated.
    assert not torch.equal(actions[0, 1], prefix[0, 1])


def test_prefix_conditions_the_generated_suffix() -> None:
    head = _rtc_head()
    context = torch.randn(1, 3, 16)
    noise = torch.randn(6, 4)
    free = head.sample(context, noise=noise)
    pinned = head.sample(context, noise=noise, action_prefix=torch.full((1, 2, 4), 0.9), prefix_length=2)
    torch.testing.assert_close(pinned[:, :2], torch.full((1, 2, 4), 0.9), rtol=0, atol=0)
    # Same noise, different clean prefix: per-row conditioning changes the suffix.
    assert not torch.allclose(pinned[:, 2:], free[:, 2:])


def test_zero_prefix_length_matches_prefix_free_sampling_exactly() -> None:
    head = _rtc_head()
    context = torch.randn(1, 3, 16)
    noise = torch.randn(6, 4)
    free = head.sample(context, noise=noise)
    zero = head.sample(context, noise=noise, action_prefix=torch.zeros(1, 2, 4), prefix_length=0)
    torch.testing.assert_close(zero, free, rtol=0, atol=0)


def test_narrow_action_dim_prefix_respects_dimension_mask() -> None:
    head = _rtc_head()
    prefix = torch.full((1, 2, 3), 0.4)
    actions = head.sample(torch.randn(1, 3, 16), action_dim=3, action_prefix=prefix, prefix_length=2)
    torch.testing.assert_close(actions[:, :2, :3], prefix, rtol=0, atol=0)
    assert torch.count_nonzero(actions[..., 3]) == 0


@pytest.mark.parametrize(
    ("n_action_steps", "rtc_prefix_length", "message"),
    [(2, -1, "rtc_prefix_length must be >= 0"), (2, 3, "cannot exceed chunk_size")],
)
def test_config_bounds_rtc_prefix_length(
    tmp_path: Path, n_action_steps: int, rtc_prefix_length: int, message: str
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    base = _rollout_config(stats_path)
    with pytest.raises(ValueError, match=message):
        PerceptronIsaacConfig(
            device="cpu",
            native_stats_path=str(stats_path),
            chunk_size=base.chunk_size,
            n_action_steps=n_action_steps,
            rtc_prefix_length=rtc_prefix_length,
            action_dim=base.action_dim,
            proprio_dim=base.proprio_dim,
            image_size=base.image_size,
            camera_order=base.camera_order,
            input_features=base.input_features,
            output_features=base.output_features,
        )


class _PrefixRecordingSampler(_ProprioChunkSampler):
    """Pins any requested prefix, like the DiT head, and records what it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.prefixes: list[tuple[Tensor | None, int | None]] = []

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
        chunk = super().sample_action(
            stream,
            num_steps=num_steps,
            action_dim=action_dim,
            num_action_steps=num_action_steps,
            num_flow_samples=num_flow_samples,
        )
        self.prefixes.append((None if action_prefix is None else action_prefix.clone(), prefix_length))
        if action_prefix is not None:
            assert prefix_length is not None
            chunk[:, :prefix_length] = action_prefix[:, :prefix_length]
        return chunk


def test_select_action_pins_previous_chunk_unexecuted_rows(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path)  # chunk_size=4, n_action_steps=2
    config.rtc_prefix_length = 2
    with _rollout_harness(config) as harness:
        sampler = _PrefixRecordingSampler()
        harness.policy._isaac_model = sampler
        chunks: list[Tensor] = []
        for tick in range(6):
            harness.policy.select_action(_live_batch(tick, batch_size=2))
            if tick % config.n_action_steps == 0:
                assert harness.policy._last_norm_chunk is not None
                chunks.append(harness.policy._last_norm_chunk.clone())
        assert len(sampler.prefixes) == 3
        assert sampler.prefixes[0] == (None, None)
        for refill in (1, 2):
            prefix, length = sampler.prefixes[refill]
            assert length == 2
            assert prefix is not None
            # Each environment's prefix is its own previous chunk rows [n_action_steps, n + P).
            torch.testing.assert_close(prefix, chunks[refill - 1][:, 2:4], rtol=0, atol=0)
        # A global reset drops the carried chunk.
        harness.policy.reset()
        harness.policy._isaac_model = sampler
        harness.policy.select_action(_live_batch(9, batch_size=2))
        assert sampler.prefixes[-1] == (None, None)


def test_select_action_owns_the_sync_prefix(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    with _rollout_harness(_rollout_config(stats_path)) as harness:
        batch: dict[str, Any] = _live_batch(0)
        with pytest.raises(ValueError, match="owns the RTC prefix"):
            harness.policy.select_action(batch, action_prefix=torch.zeros(2, 1, 2))
        # The RTC engine's continuation belongs to predict_action_chunk (test_rtc_async_engine.py).
        with pytest.raises(ValueError, match="sync path"):
            harness.policy.select_action(batch, inference_delay=1, prev_chunk_left_over=torch.zeros(2, 2))
