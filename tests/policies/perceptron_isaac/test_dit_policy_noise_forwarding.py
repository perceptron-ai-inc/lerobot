"""Forward caller flow_noise through the real native policy/model to the tiny DiT.

The policy receives an already packed stream; no rendering, weights, backbone,
rollout, or external service is involved.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from tests.policies.perceptron_isaac._rollout_fixtures import _rollout_config, _write_rollout_stats
from tests.policies.perceptron_isaac.test_dit_batch_rng_alignment import (
    SamplerModelShell,
    _context_inputs,
    _prepacked_stream,
    _reference_euler,
    _tiny_sampler,
)


@pytest.mark.parametrize("seed_base", [None, 1234], ids=["global-rng", "existing-seed-fork"])
@pytest.mark.parametrize("queued", [False, True], ids=["chunk", "select-action"])
def test_policy_flow_noise_reaches_real_sampler_without_consuming_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed_base: int | None, queued: bool
) -> None:
    monkeypatch.delenv("ISAAC_FLOW_SEED_BASE", raising=False)
    with torch.random.fork_rng(devices=[]):
        stats_path = tmp_path / "stats.json"
        _write_rollout_stats(stats_path)
        config = replace(
            _rollout_config(stats_path),
            num_inference_steps=10,
            num_flow_samples=8,
            n_action_steps=2,
            flow_seed_base=seed_base,
        )
        policy = PerceptronIsaacPolicy(config)
        head, expert = _tiny_sampler(layerwise=True)
        layers, _ = _context_inputs(1)
        model = SamplerModelShell(head, layers)
        noise = torch.arange(8 * 4 * 64, dtype=torch.float32).reshape(8, 4, 64) / 1000
        original = noise.clone()
        expected = _reference_euler(
            noise,
            tuple(layer.repeat(8, 1, 1) for layer in layers),
            torch.tensor([[1, 1, 1, 0]]).repeat(8, 1),
            config.action_dim,
        ).mean(0, keepdim=True)[..., : config.action_dim]
        # Only bypass checkpoint loading. The complete policy/model/sampler forwarding path is real.
        with patch.object(policy, "_isaac_model", model):
            for seed in (37, 901):
                policy.reset()
                torch.manual_seed(seed)
                before = torch.get_rng_state().clone()
                expert.reads.clear()
                batch = {PERCEPTRON_ISAAC_STREAM_KEY: _prepacked_stream()}
                if queued:
                    actual = policy.select_action(batch, flow_noise=noise)
                    torch.testing.assert_close(actual, expected[:, 0], rtol=1e-6, atol=1e-6)
                    assert actual.shape == (1, config.action_dim)
                    # Noise belongs to a replan; a cached action must not run or redraw the expert.
                    cached = policy.select_action(batch, flow_noise=noise + 100)
                    torch.testing.assert_close(cached, expected[:, 1], rtol=1e-6, atol=1e-6)
                else:
                    actual = policy.predict_action_chunk(batch, flow_noise=noise)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                    assert actual.shape == (1, config.chunk_size, config.action_dim)
                assert torch.equal(torch.get_rng_state(), before)
                torch.testing.assert_close(noise, original, rtol=0, atol=0)
                assert len(expert.reads) == 10
                assert all(read.state.shape == (8, config.chunk_size, 64) for read in expert.reads)
