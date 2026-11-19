"""Auxiliary diagnostics follow enable coefficients without hiding active overflow."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import perceptron_isaac_training_loss
from tests.policies.perceptron_isaac.test_p2_moe_followup import (
    _all_null_native_model,
    _capture_native_forward,
    _native_blocks,
    _padded_stream,
)


@pytest.mark.parametrize(
    ("balance", "z"),
    [(0.0, 0.0), (0.02, 0.0), (0.0, 0.001), (0.02, 0.001)],
    ids=["both_disabled", "balance_only", "z_only", "both_enabled"],
)
def test_native_auxiliary_diagnostics_follow_independent_coefficients_without_hiding_active_overflow(
    balance: float, z: float
) -> None:
    model = _all_null_native_model()
    with torch.no_grad():
        for block in _native_blocks(model):
            block.gate.weight[:, 0] = 2e20
    stream = _padded_stream(0)
    capture = _capture_native_forward(model, stream)
    assert all(torch.isfinite(logits).all() for logits in capture.logits)
    assert all(not bool(torch.isfinite(value)) for value in capture.output.additional_info["router_z_loss"])
    result = perceptron_isaac_training_loss(
        capture.output,
        stream,
        flow_timesteps=torch.tensor([[0.5]]),
        flow_noise=torch.zeros(1, 1, 3, 4),
        moe_load_balancing_loss_coeff=balance,
        moe_z_loss_coeff=z,
    )
    assert ("load_balancing_loss" in result.metrics) == bool(balance)
    assert ("router_z_loss" in result.metrics) == bool(z)
    if balance:
        assert torch.isfinite(result.metrics["load_balancing_loss"])
    if z:
        assert not torch.isfinite(result.metrics["router_z_loss"])
    for value in (result.loss, result.per_sample_loss, result.metrics["loss"]):
        assert bool(torch.isfinite(value).all()) == (z == 0.0)
