"""P2.09/P2.14: native RTC-off metadata and retired execution refusals.

Only fixture JSON is varied. Native parsing, inventory validation and tensor loading
remain real; the input checkpoint bytes must never be migrated in place.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import Tensor

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    Mk1CheckpointContractError,
)
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import (
    build_action_expert_head,
    setup_qwen35_vla_for_training,
)
from tests.policies.perceptron_isaac.test_action_expert_contract import _dit_contract, _molmoact_cfg
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _write_portable_isaac05_checkpoint,
)


def _checkpoint_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(
    ("cap", "probability", "sampling", "mean"),
    [
        pytest.param(0, None, "uniform", 3.0, id="rtc_disabled_zero_budget"),
        pytest.param(0, 0.0, "exponential", 1.2, id="rtc_disabled_inactive_recipe"),
        pytest.param(12, 0.0, "poisson", 7.0, id="rtc_disabled_probability_zero"),
        pytest.param(12, 0.5, "poisson", 5.0, id="rtc_capable_source"),
    ],
)
def test_native_parser_and_loader_accept_rtc_off_and_capable_source_without_rewriting(
    tmp_path: Path,
    cap: int,
    probability: float | None,
    sampling: str,
    mean: float,
) -> None:
    root = tmp_path / "native"
    _, raw, value = _write_portable_isaac05_checkpoint(root)
    profile = {
        "rtc_max_delay_steps": cap,
        "rtc_probability": probability,
        "rtc_delay_sampling": sampling,
        "rtc_poisson_mean": mean,
    }
    for action in (raw["action_expert"], raw["isaac05_vla"]["action_expert"]):
        action.update(profile)
    (root / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    original = copy.deepcopy(raw)
    before = _checkpoint_bytes(root)
    # This is the native isaac_0_5 normalization/parser, not a legacy or skipped path.
    contract = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
    assert raw == original
    source_action = dict(contract.action_expert.native_metadata)
    for name, expected in profile.items():
        assert source_action[name] == expected
    model, _, loaded = load_mk1_vla_from_hf(
        root,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
        allowed_storage_dtypes=frozenset({"F32"}),
        load_intent="training",
    )
    assert loaded == contract
    assert set(model.state_dict()) == set(contract.expected_tensor_shapes())
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, torch.full_like(parameter, value.item()), rtol=0, atol=0)
    assert _checkpoint_bytes(root) == before


@pytest.mark.parametrize("mismatch", ["duplicate_profile", "architecture"])
def test_rtc_off_native_acceptance_keeps_structural_integrity(tmp_path: Path, mismatch: str) -> None:
    _, raw, _ = _write_portable_isaac05_checkpoint(tmp_path / "native")
    # The original supported recipe makes this control independent of the new off-profile acceptance.
    if mismatch == "duplicate_profile":
        raw["action_expert"]["rtc_probability"] = 0.0
    else:
        raw["architectures"] = ["UnsupportedArchitecture"]
    message = "identical" if mismatch == "duplicate_profile" else "architectures"
    with pytest.raises(Mk1CheckpointContractError, match=message):
        Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)


@pytest.mark.parametrize("allow_ood", [False, True])
@pytest.mark.parametrize("length_form", ["inferred", "scalar", "list", "tensor"])
def test_rtc_capable_dit_rejects_execution_prefix_even_with_ood_override(
    allow_ood: bool, length_form: str
) -> None:
    head = build_action_expert_head(_dit_contract(rtc_max_delay_steps=2, rtc_probability=0.5), vlm_dim=16)
    prefix_length = _prefix_length(length_form)
    with pytest.raises(ValueError, match="(?i)(rtc|prefix)"):
        head.sample(
            torch.zeros(1, 3, 16),
            action_prefix=torch.zeros(1, 1, 4),
            prefix_length=prefix_length,
            allow_ood_rtc_prefix=allow_ood,
        )


def _prefix_length(length_form: str) -> int | list[int] | Tensor | None:
    match length_form:
        case "inferred":
            return None
        case "scalar":
            return 1
        case "list":
            return [1]
        case "tensor":
            return torch.tensor([1])
        case _:
            raise AssertionError(f"Unknown prefix length fixture: {length_form}")


@pytest.mark.parametrize("length_form", ["inferred", "scalar", "tensor"])
def test_legacy_expert_also_rejects_rtc_execution_prefix(length_form: str) -> None:
    head = build_action_expert_head(_molmoact_cfg(rtc_max_delay_steps=2), vlm_dim=16)
    with pytest.raises(ValueError, match="(?i)(rtc|prefix)"):
        head.sample(
            torch.zeros(1, 3, 16),
            action_prefix=torch.zeros(1, 1, 4),
            prefix_length=_prefix_length(length_form),
        )


@pytest.mark.parametrize("draws", [1, 2])
def test_expert_training_rejects_rtc_per_row_time_conditioning(draws: int) -> None:
    head = build_action_expert_head(_dit_contract(rtc_max_delay_steps=2, rtc_probability=0.5), vlm_dim=16)
    states = torch.zeros(1, 6, 4)
    # A clean prefix and a noisy suffix invoke the removed dual-time RTC path.
    times = torch.tensor([[0.0, 0.5, 0.5, 0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="(?i)(rtc|time|prefix)"):
        head(
            torch.zeros(1, 3, 16),
            torch.ones(1, 3, dtype=torch.bool),
            states if draws == 1 else states.unsqueeze(0).expand(draws, -1, -1, -1),
            times if draws == 1 else times.unsqueeze(0).expand(draws, -1, -1),
        )


@pytest.mark.parametrize("rtc_request", ["delay", "dual_timestep"])
def test_training_setup_rejects_requested_rtc_delay(tmp_path: Path, rtc_request: str) -> None:
    root = tmp_path / "native"
    _write_portable_isaac05_checkpoint(root)
    model, _, _ = load_mk1_vla_from_hf(
        root,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
        allowed_storage_dtypes=frozenset({"F32"}),
        load_intent="training",
    )
    with pytest.raises(ValueError, match="(?i)(rtc|delay|timestep)"):
        if rtc_request == "delay":
            setup_qwen35_vla_for_training(model, rtc_max_delay_steps=2)
        else:
            setup_qwen35_vla_for_training(model, dual_timestep_ratio=0.5)


def test_rtc_capable_weights_keep_ordinary_sampling_abi_and_rng_order() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(19)
        head = build_action_expert_head(_dit_contract(rtc_max_delay_steps=2, rtc_probability=0.5), vlm_dim=16)
        with torch.no_grad():
            head.action_expert.final_layer.linear.weight.normal_(std=0.03)
            head.action_expert.final_layer.linear.bias.fill_(0.2)
        reference = build_action_expert_head(_molmoact_cfg(), vlm_dim=16)
        reference.load_state_dict(head.state_dict(), strict=True)
        context = torch.linspace(-1, 1, 48).reshape(1, 3, 16)
        torch.manual_seed(23)
        actual = head.sample(context, num_steps=2)
        actual_rng = torch.get_rng_state()
        torch.manual_seed(23)
        expected = reference.sample(context, num_steps=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(actual_rng, torch.get_rng_state())
