"""RTC provenance copies require typed identity, not Python's False == 0 equality."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import (
    Mk1CheckpointContract,
    Mk1CheckpointContractError,
)
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _write_portable_isaac05_checkpoint,
)
from tests.policies.perceptron_isaac.test_p2_native_regressions import _checkpoint_bytes


@dataclass(frozen=True)
class RootRtcTypeMutation:
    field: Literal["rtc_max_delay_steps", "rtc_probability", "rtc_poisson_mean"]
    nested: int | float
    root: bool | float


@pytest.mark.parametrize("reader", ["parser", "loader"])
@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(RootRtcTypeMutation("rtc_max_delay_steps", 0, False), id="budget_bool"),
        pytest.param(RootRtcTypeMutation("rtc_max_delay_steps", 0, 0.0), id="budget_float"),
        pytest.param(RootRtcTypeMutation("rtc_probability", 0, False), id="probability_bool"),
        pytest.param(RootRtcTypeMutation("rtc_poisson_mean", 1.0, True), id="poisson_bool"),
    ],
)
def test_native_rtc_root_copy_rejects_equal_but_invalid_types_without_rewriting(
    tmp_path: Path, reader: str, mutation: RootRtcTypeMutation
) -> None:
    root = tmp_path / "native"
    _, raw, _ = _write_portable_isaac05_checkpoint(root)
    for action in (raw["action_expert"], raw["isaac05_vla"]["action_expert"]):
        action.update(
            rtc_max_delay_steps=0,
            rtc_probability=0,
            rtc_delay_sampling="exponential",
            rtc_poisson_mean=1.0,
        )
        action[mutation.field] = mutation.nested
    valid = copy.deepcopy(raw)
    # This unpinned, inactive recipe must remain accepted before the root-only mutation.
    contract = Mk1CheckpointContract.parse_allowlisted(valid, allow_test_only_reduced_geometry=True)
    metadata = dict(contract.action_expert.native_metadata)
    assert type(metadata["rtc_max_delay_steps"]) is int
    assert metadata["rtc_max_delay_steps"] == 0
    assert valid == raw
    raw["action_expert"][mutation.field] = mutation.root
    assert raw["action_expert"] == raw["isaac05_vla"]["action_expert"]
    assert type(raw["action_expert"][mutation.field]) is not type(mutation.nested)
    (root / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    before_files = _checkpoint_bytes(root)
    before_raw = copy.deepcopy(raw)
    try:
        with pytest.raises(Mk1CheckpointContractError, match=mutation.field):
            if reader == "parser":
                Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
            else:
                load_mk1_vla_from_hf(
                    root,
                    dtype=torch.float32,
                    device="cpu",
                    allow_test_only_reduced_geometry=True,
                    allowed_storage_dtypes=frozenset({"F32"}),
                    load_intent="training",
                )
    finally:
        assert raw == before_raw
        assert _checkpoint_bytes(root) == before_files
