"""Tiny schema-2 declarations built from the maintained schema-1 source fixture."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract
from tests.policies.perceptron_isaac.test_action_expert_contract import _dit_contract
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import (
    _write_portable_isaac05_checkpoint,
)


def layerwise_action_config(
    *, layers: tuple[int, ...] = (0, 1), hidden_norm: str = "final_norm"
) -> dict[str, Any]:
    """Keep schema-1 payload fields; schema 2 adds only the conditioning triple."""
    return _dit_contract(
        schema_version=2,
        num_layers=len(layers),
        hidden_dim=8,
        num_heads=2,
        action_horizon=3,
        timestep_embed_dim=4,
        ffn_multiple_of=4,
        context_source="layer_hidden",
        context_layers=list(layers),
        context_hidden_norm=hidden_norm,
    )


def declare_native_layerwise(raw: dict[str, Any], *, layers: tuple[int, ...] = (0, 1)) -> None:
    """Upgrade both native action copies, not the independent schema-1 VLA attachment envelope."""
    action = raw["action_expert"]
    action.update(
        schema_version=2,
        context_source="layer_hidden",
        context_layers=list(layers),
        context_hidden_norm="final_norm",
    )
    raw["isaac05_vla"]["action_expert"] = copy.deepcopy(action)


@dataclass(frozen=True)
class LayerwiseCheckpoint:
    root: Path
    raw: dict[str, Any]
    schema1_contract: Mk1CheckpointContract
    source_value: torch.Tensor


def write_layerwise_checkpoint(root: Path) -> LayerwiseCheckpoint:
    """Write real tiny F32 shards using the unchanged geometry, then declare schema 2 honestly."""
    contract, raw, value = _write_portable_isaac05_checkpoint(root)
    declare_native_layerwise(raw)
    (root / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return LayerwiseCheckpoint(root, raw, contract, value)
