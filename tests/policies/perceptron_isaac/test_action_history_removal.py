"""Reject historical actions while retaining visual/proprio history and FAST targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.isaac_stats import (
    load_isaac_stats,
    normalize_isaac_actions,
)
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyContentPlan,
    IsaacMharmonyRenderMetadata,
    IsaacNativeMharmonyRenderer,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.tensor_stream import (
    TextType,
)
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
from tests.policies.perceptron_isaac._rollout_fixtures import (
    _assert_plan_history,
    _rollout_config,
    _write_rollout_stats,
    sdk_plans as _sdk_plans,
)

sdk_plans = _sdk_plans


@pytest.mark.parametrize("objective", ["flow", "mixed"])
def test_enabled_legacy_action_history_is_rejected_by_real_policy_config(objective: str) -> None:
    with pytest.raises(ValueError, match=r"(?i)(historical|executed.action|action_conditioning)"):
        PerceptronIsaacConfig(device="cpu", objective=objective, action_conditioning=True)


def test_saved_processor_cannot_reenable_removed_action_history(tmp_path: Path) -> None:
    saved = {
        "steps": [
            {
                "registry_name": "perceptron_isaac_mharmony_pack",
                "config": {"enabled": True, "action_conditioning": True},
            }
        ]
    }
    (tmp_path / "processor.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=r"(?i)(historical|executed.action|action_conditioning)"):
        PolicyProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="processor.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )


class _RecordingFastCodec:
    """Tokenizer-only boundary; real native codec numerics have separate controls."""

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def __call__(self, actions: np.ndarray) -> list[list[int]]:
        self.actions.append(np.asarray(actions).copy())
        return [[17, 23]]


def _history_window() -> list[dict[str, Any]]:
    return [
        {
            "images": {
                "image": np.full((16, 16, 3), 20 + tick, dtype=np.uint8),
                "wrist_image": np.full((16, 16, 3), 40 + tick, dtype=np.uint8),
            },
            "proprio": np.asarray([0.1 + tick * 0.01, -0.2], dtype=np.float32),
            "previous_action": np.asarray([0.7, -0.5], dtype=np.float32) if tick else None,
        }
        for tick in range(3)
    ]


@pytest.mark.parametrize("route", ["inference", "training"])
def test_enabled_history_metadata_is_rejected_on_both_render_paths(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan], route: str
) -> None:
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    metadata = IsaacMharmonyRenderMetadata.from_config(_rollout_config(stats_path)).to_json_dict()
    metadata["action_conditioning"] = True
    metadata_path = tmp_path / "legacy_render.json"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"(?i)(historical|executed.action|action_conditioning)"):
        renderer = IsaacNativeMharmonyRenderer(
            metadata=IsaacMharmonyRenderMetadata.from_json_file(metadata_path),
            stats=load_isaac_stats(stats_path),
        )
        if route == "inference":
            renderer.build(
                observation_window=_history_window(),
                prompt="move",
                device="cpu",
            )
        else:
            renderer.build_training(
                observation_window=_history_window(),
                prompt="move",
                action_chunk=np.zeros((4, 2), dtype=np.float32),
                fast_processor=_RecordingFastCodec(),
                device="cpu",
            )


def test_policy_does_not_expose_executed_action_recorder_lifecycle() -> None:
    assert "record_executed_action" not in vars(PerceptronIsaacPolicy)


def test_disabled_history_preserves_training_fast_targets_and_visual_proprio_history(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    """Control: remove conditioning, not FAST action supervision or observation history."""
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    stats = load_isaac_stats(stats_path)
    renderer = IsaacNativeMharmonyRenderer(
        metadata=IsaacMharmonyRenderMetadata.from_config(_rollout_config(stats_path)), stats=stats
    )
    codec = _RecordingFastCodec()
    actions = np.asarray([[0.1, -0.2], [0.2, -0.1], [0.3, 0.0], [0.4, 0.1]], dtype=np.float32)
    stream = renderer.build_training(
        observation_window=_history_window(),
        prompt="move the block",
        action_chunk=actions,
        fast_processor=codec,
        action_is_pad=[False, False, False, True],
        anchor_timestamp_seconds=0.1,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(sdk_plans) == 1
    plan = sdk_plans[0]
    _assert_plan_history(plan, 0, [0, 1, 2])
    assert [item["kind"] for item in plan.user_content].count("vector") == 1
    assert len(codec.actions) == 1, "FAST encodes the action target chunk, never executed-action history."
    expected = normalize_isaac_actions(actions, stats.action)
    np.testing.assert_allclose(codec.actions[0], expected[None, ...])
    flow, fast = plan.assistant_content
    np.testing.assert_allclose(flow["tags"]["action_target"], expected)
    assert fast["tokens"] == [17, 23]
    assert flow["tags"]["action_is_pad"] == fast["tags"]["action_is_pad"] == [False, False, False, True]
    assert [event.type for event in stream.streams[0].events][-2:] == [TextType.action_c, TextType.action]
