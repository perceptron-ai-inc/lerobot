"""Prompts carry episode-time items only where the checkpoint's recipe trained them.

Released YAM and SO-100/101 recipes set ``emit_observation_timestamps=false`` and LIBERO sets it
true. Render metadata saved before the field existed is resolved from the checkpoint's selected
recipe. The plan keeps the observation times either way; only the prompt item is conditional.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.isaac_stats import load_isaac_stats
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyContentPlan,
    IsaacMharmonyRenderMetadata,
    build_isaac_mharmony_content_plan,
    hydrate_isaac_observation_timestamps,
    load_native_render_metadata,
    read_recipe_observation_timestamps,
)
from tests.policies.perceptron_isaac._rollout_fixtures import _rollout_config, _write_rollout_stats
from tests.policies.perceptron_isaac.test_batched_online_rollout import (
    _live_batch,
    _rollout_harness,
    sdk_plans as _sdk_plans,
)

sdk_plans = _sdk_plans


def _recipe(dataset: str, scope: str, emit: object, *, objective: str = "Flow") -> dict[str, Any]:
    return {
        "policy_state_dataset": dataset,
        "normalization_scope": scope,
        "objective": objective,
        "lowering": {"observation": {"emit_observation_timestamps": emit}},
    }


def _write_recipe(path: Path, recipes: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "recipes": recipes}))
    return path


def _timestamp_items(plan: IsaacMharmonyContentPlan) -> list[dict[str, Any]]:
    return [item for item in plan.user_content if item["kind"] == "timestamp"]


@pytest.mark.parametrize(
    ("recipes", "scope", "expected"),
    [
        pytest.param([_recipe("yam", "a", False)], "a", False, id="selected_scope_off"),
        pytest.param([_recipe("yam", "a", True)], "a", True, id="selected_scope_on"),
        pytest.param([_recipe("yam", "a", False), _recipe("yam", "b", True)], "b", True, id="scope_narrows"),
        pytest.param(
            [_recipe("yam", "a", False), _recipe("yam", "b", False)], "missing", False, id="dataset_agrees"
        ),
        pytest.param(
            [_recipe("yam", "a", False), _recipe("other", "a", True)], "a", False, id="other_dataset_ignored"
        ),
        pytest.param(
            [_recipe("yam", "a", False), _recipe("yam", "a", True, objective="FAST")],
            "a",
            False,
            id="fast_objective_ignored",
        ),
        pytest.param([_recipe("other", "a", True)], "a", None, id="dataset_not_declared"),
        pytest.param(
            [{"policy_state_dataset": "yam", "normalization_scope": "a", "objective": "Flow"}],
            "a",
            None,
            id="flag_not_declared",
        ),
    ],
)
def test_recipe_flag_follows_the_selected_flow_recipe(
    tmp_path: Path, recipes: list[dict[str, Any]], scope: str, expected: bool | None
) -> None:
    path = _write_recipe(tmp_path / "policy_inference_recipe.json", recipes)
    assert (
        read_recipe_observation_timestamps(path, policy_state_dataset="yam", normalization_scope=scope)
        is expected
    )


def test_recipe_flag_rejects_ambiguous_and_malformed_declarations(tmp_path: Path) -> None:
    ambiguous = _write_recipe(
        tmp_path / "ambiguous.json", [_recipe("yam", "a", False), _recipe("yam", "b", True)]
    )
    with pytest.raises(ValueError, match="disagree"):
        read_recipe_observation_timestamps(ambiguous, policy_state_dataset="yam", normalization_scope="c")
    malformed = _write_recipe(tmp_path / "malformed.json", [_recipe("yam", "a", "false")])
    with pytest.raises(ValueError, match="must be a boolean"):
        read_recipe_observation_timestamps(malformed, policy_state_dataset="yam", normalization_scope="a")
    with pytest.raises(ValueError, match="boolean or null"):
        IsaacMharmonyRenderMetadata(emit_observation_timestamps="false")  # type: ignore[arg-type]


def _metadata_and_stats(tmp_path: Path, *, n_obs_steps: int = 3):
    stats_path = tmp_path / "stats.json"
    _write_rollout_stats(stats_path)
    config = _rollout_config(stats_path, n_obs_steps=n_obs_steps)
    stats = load_isaac_stats(str(stats_path))
    metadata = IsaacMharmonyRenderMetadata.from_config(config, stats_fps=stats.target_fps)
    return config, metadata, stats


def _plan(metadata: IsaacMharmonyRenderMetadata, stats) -> IsaacMharmonyContentPlan:
    frame = {
        "images": {camera: np.zeros((16, 16, 3), dtype=np.uint8) for camera in metadata.camera_order},
        "proprio": np.zeros(metadata.proprio_dim, dtype=np.float32),
    }
    return build_isaac_mharmony_content_plan(
        observation_window=[frame] * metadata.n_obs_steps,
        prompt="move the block",
        metadata=metadata,
        stats=stats,
        anchor_timestamp_seconds=1.5,
    )


@pytest.mark.parametrize(("emit", "expected_items"), [(False, 0), (True, 3), (None, 3)])
def test_plan_emits_timestamp_items_only_when_the_recipe_does(
    tmp_path: Path, emit: bool | None, expected_items: int
) -> None:
    _, metadata, stats = _metadata_and_stats(tmp_path)
    plan = _plan(replace(metadata, emit_observation_timestamps=emit), stats)
    assert len(_timestamp_items(plan)) == expected_items
    # Capture times stay on the plan and on each observation for scheduling and alignment.
    assert plan.anchor_timestamp_seconds == 1.5
    assert len(plan.observation_timestamps_seconds) == 3
    images = [item for item in plan.user_content if item["kind"] == "image"]
    assert [item["metadata"]["timestamp_seconds"] for item in images[-2:]] == [1.5, 1.5]
    # Apart from the timestamp items, the prompt has the same items in the same order.
    emitting = _plan(replace(metadata, emit_observation_timestamps=True), stats)

    def other_items(candidate: IsaacMharmonyContentPlan) -> list[tuple[str, object, object]]:
        return [
            (item["kind"], item.get("text"), item.get("media_id"))
            for item in candidate.user_content
            if item["kind"] != "timestamp"
        ]

    assert other_items(plan) == other_items(emitting)


def test_hydration_prefers_recorded_value_then_recipe_then_legacy_emission(tmp_path: Path) -> None:
    _, metadata, _ = _metadata_and_stats(tmp_path)
    dataset = str(metadata.dataset_name)
    off = _write_recipe(tmp_path / "off.json", [_recipe(dataset, "scope", False)])
    assert metadata.emit_observation_timestamps is None
    assert (
        hydrate_isaac_observation_timestamps(metadata, recipe_path=off).emit_observation_timestamps is False
    )
    recorded = replace(metadata, emit_observation_timestamps=True)
    assert hydrate_isaac_observation_timestamps(recorded, recipe_path=off).emit_observation_timestamps is True
    assert (
        hydrate_isaac_observation_timestamps(metadata, recipe_path=None).emit_observation_timestamps is True
    )
    missing = tmp_path / "absent.json"
    assert (
        hydrate_isaac_observation_timestamps(metadata, recipe_path=missing).emit_observation_timestamps
        is True
    )


def _package_config(tmp_path: Path, *, recipe_emit: bool | None, n_obs_steps: int) -> PerceptronIsaacConfig:
    """A package whose render metadata predates the field, with an optional hf_model recipe."""
    config, metadata, _ = _metadata_and_stats(tmp_path, n_obs_steps=n_obs_steps)
    package = tmp_path / "package"
    package.mkdir()
    saved = metadata.to_json_dict()
    del saved["emit_observation_timestamps"]
    (package / "native_render_metadata.json").write_text(json.dumps(saved))
    if recipe_emit is not None:
        _write_recipe(
            package / "hf_model" / "policy_inference_recipe.json",
            [_recipe(str(metadata.dataset_name), "scope", recipe_emit)],
        )
    config.native_render_metadata_path = str(package / "native_render_metadata.json")
    config.pretrained_path = package
    return config


@pytest.mark.parametrize(("recipe_emit", "expected"), [(False, False), (True, True), (None, True)])
def test_loader_reads_the_recipe_for_metadata_that_predates_the_field(
    tmp_path: Path, recipe_emit: bool | None, expected: bool
) -> None:
    config = _package_config(tmp_path, recipe_emit=recipe_emit, n_obs_steps=3)
    assert load_native_render_metadata(config).emit_observation_timestamps is expected


def test_sync_and_chunk_inference_omit_timestamp_items_for_a_timestamp_free_recipe(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    config = _package_config(tmp_path, recipe_emit=False, n_obs_steps=1)
    with _rollout_harness(config) as harness:
        harness.policy.select_action(_live_batch(0, batch_size=1))
        assert len(sdk_plans) == 1
        assert _timestamp_items(sdk_plans[-1]) == []
        harness.policy.reset()
        chunk = harness.policy.predict_action_chunk(
            _live_batch(1, batch_size=1), inference_delay=0, prev_chunk_left_over=None
        )
        assert chunk.shape == (1, config.chunk_size, config.action_dim)
        assert len(sdk_plans) == 2
        assert _timestamp_items(sdk_plans[-1]) == []
        # The model has no time input, so chunk callers owe it no episode time.
        spec = harness.policy.chunk_inference_spec()
        assert spec.prediction_steps == config.chunk_size
        assert not spec.needs_episode_time


def test_timestamp_emitting_recipe_keeps_sync_tokens_and_asks_chunk_callers_for_episode_time(
    tmp_path: Path, sdk_plans: list[IsaacMharmonyContentPlan]
) -> None:
    config = _package_config(tmp_path, recipe_emit=True, n_obs_steps=1)
    with _rollout_harness(config) as harness:
        harness.policy.select_action(_live_batch(0, batch_size=1))
        items = _timestamp_items(sdk_plans[-1])
        assert len(items) == 1
        assert items[0]["metadata"]["timestamp_seconds"] == 0.0
        assert harness.policy.chunk_inference_spec().needs_episode_time
