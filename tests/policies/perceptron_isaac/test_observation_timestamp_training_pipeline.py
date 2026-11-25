"""A restored training pipeline renders episode-time tokens only for a recipe that emits them.

Fine-tuning restores the package's saved pipeline. Its inlined render metadata may predate
``emit_observation_timestamps``, so the restore resolves it from the checkpoint's recipe. The
stream below is rendered by the real pack step and mHarmony encoding; only FAST tokenization is
replaced by the shared spy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="requires lerobot[dataset]")
pytest.importorskip("mharmony", reason="mharmony is required (install lerobot[perceptron_isaac])")

import mharmony

from lerobot.policies.perceptron_isaac.mharmony_native import IsaacMharmonyRenderMetadata
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacMharmonyPackProcessorStep,
    make_perceptron_isaac_pre_post_processors,
    make_perceptron_isaac_pre_post_processors_from_pretrained,
)
from lerobot.policies.perceptron_isaac.tensor_stream import TextType
from tests.policies.perceptron_isaac._p2_data_fixtures import (
    FastInputSpy,
    data_batch,
    data_config,
    data_stats_payload,
)
from tests.policies.perceptron_isaac.test_p2_episode_footers import episode_metadata as _episode_metadata

episode_metadata = _episode_metadata


def _pack(pipeline) -> PerceptronIsaacMharmonyPackProcessorStep:
    return next(step for step in pipeline.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep))


@pytest.mark.parametrize(("recipe_emit", "expected_events"), [(False, 0), (True, 1), (None, 1)])
def test_restored_training_pipeline_follows_the_recipe(
    tmp_path: Path, episode_metadata, recipe_emit: bool | None, expected_events: int
) -> None:
    config = data_config(tmp_path, data_stats_payload())
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(config)
    # An imported package inlines render metadata; this one was saved before the field existed.
    inlined = IsaacMharmonyRenderMetadata.from_config(config, stats_fps=20.0).to_json_dict()
    del inlined["emit_observation_timestamps"]
    # Its training prompt contract is already inlined, so the restore needs no other sidecar.
    inlined["training_proprio_contract"] = {
        "schema": "perceptron_isaac_training_proprio_contract_v1",
        "source": "robotics_configurations.yml",
        "dataset_name": str(config.dataset_name),
        "output_name": "state",
        "sources": [{"key": "state", "dim": 2}],
        "components": [{"name": "joints", "source": "state", "range": [0, 2], "channel": "proprio"}],
        "excluded": [],
        "target_dim": 2,
        "contract_version": 1,
        "contract_hash": "0" * 64,
    }
    _pack(preprocessor).render_metadata = inlined
    package = tmp_path / "package"
    package.mkdir()
    preprocessor.save_pretrained(package)
    postprocessor.save_pretrained(package)
    if recipe_emit is not None:
        (package / "hf_model").mkdir()
        (package / "hf_model" / "policy_inference_recipe.json").write_text(
            json.dumps(
                {
                    "recipes": [
                        {
                            "policy_state_dataset": str(config.dataset_name),
                            "normalization_scope": "scope",
                            "objective": "Flow",
                            "lowering": {"observation": {"emit_observation_timestamps": recipe_emit}},
                        }
                    ]
                }
            )
        )
    config.pretrained_path = package

    restored, _ = make_perceptron_isaac_pre_post_processors_from_pretrained(
        config, str(package), dataset_meta=episode_metadata
    )

    pack = _pack(restored)
    assert pack.render_metadata["emit_observation_timestamps"] is (recipe_emit is not False)
    pack.dtype = "float32"
    pack._fast_processor = FastInputSpy()
    try:
        stream = restored(data_batch())[PERCEPTRON_ISAAC_STREAM_KEY]
    except mharmony.HarmonyError as error:
        if "vocab.json not found" not in str(error):
            raise
        # The restored flag is checked above. Counting rendered events needs the real encoding.
        pytest.skip("the local Qwen3.5 vocabulary that mHarmony encodes with is unavailable")
    timestamp_events = [event for event in stream.streams[0].events if event.type == TextType.timestamp]
    assert len(timestamp_events) == expected_events
