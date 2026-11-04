"""Focused P2 data regressions; original RED preserved in data-red-frozen.patch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.utils import delta_window
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.perceptron_isaac.isaac_stats import (
    isaac_stats_from_dict,
    normalize_isaac_actions,
)
from lerobot.policies.perceptron_isaac.mharmony_adapter import load_mharmony_encoding
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_RENDER_META_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacMharmonyPackProcessorStep,
    make_perceptron_isaac_pre_post_processors,
    make_perceptron_isaac_pre_post_processors_from_pretrained,
)
from lerobot.policies.perceptron_isaac.tensor_stream import TensorStream, TextType, VectorType
from lerobot.processor.converters import batch_to_transition
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, TRUNCATED
from tests.policies.perceptron_isaac._p2_data_fixtures import (
    FastInputSpy,
    action_event,
    data_batch,
    data_config,
    data_pack,
    data_stats_payload,
    final_footer_id,
)


@pytest.fixture
def episode_metadata(tmp_path: Path) -> LeRobotDatasetMetadata:
    """Real v3 episode table: unequal lengths and nonzero absolute start for episode 1."""
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/p2-terminal",
        root=tmp_path / "episodes",
        fps=20,
        features={
            OBS_STATE: {"dtype": "float32", "shape": (2,), "names": ["joint.pos", "gripper.pos"]},
            ACTION: {"dtype": "float32", "shape": (2,), "names": ["joint.pos", "gripper.pos"]},
            "observation.images.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
            TRUNCATED: {"dtype": "bool", "shape": (1,), "names": None},
        },
        use_videos=False,
    )
    meta.save_episode_tasks(["pick up the block"])
    for index, length in enumerate([5, 8]):
        meta.save_episode(index, length, ["pick up the block"], {}, {})
    meta.finalize()
    meta.ensure_readable()
    assert meta.episodes is not None
    assert meta.episodes[1]["dataset_from_index"] == 5
    assert meta.episodes[1]["dataset_to_index"] == 13
    return meta


@pytest.mark.parametrize(
    "episode_index,frame_index,truncated,current_done,expected_terminal",
    [
        pytest.param(0, 3, False, False, True, id="exact-end-no-padding"),
        pytest.param(0, 4, False, False, True, id="padded-terminal"),
        pytest.param(1, 6, False, False, True, id="nonzero-episode-offset-exact-end"),
        pytest.param(1, 3, False, True, False, id="same-anchor-longer-episode-current-done-ignored"),
        # Frozen MDS treats truncation as a terminal step too. A current-observation
        # flag cannot veto an independently known episode endpoint.
        pytest.param(0, 3, True, False, True, id="current-truncated-does-not-veto-end"),
        pytest.param(0, 4, True, False, True, id="padded-end-current-truncated-still-end"),
        pytest.param(0, 1, True, False, False, id="current-truncated-before-end-keeps-return"),
    ],
)
def test_processor_routes_mds_terminal_from_last_real_action_and_episode_endpoint(
    tmp_path: Path,
    episode_metadata: LeRobotDatasetMetadata,
    episode_index: int,
    frame_index: int,
    truncated: bool,
    current_done: bool,
    expected_terminal: bool,
) -> None:
    config = data_config(tmp_path, data_stats_payload())
    preprocessor, _ = make_perceptron_isaac_pre_post_processors(config, dataset_meta=episode_metadata)
    pack = next(
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    pack.dtype = "float32"
    fast = FastInputSpy()
    pack._fast_processor = fast
    assert episode_metadata.episodes is not None
    episode = episode_metadata.episodes[episode_index]
    start, end = episode["dataset_from_index"], episode["dataset_to_index"]
    indices, pad = delta_window(start + frame_index, [0, 1], start, end)
    last_real = max(index for index, padded in zip(indices, pad.tolist(), strict=True) if not padded)
    assert (last_real == end - 1) is expected_terminal
    batch = data_batch()
    batch.update(
        episode_index=torch.tensor([episode_index]),
        frame_index=torch.tensor([frame_index]),
        index=torch.tensor([start + frame_index]),
        action_is_pad=pad[None],
    )
    if bool(pad[1]):
        batch[ACTION][:, 1] = batch[ACTION][:, 0]  # Dataset endpoint replication, not a second real action.
    batch[DONE] = torch.tensor([current_done])
    batch[TRUNCATED] = torch.tensor([truncated])
    rendered = preprocessor(batch)
    stream = rendered[PERCEPTRON_ISAAC_STREAM_KEY]
    assert isinstance(stream, TensorStream)
    # Compare the actual processor output to its already-working leaf with trustworthy
    # terminal explicitly supplied. This isolates routing from rendering and tokenization.
    renderer = pack._ensure_renderer()
    reference = renderer.build_training(
        observation_window=[
            {"images": {"image": np.zeros((256, 256, 3), dtype=np.uint8)}, "proprio": np.array([0.5, 25.0])}
        ],
        prompt="pick up the block",
        action_chunk=batch[ACTION].numpy()[0],
        action_is_pad=pad.tolist(),
        fast_processor=FastInputSpy(),
        terminal=expected_terminal,
        device="cpu",
        dtype=torch.float32,
        temporal_patch_size=2,
        anchor_timestamp_seconds=frame_index / 20.0,
        fast_clip_normalized_max=1.0,
    )
    assert action_event(stream).tags == action_event(reference).tags
    assert action_event(stream, TextType.action).tags == action_event(reference, TextType.action).tags
    torch.testing.assert_close(
        action_event(stream, TextType.action).data, action_event(reference, TextType.action).data
    )
    assert action_event(stream).tags["action_is_pad"] == pad.tolist()
    assert rendered[PERCEPTRON_ISAAC_RENDER_META_KEY]["anchor_timestamp_seconds"] == [frame_index / 20.0]
    encoding = load_mharmony_encoding(
        renderer.metadata.mharmony_encoding_name,
        reserved_token_groups=renderer.metadata.mharmony_reserved_token_groups,
    )
    token = "<|end|>" if expected_terminal else "<|return|>"
    expected_id = encoding.encode(token, allowed_special={token})[0]
    assert expected_id == (248046 if expected_terminal else 248083)
    assert final_footer_id(reference) == expected_id, "Existing renderer is the positive leaf control."
    assert final_footer_id(stream) == expected_id, "Pack/data routing must supply actual chunk terminality."


def test_padding_without_authoritative_episode_metadata_keeps_return_footer(tmp_path: Path) -> None:
    pack, _ = data_pack(data_config(tmp_path, data_stats_payload()))
    batch = data_batch()
    batch["action_is_pad"] = torch.tensor([[False, True]])
    batch[ACTION][:, 1] = batch[ACTION][:, 0]
    batch[DONE] = torch.tensor([True])
    batch[TRUNCATED] = torch.tensor([True])
    out = pack(batch_to_transition(batch))
    stream = out[TransitionKey.COMPLEMENTARY_DATA][PERCEPTRON_ISAAC_STREAM_KEY]
    assert action_event(stream).tags["action_is_pad"] == [False, True]
    assert final_footer_id(stream) == 248083, "Padding/current observation done cannot establish terminality."


def test_saved_pipeline_rebinds_training_episode_endpoints(
    tmp_path: Path, episode_metadata: LeRobotDatasetMetadata
) -> None:
    config = data_config(tmp_path, data_stats_payload())
    preprocessor, postprocessor = make_perceptron_isaac_pre_post_processors(config)
    package = tmp_path / "saved_pipeline"
    package.mkdir()
    preprocessor.save_pretrained(package)
    postprocessor.save_pretrained(package)
    restored, _ = make_perceptron_isaac_pre_post_processors_from_pretrained(
        config, str(package), dataset_meta=episode_metadata
    )
    pack = next(step for step in restored.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep))
    pack.dtype = "float32"
    pack._fast_processor = FastInputSpy()
    batch = data_batch()
    rendered = restored(batch)
    stream = rendered[PERCEPTRON_ISAAC_STREAM_KEY]
    assert final_footer_id(stream) == 248046
    assert action_event(stream).tags["action_is_pad"] == [False, False]
    assert rendered[PERCEPTRON_ISAAC_RENDER_META_KEY]["anchor_timestamp_seconds"] == [0.15]
    # Endpoint knowledge is run-local; it must not become serialized policy state.
    assert "episode_metadata" not in pack.get_config()
    assert "_dataset_metadata" not in pack.get_config()


def test_terminal_routing_keeps_mixed_batch_suite_stats_and_frame_anchors(
    tmp_path: Path, episode_metadata: LeRobotDatasetMetadata
) -> None:
    first = data_stats_payload()
    second = data_stats_payload()
    for block in (second["action"], second["proprio"]):
        block["q01"] = [1.0, 100.0]
        block["q99"] = [2.0, 200.0]
    suites = tmp_path / "suites.json"
    suites.write_text(json.dumps({"libero_spatial": first, "libero_object": second}))
    mapping = tmp_path / "suite_by_task.json"
    mapping.write_text(json.dumps({"0": "libero_spatial", "1": "libero_object"}))
    config = data_config(tmp_path, first)
    config.suite_stats_path = str(suites)
    config.suite_by_task_index_path = str(mapping)
    preprocessor, _ = make_perceptron_isaac_pre_post_processors(config, dataset_meta=episode_metadata)
    pack = next(
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    pack.dtype = "float32"
    fast = FastInputSpy()
    pack._fast_processor = fast
    batch = data_batch()
    batch.update(
        {
            "observation.images.image": torch.zeros(2, 1, 3, 256, 256, dtype=torch.uint8),
            OBS_STATE: torch.tensor([[[0.5, 25.0]], [[1.5, 125.0]]]),
            ACTION: torch.tensor([[[0.5, 25.0], [0.75, 75.0]], [[1.5, 125.0], [1.75, 175.0]]]),
            "episode_index": torch.tensor([0, 1]),
            "frame_index": torch.tensor([3, 3]),
            "index": torch.tensor([3, 8]),
            "task_index": torch.tensor([0, 1]),
            "timestamp": torch.tensor([0.3, 0.3]),
            "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
            "task": ["pick up the block", "pick up the block"],
        }
    )
    rendered = preprocessor(batch)
    stream = rendered[PERCEPTRON_ISAAC_STREAM_KEY]
    assert rendered[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY] == [0, 1]
    assert rendered[PERCEPTRON_ISAAC_RENDER_META_KEY]["anchor_timestamp_seconds"] == [0.15, 0.15]
    assert len(stream.streams) == len(fast.inputs) == 2
    footers = []
    for index, payload in enumerate((first, second)):
        sample = TensorStream([stream.streams[index]])
        stats = isaac_stats_from_dict(payload)
        expected = normalize_isaac_actions(batch[ACTION].numpy()[index], stats.action)
        np.testing.assert_allclose(action_event(sample).tags["action_target"], expected)
        np.testing.assert_allclose(fast.inputs[index][0], expected)
        assert action_event(sample).tags["action_is_pad"] == [False, False]
        vector = next(event for event in sample.streams[0].events if event.type == VectorType.vector)
        np.testing.assert_allclose(vector.data.float().numpy().reshape(-1), expected[0])
        footers.append(final_footer_id(sample))
    assert footers == [248046, 248083], "Terminality is per sample, not per microbatch or task scope."
