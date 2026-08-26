from __future__ import annotations

import json
import struct
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.isaac_mds import (  # noqa: E402
    CONVERSION_MANIFEST,
    ISAAC_YAM_CAMERAS,
    ISAAC_YAM_POSITION_NAMES,
    IsaacMdsConversionError,
    IsaacMdsEpisode,
    convert_isaac_mds_to_lerobot,
    read_mds_v2_sample,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def _tensor(values: np.ndarray) -> dict:
    return {
        "type": "tensor",
        "data": {
            "dtype": "float32",
            "shape": [14],
            "values": np.asarray(values, dtype=np.float32).tolist(),
        },
    }


def _stats() -> dict:
    return {
        key: {
            statistic: (np.arange(14, dtype=np.float32) + offset).tolist()
            for statistic, offset in (("min", -2.0), ("max", 2.0), ("q01", -1.0), ("q99", 1.0))
        }
        for key in ("action", "proprio")
    }


def _sample(num_steps: int = 5) -> dict:
    steps = []
    for index in range(num_steps):
        observation = {
            "state": _tensor(np.arange(14, dtype=np.float32) + index),
            **{
                f"image.{camera}": {"type": "reference", "index": camera_index}
                for camera_index, camera in enumerate(ISAAC_YAM_CAMERAS)
            },
        }
        steps.append(
            {
                "timestamp_seconds": 10.0 + index / 30.0,
                "observation": observation,
                "action": _tensor(np.arange(14, dtype=np.float32) + 100 + index),
            }
        )
    return {
        "content": [
            {
                "episode_index": 17,
                "task": "fold the cloth",
                "steps": steps,
                "references": [{"content": "unused"} for _ in ISAAC_YAM_CAMERAS],
                "metadata": {"ingestion": {"stats": _stats()}},
            }
        ],
        "metadata": {"dataset": "synthetic-yam", "episode_index": 17},
        "original_sample_index": 23,
        "references": [],
        "schema_version": 1,
        "tools": [],
        "type": "Document",
    }


def _write_mds(path, samples: list[dict]) -> None:
    rows = []
    for sample in samples:
        columns = [
            json.dumps(sample["content"]).encode(),
            json.dumps(sample["metadata"]).encode(),
            int(sample["original_sample_index"]).to_bytes(8, "little", signed=True),
            json.dumps(sample["references"]).encode(),
            json.dumps(sample["schema_version"]).encode(),
            json.dumps(sample["tools"]).encode(),
            json.dumps(sample["type"]).encode(),
        ]
        variable_sizes = [len(value) for index, value in enumerate(columns) if index != 2]
        rows.append(struct.pack(f"<{len(variable_sizes)}I", *variable_sizes) + b"".join(columns))
    header_size = 4 * (len(rows) + 2)
    offsets = [header_size]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    path.write_bytes(struct.pack(f"<{len(offsets) + 1}I", len(rows), *offsets) + b"".join(rows))


def test_read_mds_v2_sample_round_trip(tmp_path):
    shard = tmp_path / "shard.00000.mds"
    expected = _sample()
    _write_mds(shard, [expected])

    actual = read_mds_v2_sample(shard, 0)

    assert actual == expected
    with pytest.raises(IndexError):
        read_mds_v2_sample(shard, 1)


def test_convert_isaac_mds_preserves_windows_padding_order_and_stats(tmp_path, monkeypatch):
    import lerobot.datasets.isaac_mds as isaac_mds

    shard = tmp_path / "shard.00000.mds"
    _write_mds(shard, [_sample()])
    frames = [
        {
            camera: np.full((8, 10, 3), frame_index + camera_index, dtype=np.uint8)
            for camera_index, camera in enumerate(ISAAC_YAM_CAMERAS)
        }
        for frame_index in range(5)
    ]

    @contextmanager
    def fake_decode(episode: IsaacMdsEpisode):
        assert len(episode.steps) == 5
        yield iter(frames)

    monkeypatch.setattr(isaac_mds, "decode_episode_video_frames", fake_decode)
    output = convert_isaac_mds_to_lerobot(
        [shard],
        tmp_path / "converted",
        repo_id="test/isaac_yam_mds",
        fps=30,
        use_videos=False,
    )

    manifest = json.loads((output / CONVERSION_MANIFEST).read_text())
    assert manifest["episodes"] == 1
    assert manifest["frames"] == 5
    assert manifest["camera_order"] == list(ISAAC_YAM_CAMERAS)
    assert manifest["state_action_names"] == list(ISAAC_YAM_POSITION_NAMES)
    assert manifest["stats_verification"]["passed"] is True
    quantile_report = manifest["stats_verification"]["features"]["action"]["q01"]
    # Both traces are fed the same converted arrays, so this is a writer round-trip check
    # rather than a comparison against the Genesis ingestion statistics.
    assert quantile_report["formula"] == "2 * E_grid + 1e-6"
    assert quantile_report["scope"] == "converted_array_round_trip"
    assert len(quantile_report["error_budget"]) == 14
    assert manifest["stats_verification"]["features"]["action"]["count"]["atol"] == 0.0
    grid_report = manifest["stats_verification"]["quantile_grid_traces"]["action"]
    assert grid_report["source"]["num_bins"] == 5000
    assert grid_report["lerobot"]["grids"]
    assert grid_report["source"] == grid_report["lerobot"]
    assert len(manifest["source_stats_profiles"]) == 1

    dataset = LeRobotDataset(
        "test/isaac_yam_mds",
        root=output,
        delta_timestamps={
            "observation.state": [-2 / 30, -1 / 30, 0.0],
            "action": [index / 30 for index in range(30)],
        },
    )
    first = dataset[0]
    assert dataset.meta.features["action"]["names"] == list(ISAAC_YAM_POSITION_NAMES)
    assert first["task"] == "fold the cloth"
    assert first["observation.state_is_pad"].tolist() == [True, True, False]
    assert first["action_is_pad"].tolist() == [False] * 5 + [True] * 25
    np.testing.assert_allclose(first["observation.state"][-1], np.arange(14, dtype=np.float32))
    np.testing.assert_allclose(
        first["action"][0],
        np.arange(14, dtype=np.float32) + 100,
    )
    assert float(first["source.timestamp_seconds"].item()) == pytest.approx(10.0)


def test_isaac_mds_episode_validates_and_caches_timestamps():
    episode = IsaacMdsEpisode.from_sample(_sample(), sample_index=0)
    assert episode.timestamps is episode.timestamps

    wrong = _sample()
    wrong["content"][0]["steps"][1]["timestamp_seconds"] += 0.01
    episode = IsaacMdsEpisode.from_sample(wrong, sample_index=0)

    from lerobot.datasets.isaac_mds import _validate_episode_timing

    with pytest.raises(IsaacMdsConversionError, match="does not match 30 FPS"):
        _validate_episode_timing(episode, 30, tolerance_s=1e-3)
