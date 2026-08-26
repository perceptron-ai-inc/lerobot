#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract tests for DatasetWriter."""

from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import VideoEncoderConfig
from lerobot.datasets.dataset_writer import DatasetWriter, _encode_video_worker
from lerobot.datasets.io_utils import load_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.fixtures.dataset_factories import add_frames

SIMPLE_FEATURES = {
    "state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
}
VIDEO_KEY = "observation.images.cam"


def _video_features():
    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }


# ── Existing encode_video_worker tests ───────────────────────────────


def test_encode_video_worker_forwards_video_encoder(tmp_path):
    """_encode_video_worker forwards video_encoder to encode_video_frames."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(
            video_key,
            0,
            tmp_path,
            fps=30,
            video_encoder=VideoEncoderConfig(vcodec="h264", preset=None),
            encoder_threads=4,
        )

    assert captured_kwargs["video_encoder"].vcodec == "h264"
    assert captured_kwargs["encoder_threads"] == 4


def test_encode_video_worker_default_video_encoder(tmp_path):
    """_encode_video_worker passes None video_encoder which encode_video_frames defaults."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(video_key, 0, tmp_path, fps=30)

    assert captured_kwargs["video_encoder"] is None
    assert captured_kwargs["encoder_threads"] is None


# ── add_frame contracts ──────────────────────────────────────────────


def test_add_frame_increments_buffer_size(tmp_path, empty_lerobot_dataset_factory):
    """Each add_frame() call increases episode_buffer['size'] by 1."""
    dataset = empty_lerobot_dataset_factory(features=SIMPLE_FEATURES, root=tmp_path / "ds")
    assert dataset.writer.episode_buffer["size"] == 0

    add_frames(dataset, 1)
    assert dataset.writer.episode_buffer["size"] == 1

    add_frames(dataset, 1)
    assert dataset.writer.episode_buffer["size"] == 2


def test_add_frame_rejects_missing_feature(tmp_path, empty_lerobot_dataset_factory):
    """add_frame() raises ValueError when a required feature is missing."""
    dataset = empty_lerobot_dataset_factory(features=SIMPLE_FEATURES, root=tmp_path / "ds")
    with pytest.raises(ValueError, match="Missing features"):
        dataset.add_frame({"task": "Dummy task", "state": torch.randn(6)})
        # missing 'action'


# ── save_episode contracts ───────────────────────────────────────────


def test_save_episode_writes_parquet(tmp_path, empty_lerobot_dataset_factory):
    """After save_episode(), at least one .parquet file exists under data/."""
    dataset = empty_lerobot_dataset_factory(features=SIMPLE_FEATURES, root=tmp_path / "ds")
    add_frames(dataset, 3)
    dataset.save_episode()

    parquet_files = list((tmp_path / "ds" / "data").rglob("*.parquet"))
    assert len(parquet_files) > 0


def test_save_episode_updates_counters(tmp_path, empty_lerobot_dataset_factory):
    """After save_episode(), metadata counters are updated."""
    dataset = empty_lerobot_dataset_factory(features=SIMPLE_FEATURES, root=tmp_path / "ds")
    add_frames(dataset, 5)
    dataset.save_episode()

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 5


def test_save_episode_resets_buffer(tmp_path, empty_lerobot_dataset_factory):
    """After save_episode(), the episode buffer is reset."""
    dataset = empty_lerobot_dataset_factory(features=SIMPLE_FEATURES, root=tmp_path / "ds")
    add_frames(dataset, 3)
    dataset.save_episode()

    assert dataset.writer.episode_buffer["size"] == 0


def test_save_multiple_episodes(tmp_path, empty_lerobot_dataset_factory):
    """Recording 3 episodes results in correct total counts."""
    dataset = empty_lerobot_dataset_factory(features=SIMPLE_FEATURES, root=tmp_path / "ds")
    total_frames = 0
    for ep in range(3):
        n_frames = ep + 2  # 2, 3, 4
        add_frames(dataset, n_frames)
        dataset.save_episode()
        total_frames += n_frames

    assert dataset.meta.total_episodes == 3
    assert dataset.meta.total_frames == total_frames


# ── clear / lifecycle ────────────────────────────────────────────────


def test_clear_resets_buffer(tmp_path, empty_lerobot_dataset_factory):
    """clear_episode_buffer() resets the buffer size to 0."""
    dataset = empty_lerobot_dataset_factory(features=SIMPLE_FEATURES, root=tmp_path / "ds")
    add_frames(dataset, 1)
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.clear_episode_buffer()
    assert dataset.writer.episode_buffer["size"] == 0


def test_clear_removes_video_frame_staging_dir(tmp_path, empty_lerobot_dataset_factory):
    """clear_episode_buffer() removes PNG staging dirs for video features."""
    features = _video_features()
    dataset = empty_lerobot_dataset_factory(
        features=features,
        root=tmp_path / "ds",
        use_videos=True,
    )

    add_frames(dataset, 1)
    video_staging_dir = (
        dataset.root
        / Path(DEFAULT_IMAGE_PATH.format(image_key=VIDEO_KEY, episode_index=0, frame_index=0)).parent
    )
    assert video_staging_dir.is_dir()

    dataset.clear_episode_buffer()

    assert dataset.writer.episode_buffer["size"] == 0
    assert not video_staging_dir.exists()


def test_batched_encoding_staging_survives_save(tmp_path, empty_lerobot_dataset_factory):
    """The post-save clear must NOT delete video staging frames.

    With ``batch_encoding_size > 1`` the frames of already-saved episodes stay
    on disk until the batch encode runs; the encoder deletes them afterwards.
    A blanket switch of the post-save cleanup to ``camera_keys`` (as done in the
    discard path) would silently break batched encoding.
    """
    features = _video_features()
    dataset = empty_lerobot_dataset_factory(
        features=features,
        root=tmp_path / "ds",
        use_videos=True,
        batch_encoding_size=2,
    )
    add_frames(dataset, 3)

    staging_dir = dataset.writer._get_image_file_dir(0, VIDEO_KEY)
    assert staging_dir.is_dir()

    dataset.save_episode()  # first of a batch of 2: no encoding yet

    assert staging_dir.is_dir() and any(staging_dir.iterdir())


def test_resumed_batched_finalize_flushes_and_updates_new_episode_metadata(
    tmp_path, empty_lerobot_dataset_factory
):
    """Resumed batch encoding can index and update metadata rows buffered after resume."""
    features = _video_features()
    root = tmp_path / "resumed-batch"

    def fake_save_video(
        _writer,
        _video_key,
        episode_index,
        _temp_path=None,
        *,
        previous_episode=None,
    ):
        del previous_episode
        return {
            "episode_index": episode_index,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": episode_index,
            f"videos/{VIDEO_KEY}/from_timestamp": float(episode_index),
            f"videos/{VIDEO_KEY}/to_timestamp": float(episode_index + 1),
        }

    with patch.object(DatasetWriter, "_save_episode_video", autospec=True, side_effect=fake_save_video):
        initial = empty_lerobot_dataset_factory(
            features=features,
            root=root,
            use_videos=True,
            batch_encoding_size=3,
        )
        _record_episodes(initial, 1)
        initial.finalize()

        resumed = LeRobotDataset.resume(DUMMY_REPO_ID, root=root, batch_encoding_size=3)
        _record_episodes(resumed, 2)
        resumed.finalize()

    episodes = load_episodes(root)
    assert len(episodes) == 3
    assert [episodes[index]["episode_index"] for index in range(3)] == [0, 1, 2]
    assert [episodes[index][f"videos/{VIDEO_KEY}/file_index"] for index in range(3)] == [0, 1, 2]
    assert [episodes[index][f"videos/{VIDEO_KEY}/from_timestamp"] for index in range(3)] == [
        0.0,
        1.0,
        2.0,
    ]


def test_finalize_closes_data_parquet_before_pending_video_encoding(tmp_path, empty_lerobot_dataset_factory):
    """A video encoding failure must not leave saved frame data footerless."""
    features = _video_features()
    root = tmp_path / "failed-batch-finalize"
    dataset = empty_lerobot_dataset_factory(
        features=features,
        root=root,
        use_videos=True,
        batch_encoding_size=2,
    )
    add_frames(dataset, 1)
    dataset.save_episode()

    with (
        patch.object(dataset.writer, "_save_episode_video", side_effect=RuntimeError("encode failed")),
        pytest.raises(RuntimeError, match="encode failed"),
    ):
        dataset.finalize()

    assert dataset.writer._pq_writer is None
    data_path = root / dataset.meta.data_path.format(chunk_index=0, file_index=0)
    assert pq.read_table(data_path).num_rows == 1
    assert len(load_episodes(root)) == 1

    # Prevent the test cleanup finalizer from retrying the deliberately failed
    # video encode; the raw staging frames are removed with tmp_path.
    dataset.writer._episodes_since_last_encoding = 0
    dataset.finalize()


def test_finalize_is_idempotent(tmp_path, empty_lerobot_dataset_factory):
    """Calling finalize() twice does not raise."""
    dataset = empty_lerobot_dataset_factory(features=SIMPLE_FEATURES, root=tmp_path / "ds")
    add_frames(dataset, 3)
    dataset.save_episode()

    dataset.finalize()
    dataset.finalize()  # second call should not raise


def test_finalize_then_read_roundtrip(tmp_path, empty_lerobot_dataset_factory):
    """Write data, finalize, re-open, and verify data matches."""
    root = tmp_path / "roundtrip"
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = empty_lerobot_dataset_factory(features=features, root=root)

    # Record known values
    known_states = []
    for i in range(5):
        state = torch.tensor([float(i), float(i * 10)])
        known_states.append(state)
        dataset.add_frame({"task": "Test task", "state": state})
    dataset.save_episode()
    dataset.finalize()

    # Read back
    for i in range(5):
        item = dataset[i]
        assert torch.allclose(item["state"], known_states[i], atol=1e-5)


def test_batch_boundary_appends_to_previous_batch_video_when_under_budget(tmp_path, monkeypatch):
    """The first episode of a new encoding batch must respect the video size budget:
    with room left in the previous batch's file it appends instead of forcing a new
    file per batch (which produced one video per batch_encoding_size episodes)."""
    from types import SimpleNamespace

    import lerobot.datasets.dataset_writer as dataset_writer_module
    from lerobot.datasets.dataset_writer import DatasetWriter

    writer = DatasetWriter.__new__(DatasetWriter)
    writer._meta = SimpleNamespace(
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        video_files_size_in_mb=500.0,
        chunks_size=1000,
        latest_episode=None,
        episodes=None,
        video_keys=[VIDEO_KEY],
        depth_keys=[],
    )
    writer._root = tmp_path

    temp_dir = tmp_path / "tmp_ep"
    temp_dir.mkdir()
    ep_path = temp_dir / "ep.mp4"
    ep_path.write_bytes(b"episode")
    previous_path = tmp_path / writer._meta.video_path.format(
        video_key=VIDEO_KEY, chunk_index=0, file_index=0
    )
    previous_path.parent.mkdir(parents=True)
    previous_path.write_bytes(b"previous-batch")

    monkeypatch.setattr(dataset_writer_module, "get_file_size_in_mb", lambda path: 5.0)
    monkeypatch.setattr(dataset_writer_module, "get_video_duration_in_s", lambda path: 1.0)
    concat_calls = []
    monkeypatch.setattr(
        dataset_writer_module,
        "concatenate_video_files",
        lambda paths, out: concat_calls.append((list(paths), out)),
    )

    previous_episode = {
        "episode_index": 9,
        f"videos/{VIDEO_KEY}/chunk_index": 0,
        f"videos/{VIDEO_KEY}/file_index": 0,
        f"videos/{VIDEO_KEY}/to_timestamp": 10.0,
    }
    metadata = writer._save_episode_video(VIDEO_KEY, 10, temp_path=ep_path, previous_episode=previous_episode)

    assert concat_calls, "under-budget batch boundary must append, not roll a new file"
    assert metadata[f"videos/{VIDEO_KEY}/file_index"] == 0
    assert metadata[f"videos/{VIDEO_KEY}/from_timestamp"] == 10.0
    assert metadata[f"videos/{VIDEO_KEY}/to_timestamp"] == 11.0


def _install_stub_video_pipeline(monkeypatch, *, episode_size_mb: float):
    """Real _save_episode_video, stubbed codec boundary: encoding produces a tiny stub
    file, sizes/durations are fixed, concatenation just rewrites the target."""
    import tempfile as tempfile_module
    from pathlib import Path

    import lerobot.datasets.dataset_writer as dataset_writer_module
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_writer import DatasetWriter

    concat_calls: list[list] = []

    def fake_encode(self, video_key, episode_index):
        temp = Path(tempfile_module.mkdtemp(dir=self._root)) / f"{video_key}_{episode_index:03d}.mp4"
        temp.write_bytes(b"episode")
        return temp

    def fake_concat(paths, out):
        concat_calls.append(list(paths))
        Path(out).write_bytes(b"concatenated")

    monkeypatch.setattr(DatasetWriter, "_encode_temporary_episode_video", fake_encode)
    monkeypatch.setattr(dataset_writer_module, "get_file_size_in_mb", lambda path: episode_size_mb)
    monkeypatch.setattr(dataset_writer_module, "get_video_duration_in_s", lambda path: 1.0)
    monkeypatch.setattr(dataset_writer_module, "concatenate_video_files", fake_concat)
    monkeypatch.setattr(LeRobotDatasetMetadata, "update_video_info", lambda self, *args, **kwargs: None)
    return concat_calls


def _record_episodes(dataset, count):
    for _ in range(count):
        add_frames(dataset, 1)
        dataset.save_episode()


@pytest.mark.parametrize(
    ("episode_size_mb", "expected_file_indices", "expected_concat_count"),
    [
        pytest.param(5.0, [0, 0, 0, 0], 3, id="append-under-budget"),
        pytest.param(300.0, [0, 1, 2, 3], 0, id="roll-over-budget"),
    ],
)
def test_batch_boundaries_through_the_real_video_save(
    tmp_path,
    monkeypatch,
    empty_lerobot_dataset_factory,
    episode_size_mb,
    expected_file_indices,
    expected_concat_count,
):
    """Encoding batches append or roll according to the shared video size budget."""
    features = _video_features()
    concat_calls = _install_stub_video_pipeline(monkeypatch, episode_size_mb=episode_size_mb)

    dataset = empty_lerobot_dataset_factory(
        features=features,
        root=tmp_path / "batched",
        use_videos=True,
        batch_encoding_size=2,
    )
    _record_episodes(dataset, 4)
    dataset.finalize()

    episodes = load_episodes(tmp_path / "batched")
    file_indices = [episodes[index][f"videos/{VIDEO_KEY}/file_index"] for index in range(4)]
    assert file_indices == expected_file_indices
    assert len(concat_calls) == expected_concat_count


def test_resume_with_unmaterialized_prior_video_rolls_instead_of_crashing(
    tmp_path, monkeypatch, empty_lerobot_dataset_factory
):
    """Metadata-only resume (Hub-finalized dataset, videos never pulled locally): the
    batch-boundary episode must roll a new file from the metadata indices instead of
    stat-crashing on the missing prior-session file."""
    features = _video_features()
    _install_stub_video_pipeline(monkeypatch, episode_size_mb=5.0)
    root = tmp_path / "resumed-missing-video"

    initial = empty_lerobot_dataset_factory(
        features=features,
        root=root,
        use_videos=True,
        batch_encoding_size=2,
    )
    _record_episodes(initial, 2)
    initial.finalize()

    # Simulate a fresh machine that pulled metadata only.
    for video_file in (root / "videos").rglob("*.mp4"):
        video_file.unlink()

    resumed = LeRobotDataset.resume(DUMMY_REPO_ID, root=root, batch_encoding_size=2)
    _record_episodes(resumed, 2)
    resumed.finalize()

    episodes = load_episodes(root)
    assert len(episodes) == 4
    first_session = episodes[1][f"videos/{VIDEO_KEY}/file_index"]
    boundary = episodes[2][f"videos/{VIDEO_KEY}/file_index"]
    assert boundary == first_session + 1  # rolled forward, no FileNotFoundError
    assert episodes[3][f"videos/{VIDEO_KEY}/file_index"] == boundary  # then appended locally
