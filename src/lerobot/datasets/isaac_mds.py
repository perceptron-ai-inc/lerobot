"""Strict conversion of Genesis ISAAC trajectory MDS shards to LeRobotDataset."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import struct
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MDS_COLUMN_NAMES = (
    "content",
    "metadata",
    "original_sample_index",
    "references",
    "schema_version",
    "tools",
    "type",
)
MDS_COLUMN_SIZES = (None, None, 8, None, None, None, None)
MDS_COLUMN_ENCODINGS = ("json", "json", "int", "json", "json", "json", "json")
ISAAC_YAM_CAMERAS = ("top", "left", "right")
ISAAC_YAM_POSITION_NAMES = tuple(
    name
    for side in ("left", "right")
    for name in (*(f"{side}_joint_{index}.pos" for index in range(6)), f"{side}_gripper.pos")
)
CONVERSION_MANIFEST = "isaac_mds_conversion.json"
QUANTILE_BINS = 5000


class IsaacMdsConversionError(ValueError):
    """Raised when an MDS shard violates the authenticated ISAAC data contract."""


class QuantileGridTrace:
    """Record the compact linear grids used by v0.6 running quantiles."""

    def __init__(self, width: int, *, num_bins: int = QUANTILE_BINS):
        self.width = width
        self.num_bins = num_bins
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self.grids: list[dict[str, Any]] = []

    def update(self, batch: np.ndarray) -> None:
        values = np.asarray(batch).reshape(-1, self.width)
        new_min = np.min(values, axis=0)
        new_max = np.max(values, axis=0)
        if self._min is None:
            self._min = new_min.copy()
            self._max = new_max.copy()
            padding = np.full(self.width, 1e-10)
        else:
            updated_min = np.minimum(self._min, new_min)
            updated_max = np.maximum(self._max, new_max)
            if np.array_equal(updated_min, self._min) and np.array_equal(updated_max, self._max):
                return
            self._min = updated_min
            self._max = updated_max
            padding = (self._max - self._min) * 1e-10
        start = self._min - padding
        end = self._max + padding
        self.grids.append(
            {
                "update": len(self.grids),
                "start": start.tolist(),
                "end": end.tolist(),
                "bin_width": ((end - start) / self.num_bins).tolist(),
            }
        )

    @property
    def error_bound(self) -> np.ndarray:
        if not self.grids:
            raise IsaacMdsConversionError("Quantile grid trace has no updates.")
        return np.sum(
            np.asarray([grid["bin_width"] for grid in self.grids], dtype=np.float64),
            axis=0,
        )

    def report(self) -> dict[str, Any]:
        return {
            "algorithm": "lerobot.RunningQuantileStats",
            "num_bins": self.num_bins,
            "grids": self.grids,
            "error_bound": self.error_bound.tolist(),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mds_sample_count(path: str | Path) -> int:
    path = Path(path)
    with path.open("rb", buffering=0) as stream:
        raw = stream.read(4)
    if len(raw) != 4:
        raise IsaacMdsConversionError(f"Truncated MDS shard header: {path}.")
    return struct.unpack("<I", raw)[0]


def read_mds_v2_sample(path: str | Path, sample_index: int) -> dict[str, Any]:
    """Read one row from the exact uncompressed Genesis JSON MDS v2 schema."""
    path = Path(path)
    count = mds_sample_count(path)
    if sample_index < 0 or sample_index >= count:
        raise IndexError(f"MDS sample index {sample_index} is outside [0, {count}).")
    with path.open("rb", buffering=0) as stream:
        stream.seek((sample_index + 1) * 4)
        pair = stream.read(8)
        if len(pair) != 8:
            raise IsaacMdsConversionError(f"Truncated MDS offset table in {path}.")
        begin, end = struct.unpack("<II", pair)
        if not 0 < begin < end <= path.stat().st_size:
            raise IsaacMdsConversionError(f"Invalid MDS offsets for {path}[{sample_index}]: {(begin, end)}.")
        stream.seek(begin)
        data = stream.read(end - begin)

    cursor = 0
    sizes: list[int] = []
    for fixed_size in MDS_COLUMN_SIZES:
        if fixed_size is None:
            if cursor + 4 > len(data):
                raise IsaacMdsConversionError("Truncated MDS variable-column header.")
            (size,) = struct.unpack_from("<I", data, cursor)
            cursor += 4
            sizes.append(size)
        else:
            sizes.append(fixed_size)

    sample: dict[str, Any] = {}
    for name, encoding, size in zip(
        MDS_COLUMN_NAMES,
        MDS_COLUMN_ENCODINGS,
        sizes,
        strict=True,
    ):
        blob = data[cursor : cursor + size]
        if len(blob) != size:
            raise IsaacMdsConversionError(f"Truncated MDS column {name!r}.")
        cursor += size
        try:
            sample[name] = (
                int.from_bytes(blob, byteorder="little", signed=True)
                if encoding == "int"
                else json.loads(blob)
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IsaacMdsConversionError(f"Invalid JSON in MDS column {name!r}.") from exc
    if cursor != len(data):
        raise IsaacMdsConversionError(f"MDS row has {len(data) - cursor} trailing bytes.")
    return sample


def validate_mds_index_schema(index_path: str | Path) -> None:
    """Validate an index without importing Mosaic Streaming or Genesis."""
    raw = json.loads(Path(index_path).read_text())
    if raw.get("version") != 2 or not isinstance(raw.get("shards"), list):
        raise IsaacMdsConversionError("ISAAC conversion requires an MDS v2 index with shards.")
    for index, shard in enumerate(raw["shards"]):
        actual = (
            tuple(shard.get("column_names") or ()),
            tuple(shard.get("column_sizes") or ()),
            tuple(shard.get("column_encodings") or ()),
            shard.get("compression"),
            shard.get("format"),
            shard.get("version"),
        )
        expected = (
            MDS_COLUMN_NAMES,
            MDS_COLUMN_SIZES,
            MDS_COLUMN_ENCODINGS,
            None,
            "mds",
            2,
        )
        if actual != expected:
            raise IsaacMdsConversionError(f"MDS index shard {index} has an unsupported schema.")


def _vector_payload(payload: Any, *, label: str, width: int = 14) -> np.ndarray:
    if not isinstance(payload, Mapping) or payload.get("dtype") != "float32":
        raise IsaacMdsConversionError(f"{label} must be a float32 tensor payload.")
    if payload.get("shape") != [width]:
        raise IsaacMdsConversionError(f"{label} must have shape [{width}].")
    values = np.asarray(payload.get("values"), dtype=np.float32)
    if values.shape != (width,) or not np.isfinite(values).all():
        raise IsaacMdsConversionError(f"{label} values must be finite with shape ({width},).")
    return values


def _step_vector(step: Mapping[str, Any], key: str) -> np.ndarray:
    if key == "action":
        wrapper = step.get("action")
    else:
        observation = step.get("observation")
        wrapper = observation.get(key) if isinstance(observation, Mapping) else None
    if not isinstance(wrapper, Mapping) or "data" not in wrapper:
        raise IsaacMdsConversionError(f"Trajectory step is missing {key!r} tensor data.")
    return _vector_payload(wrapper["data"], label=key)


def _stats_block(trajectory: Mapping[str, Any], key: str) -> dict[str, list[float]]:
    ingestion = (trajectory.get("metadata") or {}).get("ingestion") or {}
    block = (ingestion.get("stats") or {}).get(key)
    if not isinstance(block, Mapping):
        raise IsaacMdsConversionError(f"Trajectory is missing ingestion stats for {key!r}.")
    result: dict[str, list[float]] = {}
    for stat in ("min", "max", "q01", "q99"):
        values = np.asarray(block.get(stat), dtype=np.float32)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise IsaacMdsConversionError(f"Trajectory {key}.{stat} must be a finite 14-vector.")
        result[stat] = values.tolist()
    return result


@dataclass(frozen=True)
class IsaacMdsEpisode:
    dataset: str
    episode_index: int
    sample_index: int
    task: str
    steps: tuple[Mapping[str, Any], ...]
    references: tuple[Mapping[str, Any], ...]
    camera_reference_indices: dict[str, int]
    source_stats: dict[str, dict[str, list[float]]]

    @classmethod
    def from_sample(cls, sample: Mapping[str, Any], *, sample_index: int) -> IsaacMdsEpisode:
        content = sample.get("content")
        if sample.get("type") != "Document" or not isinstance(content, list) or len(content) != 1:
            raise IsaacMdsConversionError("ISAAC MDS row must be a single-trajectory Document.")
        trajectory = content[0]
        if not isinstance(trajectory, Mapping):
            raise IsaacMdsConversionError("ISAAC trajectory content must be an object.")
        steps = trajectory.get("steps")
        references = trajectory.get("references")
        if not isinstance(steps, list) or len(steps) < 2:
            raise IsaacMdsConversionError("ISAAC trajectory must contain at least two steps.")
        if not isinstance(references, list):
            raise IsaacMdsConversionError("ISAAC trajectory must contain media references.")
        task = trajectory.get("task")
        if not isinstance(task, str) or not task.strip():
            raise IsaacMdsConversionError("ISAAC trajectory task must be a non-empty string.")

        camera_indices: dict[str, int] = {}
        for step_index, step in enumerate(steps):
            observation = step.get("observation") if isinstance(step, Mapping) else None
            if not isinstance(observation, Mapping):
                raise IsaacMdsConversionError(f"Step {step_index} has no observation object.")
            for camera in ISAAC_YAM_CAMERAS:
                ref = observation.get(f"image.{camera}")
                if not isinstance(ref, Mapping) or ref.get("type") != "reference":
                    raise IsaacMdsConversionError(
                        f"Step {step_index} is missing image.{camera} video reference."
                    )
                ref_index = ref.get("index")
                if not isinstance(ref_index, int) or isinstance(ref_index, bool):
                    raise IsaacMdsConversionError(f"image.{camera} reference index must be an integer.")
                previous = camera_indices.setdefault(camera, ref_index)
                if previous != ref_index:
                    raise IsaacMdsConversionError(
                        f"image.{camera} changes media reference within one episode."
                    )
                if not 0 <= ref_index < len(references):
                    raise IsaacMdsConversionError(f"image.{camera} reference is outside the table.")
            _step_vector(step, "state")
            _step_vector(step, "action")

        metadata = sample.get("metadata") or {}
        dataset = str(metadata.get("dataset") or "")
        episode_index = int(trajectory.get("episode_index", sample.get("original_sample_index", -1)))
        try:
            proprio_stats = _stats_block(trajectory, "proprio")
        except IsaacMdsConversionError:
            proprio_stats = _stats_block(trajectory, "state")
        return cls(
            dataset=dataset,
            episode_index=episode_index,
            sample_index=sample_index,
            task=task.strip(),
            steps=tuple(steps),
            references=tuple(references),
            camera_reference_indices=camera_indices,
            source_stats={"action": _stats_block(trajectory, "action"), "proprio": proprio_stats},
        )

    @cached_property
    def timestamps(self) -> np.ndarray:
        values = np.asarray([step.get("timestamp_seconds") for step in self.steps], dtype=np.float64)
        if values.shape != (len(self.steps),) or not np.isfinite(values).all():
            raise IsaacMdsConversionError("ISAAC timestamps must be finite scalars.")
        if bool((np.diff(values) <= 0.0).any()):
            raise IsaacMdsConversionError("ISAAC timestamps must be strictly increasing.")
        return values


@contextmanager
def decode_episode_video_frames(episode: IsaacMdsEpisode) -> Iterator[Iterator[dict[str, np.ndarray]]]:
    """Decode all three reference videos in lockstep without retaining an episode in RAM."""
    import av

    containers = []
    decoders: dict[str, Iterator[Any]] = {}
    try:
        for camera, reference_index in episode.camera_reference_indices.items():
            reference = episode.references[reference_index]
            encoded = reference.get("content") if isinstance(reference, Mapping) else None
            if not isinstance(encoded, str):
                raise IsaacMdsConversionError(f"Reference for image.{camera} has no base64 content.")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise IsaacMdsConversionError(f"Reference for image.{camera} is not valid base64.") from exc
            container = av.open(io.BytesIO(payload), mode="r")
            containers.append(container)
            decoders[camera] = iter(container.decode(video=0))

        def frames() -> Iterator[dict[str, np.ndarray]]:
            for frame_index in range(len(episode.steps)):
                decoded: dict[str, np.ndarray] = {}
                for camera in ISAAC_YAM_CAMERAS:
                    try:
                        frame = next(decoders[camera])
                    except StopIteration as exc:
                        raise IsaacMdsConversionError(
                            f"image.{camera} video ended before frame {frame_index}."
                        ) from exc
                    decoded[camera] = frame.to_ndarray(format="rgb24")
                yield decoded
            # A video with more frames than trajectory steps is a frame-rate or re-encode
            # mismatch: the images would cover only part of the episode while states and
            # actions cover all of it, producing a self-consistent but temporally misaligned
            # dataset that nothing downstream can detect.
            for camera in ISAAC_YAM_CAMERAS:
                surplus = sum(1 for _ in decoders[camera])
                if surplus:
                    raise IsaacMdsConversionError(
                        f"image.{camera} video has {len(episode.steps) + surplus} frames but the "
                        f"trajectory has {len(episode.steps)} steps; images and actions would be "
                        "temporally misaligned."
                    )

        yield frames()
    finally:
        for container in containers:
            container.close()


def isaac_mds_features(
    camera_frames: Mapping[str, np.ndarray], *, use_videos: bool = True
) -> dict[str, dict[str, Any]]:
    if tuple(camera_frames) != ISAAC_YAM_CAMERAS:
        raise IsaacMdsConversionError(
            f"Camera order must be exactly {ISAAC_YAM_CAMERAS}, got {tuple(camera_frames)}."
        )
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": list(ISAAC_YAM_POSITION_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": list(ISAAC_YAM_POSITION_NAMES),
        },
        "source.timestamp_seconds": {"dtype": "float64", "shape": (1,), "names": None},
        "source.episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "source.sample_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for camera, frame in camera_frames.items():
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise IsaacMdsConversionError(
                f"image.{camera} must decode to uint8 HWC RGB, got {array.shape}/{array.dtype}."
            )
        features[f"observation.images.{camera}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": tuple(array.shape),
            "names": ["height", "width", "channel"],
        }
    return features


def _validate_episode_timing(episode: IsaacMdsEpisode, fps: int, *, tolerance_s: float) -> None:
    relative = episode.timestamps - episode.timestamps[0]
    expected = np.arange(len(relative), dtype=np.float64) / float(fps)
    max_error = float(np.max(np.abs(relative - expected)))
    if max_error > tolerance_s:
        raise IsaacMdsConversionError(
            f"Episode timing does not match {fps} FPS: max error {max_error:.6g}s > {tolerance_s:.6g}s."
        )


def write_isaac_mds_episode(
    dataset: LeRobotDataset,
    episode: IsaacMdsEpisode,
    decoded_frames: Iterable[Mapping[str, np.ndarray]],
    *,
    fps: int,
    timestamp_tolerance_s: float = 1e-3,
    quantile_traces: Mapping[str, QuantileGridTrace] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Write one episode while proving raw state/action ordering and model timing."""
    _validate_episode_timing(episode, fps, tolerance_s=timestamp_tolerance_s)
    state_rows = []
    action_rows = []
    frame_count = 0
    timestamps = episode.timestamps
    for frame_index, (step, images) in enumerate(zip(episode.steps, decoded_frames, strict=True)):
        if tuple(images) != ISAAC_YAM_CAMERAS:
            raise IsaacMdsConversionError("Decoded camera order changed within an episode.")
        state = _step_vector(step, "state")
        action = _step_vector(step, "action")
        frame = {
            "observation.state": state,
            "action": action,
            "source.timestamp_seconds": np.asarray([timestamps[frame_index]], dtype=np.float64),
            "source.episode_index": np.asarray([episode.episode_index], dtype=np.int64),
            "source.sample_index": np.asarray([episode.sample_index], dtype=np.int64),
            "task": episode.task,
            **{f"observation.images.{camera}": np.asarray(images[camera]) for camera in ISAAC_YAM_CAMERAS},
        }
        dataset.add_frame(frame)
        state_rows.append(state)
        action_rows.append(action)
        frame_count += 1
    if frame_count != len(episode.steps):
        raise IsaacMdsConversionError(
            f"Decoded frame count {frame_count} != trajectory steps {len(episode.steps)}."
        )
    arrays = {
        "observation.state": np.stack(state_rows),
        "action": np.stack(action_rows),
    }
    if quantile_traces is not None:
        for feature, values in arrays.items():
            quantile_traces[feature].update(values)
    dataset.save_episode(parallel_encoding=False)
    return compute_episode_stats(
        arrays,
        {
            "observation.state": dataset.meta.features["observation.state"],
            "action": dataset.meta.features["action"],
        },
    )


def verify_converted_stats(
    actual: Mapping[str, Mapping[str, np.ndarray]],
    expected: Mapping[str, Mapping[str, np.ndarray]],
    *,
    quantile_traces: Mapping[str, QuantileGridTrace] | None = None,
    scalar_atol: float = 1e-6,
    scalar_rtol: float = 1e-6,
) -> dict[str, Any]:
    """Compare output stats with source-frame stats under explicit error budgets."""
    report: dict[str, Any] = {"passed": True, "features": {}}
    for feature in ("observation.state", "action"):
        feature_report = {}
        for statistic in ("count", "min", "max", "mean", "std", "q01", "q99"):
            left = np.asarray(actual[feature][statistic])
            right = np.asarray(expected[feature][statistic])
            if left.shape != right.shape:
                feature_report[statistic] = {
                    "passed": False,
                    "actual_shape": list(left.shape),
                    "expected_shape": list(right.shape),
                }
                report["passed"] = False
                continue
            is_quantile = statistic.startswith("q")
            absolute_error = np.abs(left - right)
            if is_quantile and quantile_traces is not None:
                trace = quantile_traces[feature]
                error_budget = 2 * trace.error_bound + 1e-6
                passed = bool(np.all(absolute_error <= error_budget))
                feature_report[statistic] = {
                    "passed": passed,
                    "absolute_error": absolute_error.tolist(),
                    "error_budget": error_budget.tolist(),
                    # Both traces are fed the same converted arrays, so this is a
                    # writer-round-trip self-check, NOT a comparison against the Genesis
                    # ingestion statistics (those are only hashed into source_stats_profiles).
                    # Name the terms for what they are rather than implying an independent
                    # source-side error term.
                    "scope": "converted_array_round_trip",
                    "compares": "dataset.meta.stats vs aggregate_stats(episode_stats)",
                    "formula": "2 * E_grid + 1e-6",
                }
            else:
                atol = 0.0 if statistic == "count" else scalar_atol
                rtol = 0.0 if statistic == "count" else scalar_rtol
                passed = bool(np.allclose(left, right, atol=atol, rtol=rtol))
                feature_report[statistic] = {
                    "passed": passed,
                    "max_abs_error": float(np.max(absolute_error)),
                    "atol": atol,
                    "rtol": rtol,
                }
            report["passed"] = bool(report["passed"] and passed)
        report["features"][feature] = feature_report
    if quantile_traces is not None:
        report["quantile_grid_traces"] = {
            feature: {implementation: trace.report() for implementation in ("source", "lerobot")}
            for feature, trace in quantile_traces.items()
        }
    if not report["passed"]:
        raise IsaacMdsConversionError("Converted LeRobotDataset stats exceed the declared tolerance.")
    return report


def convert_isaac_mds_to_lerobot(
    shards: Sequence[str | Path],
    output_root: str | Path,
    *,
    repo_id: str,
    fps: int = 30,
    max_episodes: int | None = None,
    use_videos: bool = True,
    index_path: str | Path | None = None,
) -> Path:
    """Atomically convert local Genesis ISAAC MDS shards into a v0.6 dataset."""
    shard_paths = [Path(path).resolve() for path in shards]
    if not shard_paths or any(not path.is_file() for path in shard_paths):
        raise IsaacMdsConversionError("Every ISAAC MDS shard must be an existing local file.")
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise IsaacMdsConversionError("ISAAC MDS conversion fps must be a positive integer.")
    if max_episodes is not None and max_episodes <= 0:
        raise IsaacMdsConversionError("max_episodes must be positive when set.")
    if index_path is not None:
        validate_mds_index_schema(index_path)

    output = Path(output_root).resolve()
    if output.exists():
        raise IsaacMdsConversionError(f"Conversion output already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.convert-", dir=output.parent))
    temporary = staging / "dataset"
    dataset: LeRobotDataset | None = None
    episode_stats = []
    source_profiles: dict[str, dict[str, Any]] = {}
    converted_episodes = 0
    converted_frames = 0
    quantile_traces = {feature: QuantileGridTrace(14) for feature in ("observation.state", "action")}
    try:
        for shard_path in shard_paths:
            for sample_index in range(mds_sample_count(shard_path)):
                if max_episodes is not None and converted_episodes >= max_episodes:
                    break
                sample = read_mds_v2_sample(shard_path, sample_index)
                episode = IsaacMdsEpisode.from_sample(sample, sample_index=sample_index)
                with decode_episode_video_frames(episode) as decoded:
                    decoded_iterator = iter(decoded)
                    try:
                        first_frame = next(decoded_iterator)
                    except StopIteration as exc:
                        raise IsaacMdsConversionError("ISAAC episode decoded no video frames.") from exc
                    if dataset is None:
                        dataset = LeRobotDataset.create(
                            repo_id=repo_id,
                            fps=fps,
                            features=isaac_mds_features(first_frame, use_videos=use_videos),
                            root=temporary,
                            robot_type="bi_yam_follower",
                            use_videos=use_videos,
                        )
                    current_shapes = {
                        camera: tuple(np.asarray(frame).shape) for camera, frame in first_frame.items()
                    }
                    expected_shapes = {
                        camera: tuple(dataset.meta.features[f"observation.images.{camera}"]["shape"])
                        for camera in ISAAC_YAM_CAMERAS
                    }
                    if current_shapes != expected_shapes:
                        raise IsaacMdsConversionError(
                            f"Camera geometry changed: {current_shapes} != {expected_shapes}."
                        )
                    stats = write_isaac_mds_episode(
                        dataset,
                        episode,
                        chain((first_frame,), decoded_iterator),
                        fps=fps,
                        quantile_traces=quantile_traces,
                    )
                episode_stats.append(stats)
                profile_json = json.dumps(episode.source_stats, sort_keys=True, separators=(",", ":"))
                profile_sha = hashlib.sha256(profile_json.encode()).hexdigest()
                source_profiles.setdefault(
                    profile_sha,
                    {"sha256": profile_sha, "stats": episode.source_stats, "episodes": []},
                )["episodes"].append(
                    {
                        "dataset": episode.dataset,
                        "episode_index": episode.episode_index,
                        "sample_index": episode.sample_index,
                    }
                )
                converted_episodes += 1
                converted_frames += len(episode.steps)
            if max_episodes is not None and converted_episodes >= max_episodes:
                break
        if dataset is None or converted_episodes == 0:
            raise IsaacMdsConversionError("No ISAAC episodes were converted.")

        expected_stats = aggregate_stats(episode_stats)
        stats_report = verify_converted_stats(
            dataset.meta.stats,
            expected_stats,
            quantile_traces=quantile_traces,
        )
        dataset.finalize()
        manifest = {
            "schema": "perceptron_isaac_mds_conversion_v1",
            "repo_id": repo_id,
            "fps": fps,
            "robot_type": "bi_yam_follower",
            "camera_order": list(ISAAC_YAM_CAMERAS),
            "state_action_names": list(ISAAC_YAM_POSITION_NAMES),
            "episodes": converted_episodes,
            "frames": converted_frames,
            "source_shards": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in shard_paths
            ],
            "source_stats_profiles": list(source_profiles.values()),
            "stats_verification": stats_report,
            "normalization_note": (
                "LeRobot dataset quantiles describe converted frames. ISAAC model normalization "
                "continues to use checkpoint-owned flow_matching_stats_v1 profiles."
            ),
        }
        (temporary / CONVERSION_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
        staging.rmdir()
        return output
    except BaseException:
        if dataset is not None:
            dataset.finalize()
        shutil.rmtree(staging, ignore_errors=True)
        raise
