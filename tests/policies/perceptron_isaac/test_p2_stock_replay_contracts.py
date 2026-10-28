"""Review R1 replay contracts: real metadata/writer, Accelerator and two-Gloo loading.

Only new tests live here. The frozen stock17 module and its oracles stay unchanged.
Loader workers record original results/exceptions before any subsequent collective;
setup errors, peer EOF and timeouts are not accepted as product RED. Stock A3 controls
reuse the frozen real train/update/checkpoint harness, including its artifact-IO limit.
"""

from __future__ import annotations

import io
import logging
import os
import time
import traceback
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator

pytest.importorskip("transformers", reason="requires the parent offline ISAAC environment")
pytest.importorskip("mharmony", reason="requires the parent offline ISAAC environment")

from lerobot.common.train_utils import (
    load_training_metadata,
    load_training_window,
    resume_before_prepare,
    save_training_state,
)
from lerobot.common.training_window import (
    TrainingWindowCheckpoint,
    plan_training_window,
)
from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed.factory import make_accelerator
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from lerobot.policies.training_loss_normalization import TrainingLossNormalization, TrainingObjectiveSchema
from lerobot.processor import ImageAugmentationProcessorStep
from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig
from lerobot.utils.constants import TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.io_utils import write_json
from lerobot.utils.random_utils import serialize_rng_state
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_p2_stock_accumulation import (
    Case,
    Run,
    _assert_reference,
    _assert_tree_equal,
    _launch,
)
from tests.policies.perceptron_isaac.test_training_window_state import _normalizer


class SidecarCase(str, Enum):
    HISTORICAL = "historical"
    READ_LENGTH = "read_length"
    SCHEMA = "schema"
    TOTALS = "totals"
    MALFORMED_RANK = "malformed_rank"
    RANK_LOCAL = "rank_local"
    BOUNDARY = "boundary"
    MISSING_NEW_BOUNDARY = "missing_new_boundary"


@dataclass(frozen=True)
class ReplayLoadCase:
    kind: SidecarCase
    step: int = 1
    saved_accumulation: int = 2
    resumed_accumulation: int = 2


class LoadObservation(TypedDict):
    """Safe receipt of the actual load call; serialized state retains its production shape."""

    status: Literal["state", "none", "error"]
    error_type: str | None
    error_message: str | None
    original_traceback: str | None
    saved: dict[str, Any] | None
    loaded: dict[str, Any] | None
    warnings: str


def _replay_config(root: Path, accumulation: int) -> TrainPipelineConfig:
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/replay-contract"),
        policy=DummyCheckpointConfig(device="cpu", push_to_hub=False),
        output_dir=root / "output",
        batch_size=1,
        accelerator=AcceleratorConfig(gradient_accumulation=GradientAccumulationConfig(steps=accumulation)),
    )
    config.checkpoint_path = root / "checkpoint"
    return config


def _rank_checkpoint(case: ReplayLoadCase, accelerator: Accelerator, rank: int) -> TrainingWindowCheckpoint:
    """Build valid sidecars via real count planning; mutate only the reviewed shared field."""
    torch.manual_seed(401 + rank)
    normalizer = _normalizer()
    window = None
    if case.step % case.saved_accumulation:
        length = 1 if case.kind == SidecarCase.READ_LENGTH and rank == 0 else 2
        inputs = [
            normalizer.encode_batch(
                {PERCEPTRON_ISAAC_STREAM_KEY: flow_stream(), "rank_payload": torch.tensor([rank])}
            )
            for _ in range(length)
        ]
        window = plan_training_window(0, inputs, normalizer, accelerator)
        window.record_forward_rng()
        if case.kind == SidecarCase.SCHEMA and rank == 1:
            assert len(normalizer.schema.names) == 2
            window = replace(
                window,
                normalization=TrainingLossNormalization(
                    TrainingObjectiveSchema(tuple(reversed(normalizer.schema.names))),
                    window.normalization.totals,
                    case.saved_accumulation,
                ),
            )
        if case.kind == SidecarCase.TOTALS and rank == 1:
            window = replace(
                window,
                normalization=TrainingLossNormalization(
                    normalizer.schema,
                    tuple(count + 1 for count in window.normalization.totals),
                    case.saved_accumulation,
                ),
            )
    augmentation = ImageAugmentationProcessorStep(
        config=ImageTransformsConfig(
            enable=True,
            max_num_transforms=1,
            tfs={"brightness": ImageTransformConfig(type="ColorJitter", kwargs={"brightness": (0.5, 1.5)})},
        ),
        image_keys=["observation.images.image"],
        device="cpu",
        seed=193 + rank,
    )
    augmentation.observation({"observation.images.image": torch.full((1, 3, 8, 8), 0.5)})
    torch.rand(rank + 1)
    checkpoint = TrainingWindowCheckpoint(
        step=case.step,
        read_step=case.step if window is None else window.read_step,
        world_size=accelerator.num_processes,
        process_index=rank,
        backward_divisor=case.saved_accumulation,
        window=window,
        continuation_rng=serialize_rng_state(),
        augmentation=augmentation.state_dict(),
    )
    # Every intentional shared-plan mismatch is locally valid, not a broken fixture.
    _assert_tree_equal(
        TrainingWindowCheckpoint.from_state_dict(checkpoint.state_dict()).state_dict(),
        checkpoint.state_dict(),
    )
    return checkpoint


def _replay_load_worker(rank: int, rendezvous: str, directory: str, case: ReplayLoadCase) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29500", OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    root = Path(directory)
    try:
        config = _replay_config(root, case.resumed_accumulation)
        config.resume = True
        accelerator = make_accelerator(config)
        assert accelerator.device.type == "cpu" and accelerator.num_processes == 2
        assert accelerator.gradient_accumulation_steps == case.resumed_accumulation
        saved_config = _replay_config(root, case.saved_accumulation)
        saved_config.parallelism.resolve(accelerator.num_processes)
        checkpoint = (
            None if case.kind == SidecarCase.HISTORICAL else _rank_checkpoint(case, accelerator, rank)
        )
        checkpoint_path = saved_config.checkpoint_path
        assert checkpoint_path is not None
        # Historical shape uses the maintained metadata/RNG writer without a window.
        # No sidecar-required marker/signature is invented for that older format.
        save_training_state(
            checkpoint_path,
            case.step,
            saved_config,
            accelerator=None if case.kind == SidecarCase.HISTORICAL else accelerator,
            training_window=checkpoint,
        )
        accelerator.wait_for_everyone()
        metadata = load_training_metadata(checkpoint_path / TRAINING_STATE_DIR)
        assert metadata["step"] == case.step
        assert metadata["grad_accum_steps"] == case.saved_accumulation
        assert metadata["dp_world_size"] == 2 and metadata["batch_size"] == 1
        sidecar = checkpoint_path / TRAINING_STATE_DIR / f"training_window_rank_{rank}.pt"
        if case.kind == SidecarCase.MISSING_NEW_BOUNDARY:
            assert checkpoint is not None and checkpoint.window is None
            assert checkpoint.augmentation and checkpoint.continuation_rng
            assert sidecar.is_file(), "new writer must finish before deliberate removal"
            sidecar.unlink()
        elif case.kind == SidecarCase.MALFORMED_RANK and rank == 1:
            raw = torch.load(sidecar, map_location="cpu", weights_only=True)
            raw["version"] = -1  # Corrupt only after the actual safe writer produced a valid sidecar.
            torch.save(raw, sidecar)
        accelerator.wait_for_everyone()

        warnings = io.StringIO()
        handler = logging.StreamHandler(warnings)
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            # Actual caller path: cadence changes warn here; they are not globally prohibited.
            step = resume_before_prepare(config)
        finally:
            logger.removeHandler(handler)
            handler.close()
        assert step == case.step
        observation: LoadObservation = {
            "status": "none",
            "error_type": None,
            "error_message": None,
            "original_traceback": None,
            "saved": None if checkpoint is None else checkpoint.state_dict(),
            "loaded": None,
            "warnings": warnings.getvalue(),
        }
        try:
            loaded = load_training_window(config, accelerator, step)
        except Exception as error:
            # Persist the original per-rank exception, not a later peer-disconnect symptom.
            observation["status"] = "error"
            observation["error_type"] = type(error).__name__
            observation["error_message"] = str(error)
            observation["original_traceback"] = traceback.format_exc()
        else:
            if loaded is not None:
                observation["status"] = "state"
                observation["loaded"] = loaded.state_dict()
        torch.save(observation, root / f"load-rank-{rank}.pt")
        # Both old-code ranks reach this even when exactly one load raised locally.
        accelerator.wait_for_everyone()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _launch_replay_load(root: Path, case: ReplayLoadCase) -> list[LoadObservation]:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo required")
    root.mkdir(parents=True)
    context = mp.spawn(
        _replay_load_worker, args=((root / "gloo").as_uri(), str(root), case), nprocs=2, join=False
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("replay load exceeded 180s; timeout is NOT product assertion RED")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return [torch.load(root / f"load-rank-{rank}.pt", weights_only=True) for rank in range(2)]


def _load_summary(observations: list[LoadObservation]) -> list[dict[str, Any]]:
    """Put decisive returned plans/original errors in assertion output, not only temp receipts."""
    summaries = []
    for rank, observation in enumerate(observations):
        loaded = observation["loaded"]
        window = None if loaded is None else loaded["window"]
        summaries.append(
            {
                "rank": rank,
                "status": observation["status"],
                "error_type": observation["error_type"],
                "error_message": observation["error_message"],
                "read_step": None if loaded is None else loaded["read_step"],
                "inputs": None if window is None else len(window["inputs"]),
                "schema": None if window is None else window["schema"],
                "totals": None if window is None else window["totals"],
            }
        )
    return summaries


def _assert_collective_value_error(observations: list[LoadObservation]) -> None:
    assert [item["status"] for item in observations] == ["error", "error"], _load_summary(observations)
    assert [item["error_type"] for item in observations] == ["ValueError", "ValueError"], _load_summary(
        observations
    )
    assert all(item["original_traceback"] for item in observations)


@pytest.mark.parametrize(
    ("step", "saved_accumulation", "resumed_accumulation", "pending"),
    [
        pytest.param(2, 3, 2, True, id="old-a3-step2-pending-new-a2"),
        pytest.param(3, 3, 2, False, id="old-a3-step3-boundary-new-a2"),
        pytest.param(3, 3, 3, False, id="same-a3-boundary"),
        pytest.param(2, 3, 3, True, id="same-a3-pending"),
        pytest.param(6, 3, 2, False, id="common-boundary-change"),
    ],
)
def test_historical_missing_sidecar_uses_saved_cadence(
    tmp_path: Path, step: int, saved_accumulation: int, resumed_accumulation: int, pending: bool
) -> None:
    observations = _launch_replay_load(
        tmp_path / "historical",
        ReplayLoadCase(SidecarCase.HISTORICAL, step, saved_accumulation, resumed_accumulation),
    )
    if saved_accumulation != resumed_accumulation:
        assert (
            f"grad_accum_steps: {saved_accumulation} -> {resumed_accumulation}" in observations[0]["warnings"]
        )
    else:
        assert not any(item["warnings"] for item in observations)
    if pending:
        _assert_collective_value_error(observations)
    else:
        assert [item["status"] for item in observations] == ["none", "none"], _load_summary(observations)


def test_historical_metadata_without_saved_cadence_remains_explicitly_unknown(tmp_path: Path) -> None:
    # Real pre-v0.7 shape, also used in test_checkpoint_save_resume; no modern marker fields.
    write_json({"step": 6, "num_processes": 2}, tmp_path / TRAINING_STEP)
    metadata = load_training_metadata(tmp_path)
    assert metadata["step"] == 6 and metadata["dp_world_size"] == 2
    assert metadata["grad_accum_steps"] is None
    assert metadata["batch_size"] is None
    # No invented old cadence, boundary claim or new user-visible policy for unknown history.


@pytest.mark.parametrize(
    "kind",
    [SidecarCase.READ_LENGTH, SidecarCase.SCHEMA, SidecarCase.TOTALS, SidecarCase.MALFORMED_RANK],
    ids=["different-read-length", "different-schema", "different-totals", "malformed-one-rank"],
)
def test_rank_sidecar_plan_or_validation_failure_reaches_every_rank(
    tmp_path: Path, kind: SidecarCase
) -> None:
    observations = _launch_replay_load(tmp_path / kind.value, ReplayLoadCase(kind))
    _assert_collective_value_error(observations)


@pytest.mark.parametrize(
    "kind", [SidecarCase.RANK_LOCAL, SidecarCase.BOUNDARY], ids=["rank-local", "boundary"]
)
def test_matching_shared_plan_preserves_rank_local_payload_rng_and_augmentation(
    tmp_path: Path, kind: SidecarCase
) -> None:
    step = 2 if kind == SidecarCase.BOUNDARY else 1
    observations = _launch_replay_load(tmp_path / kind.value, ReplayLoadCase(kind, step=step))
    assert [item["status"] for item in observations] == ["state", "state"], _load_summary(observations)
    for item in observations:
        assert item["loaded"] is not None and item["saved"] is not None
        _assert_tree_equal(item["loaded"], item["saved"])
    first, second = observations[0]["loaded"], observations[1]["loaded"]
    assert first is not None and second is not None
    assert not torch.equal(
        first["continuation_rng"]["torch_rng_state"], second["continuation_rng"]["torch_rng_state"]
    )
    assert not torch.equal(
        first["augmentation"]["generator_state"], second["augmentation"]["generator_state"]
    )
    if kind == SidecarCase.RANK_LOCAL:
        assert first["window"] is not None and second["window"] is not None
        assert first["window"]["schema"] == second["window"]["schema"]
        assert first["window"]["totals"] == second["window"]["totals"]
        assert len(first["window"]["inputs"]) == len(second["window"]["inputs"]) == 2
        assert not torch.equal(
            first["window"]["inputs"][0]["rank_payload"], second["window"]["inputs"][0]["rank_payload"]
        )


def test_new_writer_boundary_cannot_silently_lose_all_rank_sidecars(tmp_path: Path) -> None:
    observations = _launch_replay_load(
        tmp_path / "lost-boundary", ReplayLoadCase(SidecarCase.MISSING_NEW_BOUNDARY, step=2)
    )
    _assert_collective_value_error(observations)


@pytest.mark.parametrize("checkpoint_budget", [4, 2], ids=["prepared-suffix", "saved-tail-extension"])
def test_stock_a3_two_call_prefix_resume_preserves_engine_state_and_oracle(
    tmp_path: Path, checkpoint_budget: int
) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo required")
    run = Run(Case.RESUME, steps=4, accumulation=3)
    uninterrupted = _launch(tmp_path / "full", run)
    _assert_reference(uninterrupted, run)  # Frozen concatenated objective/gradient/counter oracle.
    interrupted = _launch(
        tmp_path / "interrupted",
        Run(Case.RESUME, steps=checkpoint_budget, accumulation=3, interrupt_at=2),
    )
    checkpoint = tmp_path / "interrupted" / "output" / "checkpoints" / "000002"
    metadata = load_training_metadata(checkpoint / TRAINING_STATE_DIR)
    assert metadata["step"] == 2 and metadata["grad_accum_steps"] == 3
    for rank in range(2):
        state = TrainingWindowCheckpoint.from_state_dict(
            torch.load(
                checkpoint / TRAINING_STATE_DIR / f"training_window_rank_{rank}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        assert state.window is not None and state.window.consumed == 2
        assert len(state.window.inputs) == min(3, checkpoint_budget)
        assert state.read_step == min(3, checkpoint_budget)
    moved = tmp_path / "moved-checkpoint"
    checkpoint.rename(moved)
    resumed = _launch(tmp_path / "resumed", replace(run, checkpoint=moved))
    for full, prefix, continuation in zip(uninterrupted, interrupted, resumed, strict=True):
        for key in (
            "final",
            "optimizer",
            "scheduler",
            "scaler",
            "pending",
            "uniform",
            "normal",
            "augmentation",
        ):
            _assert_tree_equal(continuation[key], full[key])
        assert len(prefix["forwards"]) == 2 and len(prefix["gradients"]) == 0
        assert prefix["scheduler"]["last_epoch"] == 2
        assert len(continuation["gradients"]) == len(full["gradients"]) == 1
        assert continuation["scheduler"]["last_epoch"] == full["scheduler"]["last_epoch"] == 4
        assert continuation["backward_divisor"] == full["backward_divisor"] == 3
        assert continuation["pending"], "A3 final short tail must not force a second update"
        assert len(continuation["forwards"]) == 4, "exactly two nonsync prefix calls plus two live calls"
        _assert_tree_equal(continuation["gradients"], full["gradients"])
        for actual, expected in zip(prefix["forwards"], full["forwards"][:2], strict=True):
            _assert_tree_equal(actual["draws"], expected["draws"])
        for actual, expected in zip(continuation["forwards"], full["forwards"], strict=True):
            for field in ("indices", "images", "draws", "targets", "pads"):  # type: Literal["indices", "images", "draws", "targets", "pads"]
                _assert_tree_equal(actual[field], expected[field])
        _assert_tree_equal(
            prefix["preprocessed"] + continuation["preprocessed"],
            full["preprocessed"],
        )
