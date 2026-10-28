"""Tests-only cadence continuation gates; parent alone owns CPU runtime RED.

Use real phase-one resume, safe writers, normalizer/count planning and Accelerator.
Direct window probes test cursor invariants, not a substitute model/update engine.
Original per-rank results are saved before a healthy barrier; EOF is not evidence.
The accepted replay15 and original stock17 modules remain immutable.
"""

from __future__ import annotations

import os
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator
from safetensors.torch import load_file

pytest.importorskip("transformers", reason="requires the parent offline ISAAC environment")
pytest.importorskip("mharmony", reason="requires the parent offline ISAAC environment")

from lerobot.common.train_utils import (
    load_training_metadata,
    load_training_window,
    resume_before_prepare,
    save_training_state,
)
from lerobot.common.training_window import TrainingWindowCheckpoint, plan_training_window
from lerobot.distributed.factory import make_accelerator
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from lerobot.utils.constants import RNG_STATE, TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.io_utils import write_json
from lerobot.utils.random_utils import serialize_rng_state
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_p2_stock_accumulation import _assert_tree_equal
from tests.policies.perceptron_isaac.test_p2_stock_replay_contracts import _replay_config
from tests.policies.perceptron_isaac.test_training_window_state import _normalizer


class CadenceTarget(str, Enum):
    LEGACY = "legacy"
    MALFORMED_METADATA = "malformed_metadata"
    VALID_METADATA = "valid_metadata"
    FRESH_PLAN = "fresh_plan"
    BOUNDARY_STATE = "boundary_state"


@dataclass(frozen=True)
class CadenceCase:
    target: CadenceTarget
    step: int = 3


class CadenceObservation(TypedDict):
    """Safe receipt of the original maintained call, including the failing phase."""

    status: Literal["step", "none", "state", "error"]
    phase: str
    step: int | None
    saved_cadence: int | None
    error_type: str | None
    error_message: str | None
    original_traceback: str | None
    saved: dict[str, Any] | None
    loaded: dict[str, Any] | None


def _cadence_observation(phase: str) -> CadenceObservation:
    return {
        "status": "error",
        "phase": phase,
        "step": None,
        "saved_cadence": None,
        "error_type": None,
        "error_message": None,
        "original_traceback": None,
        "saved": None,
        "loaded": None,
    }


def _record_original_error(observation: CadenceObservation, error: Exception) -> None:
    observation["status"] = "error"
    observation["error_type"] = type(error).__name__
    observation["error_message"] = str(error)
    observation["original_traceback"] = traceback.format_exc()


def _observe_phase_one(
    root: Path, case: CadenceCase, accelerator: Accelerator, rank: int
) -> CadenceObservation:
    saved_config = _replay_config(root, 3)
    saved_config.parallelism.resolve(accelerator.num_processes)
    shared = saved_config.checkpoint_path
    assert shared is not None
    # Supported historical writer: real metadata and legacy RNG, no window sidecar.
    save_training_state(shared, case.step, saved_config)
    accelerator.wait_for_everyone()
    local = root / f"rank-{rank}" / "checkpoint"
    shutil.copytree(shared, local)
    directory = local / TRAINING_STATE_DIR
    assert not list(directory.glob("training_window_rank_*.pt"))
    assert (directory / RNG_STATE).is_file()
    saved_rng = load_file(directory / RNG_STATE)
    metadata = load_training_metadata(directory)
    assert metadata["step"] == case.step and metadata["grad_accum_steps"] == 3
    if case.target == CadenceTarget.LEGACY:
        # Actual pre-v0.7 shape; unknown cadence is not a fabricated resumed-A2 boundary.
        write_json({"step": case.step, "num_processes": 2}, directory / TRAINING_STEP)
        metadata = load_training_metadata(directory)
        assert metadata["grad_accum_steps"] is None
        assert metadata["batch_size"] is None and metadata["dp_world_size"] == 2
    elif case.target == CadenceTarget.MALFORMED_METADATA and rank == 1:
        (directory / TRAINING_STEP).write_text('{"step":', encoding="utf-8")
    accelerator.wait_for_everyone()

    config = _replay_config(root / f"rank-{rank}", 2)
    config.checkpoint_path = local
    config.resume = True
    config.parallelism.resolve(accelerator.num_processes)
    observation = _cadence_observation("resume_before_prepare")
    observation["saved_cadence"] = metadata["grad_accum_steps"]
    torch.rand(7 + rank)  # Phase one must restore the writer's real state, not current RNG.
    try:
        step = resume_before_prepare(config)
    except Exception as error:
        _record_original_error(observation, error)
        return observation
    observation["status"] = "step"
    observation["step"] = step
    _assert_tree_equal(serialize_rng_state(), saved_rng)
    if case.target == CadenceTarget.LEGACY:
        # Never let the peer's original local metadata exception enter a later collective.
        # Only the all-valid historical probe continues to the real window loader.
        observation["phase"] = "load_training_window"
        try:
            loaded = load_training_window(config, accelerator, step)
        except Exception as error:
            _record_original_error(observation, error)
        else:
            observation["status"] = "none" if loaded is None else "state"
            observation["loaded"] = None if loaded is None else loaded.state_dict()
        assert load_training_metadata(directory)["grad_accum_steps"] is None
    return observation


def _observe_relative_window(case: CadenceCase, accelerator: Accelerator, rank: int) -> CadenceObservation:
    # A3/step3 (or step6 control) is a real old boundary, with no pending prepared prefix.
    old_boundary = TrainingWindowCheckpoint(case.step, case.step, 2, rank, 3, None, serialize_rng_state(), {})
    _assert_tree_equal(
        TrainingWindowCheckpoint.from_state_dict(old_boundary.state_dict()).state_dict(),
        old_boundary.state_dict(),
    )
    assert accelerator.step == 0 and accelerator.sync_gradients
    assert accelerator.gradient_accumulation_steps == 2
    observation = _cadence_observation(case.target.value)
    observation["step"] = case.step
    observation["saved_cadence"] = 3
    if case.target == CadenceTarget.BOUNDARY_STATE:
        # Fresh resumed Accelerator is still at entry: window=None, read_step==step.
        boundary = TrainingWindowCheckpoint(case.step, case.step, 2, rank, 2, None, serialize_rng_state(), {})
        observation["saved"] = boundary.state_dict()
        try:
            restored = TrainingWindowCheckpoint.from_state_dict(boundary.state_dict())
        except Exception as error:
            _record_original_error(observation, error)
        else:
            observation["status"] = "state"
            observation["loaded"] = restored.state_dict()
        return observation

    normalizer = _normalizer()
    inputs = [
        normalizer.encode_batch(
            {PERCEPTRON_ISAAC_STREAM_KEY: flow_stream(), "rank_payload": torch.tensor([rank, index])}
        )
        for index in range(2)
    ]
    counts = normalizer.local_counts(normalizer.decode_batch(inputs[0], torch.device("cpu")))
    # Positive count control uses the same real inputs and collective, with no injected denominator.
    aligned = plan_training_window(0, inputs, normalizer, accelerator)
    assert aligned.normalization.schema == normalizer.schema
    assert aligned.normalization.totals == tuple(4 * count for count in counts)
    assert aligned.normalization.backward_divisor == 2
    observation["saved"] = aligned.state_dict()
    try:
        window = plan_training_window(case.step, inputs, normalizer, accelerator)
    except Exception as error:
        _record_original_error(observation, error)
        return observation
    assert window.start_step == case.step and window.read_step == case.step + 2
    assert window.consumed == 0 and window.normalization == aligned.normalization
    _assert_tree_equal(window.inputs, inputs)
    # Real Accelerator phase, not absolute processed_counter % resumed_A.
    with accelerator.accumulate():
        window.record_forward_rng()
        assert not accelerator.sync_gradients
    assert window.consumed == 1
    pending = TrainingWindowCheckpoint(
        case.step + 1, window.read_step, 2, rank, 2, window, serialize_rng_state(), {}
    )
    observation["phase"] = "pending_checkpoint_roundtrip"
    observation["saved"] = pending.state_dict()
    try:
        restored = TrainingWindowCheckpoint.from_state_dict(pending.state_dict())
    except Exception as error:
        _record_original_error(observation, error)
        return observation
    observation["status"] = "state"
    observation["loaded"] = restored.state_dict()
    with accelerator.accumulate():
        window.record_forward_rng()
        assert accelerator.sync_gradients
    assert window.consumed == 2
    return observation


def _cadence_worker(rank: int, rendezvous: str, directory: str, case: CadenceCase) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29500", OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    root = Path(directory)
    try:
        accelerator = make_accelerator(_replay_config(root, 2))
        assert accelerator.device.type == "cpu" and accelerator.num_processes == 2
        assert accelerator.gradient_accumulation_steps == 2
        if case.target in (CadenceTarget.FRESH_PLAN, CadenceTarget.BOUNDARY_STATE):
            observation = _observe_relative_window(case, accelerator, rank)
        else:
            observation = _observe_phase_one(root, case, accelerator, rank)
        torch.save(observation, root / f"cadence-rank-{rank}.pt")
        accelerator.wait_for_everyone()  # Original results exist before this healthy final barrier.
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _launch_cadence(root: Path, case: CadenceCase) -> list[CadenceObservation]:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo required")
    root.mkdir(parents=True)
    context = mp.spawn(
        _cadence_worker, args=((root / "gloo").as_uri(), str(root), case), nprocs=2, join=False
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("cadence probe exceeded 180s; timeout is NOT product assertion RED")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return [torch.load(root / f"cadence-rank-{rank}.pt", weights_only=True) for rank in range(2)]


def _cadence_summary(observations: list[CadenceObservation]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in item.items() if key not in ("saved", "loaded")} for item in observations
    ]


def test_true_legacy_step3_unknown_cadence_keeps_baseline_resume(tmp_path: Path) -> None:
    observations = _launch_cadence(tmp_path / "legacy", CadenceCase(CadenceTarget.LEGACY))
    assert [item["saved_cadence"] for item in observations] == [None, None]
    assert [item["step"] for item in observations] == [3, 3], _cadence_summary(observations)
    assert [item["status"] for item in observations] == ["none", "none"], _cadence_summary(observations)


@pytest.mark.parametrize(
    "step", [pytest.param(3, id="a3-step3-to-a2"), pytest.param(6, id="aligned-control")]
)
@pytest.mark.parametrize(
    "target", [CadenceTarget.FRESH_PLAN, CadenceTarget.BOUNDARY_STATE], ids=["fresh-plan", "boundary-state"]
)
def test_fresh_a2_window_after_old_a3_boundary_uses_relative_phase(
    tmp_path: Path, step: int, target: CadenceTarget
) -> None:
    observations = _launch_cadence(tmp_path / target.value, CadenceCase(target, step))
    assert [item["status"] for item in observations] == ["state", "state"], _cadence_summary(observations)
    for observation in observations:
        assert observation["saved"] is not None and observation["loaded"] is not None
        _assert_tree_equal(observation["loaded"], observation["saved"])


@pytest.mark.parametrize(
    "malformed", [pytest.param(True, id="one-rank-malformed"), pytest.param(False, id="valid-control")]
)
def test_phase_one_metadata_read_refuses_collectively_before_prepare(tmp_path: Path, malformed: bool) -> None:
    target = CadenceTarget.MALFORMED_METADATA if malformed else CadenceTarget.VALID_METADATA
    observations = _launch_cadence(tmp_path / target.value, CadenceCase(target))
    assert all(item["phase"] == "resume_before_prepare" for item in observations)
    if malformed:
        assert [item["status"] for item in observations] == ["error", "error"], _cadence_summary(observations)
        assert all(item["error_type"] in ("ValueError", "JSONDecodeError") for item in observations)
        assert all(item["original_traceback"] for item in observations)
    else:
        assert [item["status"] for item in observations] == ["step", "step"], _cadence_summary(observations)
        assert [item["step"] for item in observations] == [3, 3]
