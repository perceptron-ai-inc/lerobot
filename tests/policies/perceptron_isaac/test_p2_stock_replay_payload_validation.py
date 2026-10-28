"""Tests-only pre-prepare replay payload gates; parent owns actual CPU RED.

Use real staging, count plans, writers and two-Gloo loaders, never model forward or
late invalid augmentation restoration. Original loader receipts precede the final
healthy barrier. Nonfinite noise is a defensive control: the existing safe codec
already rejects it, independently of the missing explicit-randomness validation.
"""

from __future__ import annotations

import copy
import hashlib
import os
import time
import traceback
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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
    count_training_inputs,
    plan_training_window,
)
from lerobot.distributed.factory import make_accelerator
from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    ISAAC_FLOW_NOISE_KEY,
    ISAAC_FLOW_TIMESTEPS_KEY,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from lerobot.processor import ImageAugmentationProcessorStep
from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig
from lerobot.utils.constants import TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.random_utils import deserialize_rng_state, serialize_rng_state
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_p2_stock_accumulation import _assert_tree_equal
from tests.policies.perceptron_isaac.test_p2_stock_replay_contracts import _replay_config
from tests.policies.perceptron_isaac.test_training_window_state import _normalizer


class PayloadCase(str, Enum):
    TIMESTEP_RANGE = "timestep-range"
    NOISE_SHAPE = "noise-shape"
    NOISE_NONFINITE = "noise-nonfinite"
    GENERATOR_STATE = "generator-state"
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    GRID_SHORTHAND = "grid-shorthand"


class PayloadObservation(TypedDict):
    """Safe, rank-local receipt of the original loader outcome, not later peer errors."""

    rank: int
    phase: str
    status: Literal["state", "none", "error"]
    error_type: str | None
    error_message: str | None
    cause_type: str | None
    cause_message: str | None
    original_traceback: str | None
    saved: dict[str, Any]
    private_sidecar: dict[str, Any]
    loaded: dict[str, Any] | None
    live_before: dict[str, torch.Tensor]
    live_after: dict[str, torch.Tensor]
    sidecar_sha_before: str
    sidecar_sha_after: str
    metadata_before: bytes
    metadata_after: bytes
    local_counts: tuple[int, ...]
    accelerator_step_before: int
    accelerator_step_after: int
    sync_before: bool
    sync_after: bool


def _augmentation(seed: int) -> ImageAugmentationProcessorStep:
    return ImageAugmentationProcessorStep(
        config=ImageTransformsConfig(
            enable=True,
            max_num_transforms=1,
            tfs={"brightness": ImageTransformConfig(type="ColorJitter", kwargs={"brightness": (0.5, 1.5)})},
        ),
        image_keys=["observation.images.image"],
        device="cpu",
        seed=seed,
    )


def _change_private_payload(state: dict[str, Any], case: PayloadCase) -> None:
    """Change exactly one rank's first retained PREFIX value, never the shared plan."""
    if case == PayloadCase.GENERATOR_STATE:
        state["augmentation"]["generator_state"] = torch.tensor([1], dtype=torch.uint8)
    elif case == PayloadCase.TIMESTEP_RANGE:
        state["window"]["inputs"][0][ISAAC_FLOW_TIMESTEPS_KEY] = torch.tensor([[2.0]])
    elif case == PayloadCase.NOISE_SHAPE:
        state["window"]["inputs"][0][ISAAC_FLOW_NOISE_KEY] = torch.zeros(1, 1, 2, 3)
    elif case == PayloadCase.NOISE_NONFINITE:
        state["window"]["inputs"][0][ISAAC_FLOW_NOISE_KEY][0, 0, 0, 0] = float("nan")


def _assert_only_requested_change(private: dict[str, Any], saved: dict[str, Any], case: PayloadCase) -> None:
    repaired = copy.deepcopy(private)
    if case == PayloadCase.GENERATOR_STATE:
        repaired["augmentation"] = saved["augmentation"]
    elif case in (PayloadCase.TIMESTEP_RANGE, PayloadCase.NOISE_SHAPE, PayloadCase.NOISE_NONFINITE):
        key = ISAAC_FLOW_TIMESTEPS_KEY if case == PayloadCase.TIMESTEP_RANGE else ISAAC_FLOW_NOISE_KEY
        repaired["window"]["inputs"][0][key] = saved["window"]["inputs"][0][key]
    # Whole-state equality also preserves version, topology, counters, schema, totals,
    # suffix and RNG, hence every input to the real shared-plan digest is unchanged.
    _assert_tree_equal(repaired, saved)


def _payload_worker(rank: int, rendezvous: str, directory: str, case: PayloadCase) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29500", OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    root = Path(directory)
    try:
        config = _replay_config(root, 2)
        config.resume = True
        config.steps = 2
        accelerator = make_accelerator(config)
        assert accelerator.device.type == "cpu" and accelerator.num_processes == 2
        assert accelerator.gradient_accumulation_steps == 2
        normalizer = _normalizer()
        # The maintained fixture explicitly selects one draw; never assume the product default.
        assert normalizer.config.train_samples_per_chunk == 1
        assert (normalizer.action_horizon, normalizer.action_dim) == (2, 2)
        torch.manual_seed(401 + rank)
        inputs = []
        count_rng = serialize_rng_state()
        for _ in range(2):
            batch: dict[str, Any] = {
                PERCEPTRON_ISAAC_STREAM_KEY: flow_stream(),
                "rank_payload": torch.tensor([rank]),
            }
            if case != PayloadCase.IMPLICIT:
                batch[ISAAC_FLOW_TIMESTEPS_KEY] = torch.full((1, 1), 0.25 + rank * 0.25)
                batch[ISAAC_FLOW_NOISE_KEY] = torch.full((1, 1, 2, 2), float(rank))
            if case == PayloadCase.GRID_SHORTHAND:
                batch[ISAAC_FLOW_TIMESTEPS_KEY] = torch.full((1, 1, 2, 1), 0.5)
                batch[ISAAC_FLOW_NOISE_KEY] = torch.full((1, 2, 2), float(rank))
            counts = normalizer.local_counts(batch)
            assert normalizer.schema.names == ("flow", "text") and all(count > 0 for count in counts)
            inputs.append(normalizer.encode_batch(batch))
        local_counts = count_training_inputs(inputs, normalizer)
        window = plan_training_window(0, inputs, normalizer, accelerator)
        assert window.normalization.totals == tuple(count * 2 for count in local_counts)
        assert all(total > 0 for total in window.normalization.totals)
        _assert_tree_equal(serialize_rng_state(), count_rng)  # Includes the absent-randomness case.
        window.record_forward_rng()
        augmentation = _augmentation(193 + rank)
        augmentation.observation({"observation.images.image": torch.full((1, 3, 8, 8), 0.5)})
        _assert_tree_equal(serialize_rng_state(), count_rng)
        torch.rand(rank + 1)
        checkpoint = TrainingWindowCheckpoint(
            step=1,
            read_step=window.read_step,
            world_size=accelerator.num_processes,
            process_index=rank,
            backward_divisor=accelerator.gradient_accumulation_steps,
            window=window,
            continuation_rng=serialize_rng_state(),
            augmentation=augmentation.state_dict(),
        )
        saved = checkpoint.state_dict()
        _assert_tree_equal(TrainingWindowCheckpoint.from_state_dict(saved).state_dict(), saved)
        checkpoint_path = config.checkpoint_path
        assert checkpoint_path is not None
        save_training_state(checkpoint_path, 1, config, accelerator=accelerator, training_window=checkpoint)
        accelerator.wait_for_everyone()
        directory_path = checkpoint_path / TRAINING_STATE_DIR
        metadata = load_training_metadata(directory_path)
        assert metadata["step"] == 1 and metadata["grad_accum_steps"] == 2
        assert metadata["dp_world_size"] == 2 and metadata["batch_size"] == 1
        step = resume_before_prepare(config)
        assert step == 1
        # Establish a real valid load before mutation, on both ranks, with the actual normalizer.
        baseline_rng = serialize_rng_state()
        baseline = load_training_window(config, accelerator, step, normalizer=normalizer)
        assert baseline is not None
        _assert_tree_equal(baseline.state_dict(), saved)
        _assert_tree_equal(serialize_rng_state(), baseline_rng)
        sidecar = directory_path / f"training_window_rank_{rank}.pt"
        private = torch.load(sidecar, weights_only=True, map_location="cpu")
        _assert_tree_equal(private, saved)
        if rank == 1:
            _change_private_payload(private, case)
            _assert_only_requested_change(private, saved, case)
            torch.save(private, sidecar)
        accelerator.wait_for_everyone()
        # Deliberately separate live RNG from both the sidecar and legacy phase-one RNG.
        torch.manual_seed(901 + rank)
        torch.rand(7 + rank)
        observation: PayloadObservation = {
            "rank": rank,
            "phase": "load_training_window before prepare/data-position/replay",
            "status": "none",
            "error_type": None,
            "error_message": None,
            "cause_type": None,
            "cause_message": None,
            "original_traceback": None,
            "saved": saved,
            "private_sidecar": private,
            "loaded": None,
            "live_before": serialize_rng_state(),
            "live_after": {},
            "sidecar_sha_before": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "sidecar_sha_after": "",
            "metadata_before": (directory_path / TRAINING_STEP).read_bytes(),
            "metadata_after": b"",
            "local_counts": local_counts,
            "accelerator_step_before": accelerator.step,
            "accelerator_step_after": -1,
            "sync_before": accelerator.sync_gradients,
            "sync_after": False,
        }
        try:
            loaded = load_training_window(config, accelerator, step, normalizer=normalizer)
        except Exception as error:
            observation["status"] = "error"
            observation["error_type"] = type(error).__name__
            observation["error_message"] = str(error)
            observation["original_traceback"] = traceback.format_exc()
            cause = error.__cause__
            observation["cause_type"] = None if cause is None else type(cause).__name__
            observation["cause_message"] = None if cause is None else str(cause)
        else:
            if loaded is not None:
                observation["status"] = "state"
                observation["loaded"] = loaded.state_dict()
        observation["live_after"] = serialize_rng_state()
        observation["sidecar_sha_after"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        observation["metadata_after"] = (directory_path / TRAINING_STEP).read_bytes()
        observation["accelerator_step_after"] = accelerator.step
        observation["sync_after"] = accelerator.sync_gradients
        torch.save(observation, root / f"payload-rank-{rank}.pt")
        # Only after original outcomes are durable; no prepare, DCP or replay is invoked.
        accelerator.wait_for_everyone()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _launch_payload(root: Path, case: PayloadCase) -> list[PayloadObservation]:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo required")
    root.mkdir(parents=True)
    context = mp.spawn(
        _payload_worker, args=((root / "gloo").as_uri(), str(root), case), nprocs=2, join=False
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("payload load exceeded 180s; timeout is NOT product assertion RED")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return [
        torch.load(root / f"payload-rank-{rank}.pt", weights_only=True, map_location="cpu")
        for rank in range(2)
    ]


def _assert_observation_preserved(item: PayloadObservation, case: PayloadCase) -> None:
    _assert_tree_equal(item["live_after"], item["live_before"])
    assert item["sidecar_sha_after"] == item["sidecar_sha_before"]
    assert item["metadata_after"] == item["metadata_before"]
    assert item["accelerator_step_before"] == item["accelerator_step_after"] == 0
    assert item["sync_before"] and item["sync_after"]
    saved, private = item["saved"], item["private_sidecar"]
    assert saved["step"] == 1 and saved["read_step"] == 2
    assert saved["window"]["start_step"] == 0 and len(saved["window"]["forward_rng"]) == 1
    assert len(saved["window"]["inputs"]) == saved["backward_divisor"] == 2
    assert all(count > 0 for count in item["local_counts"])
    _assert_only_requested_change(private, saved, case if item["rank"] == 1 else PayloadCase.EXPLICIT)
    if item["loaded"] is not None and case != PayloadCase.NOISE_NONFINITE:
        _assert_tree_equal(item["loaded"], private)  # Preserve the returned bad value in baseline RED.
    if item["rank"] == 1:
        prefix = private["window"]["inputs"][0]
        if case == PayloadCase.TIMESTEP_RANGE:
            _assert_tree_equal(prefix[ISAAC_FLOW_TIMESTEPS_KEY], torch.tensor([[2.0]]))
        elif case == PayloadCase.NOISE_SHAPE:
            assert tuple(prefix[ISAAC_FLOW_NOISE_KEY].shape) == (1, 1, 2, 3)
        elif case == PayloadCase.NOISE_NONFINITE:
            assert torch.isnan(prefix[ISAAC_FLOW_NOISE_KEY][0, 0, 0, 0])
        elif case == PayloadCase.GENERATOR_STATE:
            _assert_tree_equal(
                private["augmentation"]["generator_state"], torch.tensor([1], dtype=torch.uint8)
            )


def _payload_summary(items: list[PayloadObservation]) -> list[dict[str, Any]]:
    return [
        {
            "rank": item["rank"],
            "status": item["status"],
            "error_type": item["error_type"],
            "error_message": item["error_message"],
            "cause_type": item["cause_type"],
            "cause_message": item["cause_message"],
            "step": item["private_sidecar"]["step"],
            "read_step": item["private_sidecar"]["read_step"],
            "retained_timesteps": item["private_sidecar"]["window"]["inputs"][0].get(
                ISAAC_FLOW_TIMESTEPS_KEY
            ),
            "retained_noise": item["private_sidecar"]["window"]["inputs"][0].get(ISAAC_FLOW_NOISE_KEY),
            "generator_bytes": item["private_sidecar"]["augmentation"]["generator_state"].numel(),
        }
        for item in items
    ]


@pytest.mark.parametrize(
    "case",
    [
        PayloadCase.TIMESTEP_RANGE,
        PayloadCase.NOISE_SHAPE,
        PayloadCase.NOISE_NONFINITE,
        PayloadCase.GENERATOR_STATE,
    ],
    ids=lambda case: case.value,
)
def test_rank_private_payload_refuses_collectively_before_prepare(tmp_path: Path, case: PayloadCase) -> None:
    observations = _launch_payload(tmp_path / case.value, case)
    for item in observations:
        _assert_observation_preserved(item, case)
    summary = _payload_summary(observations)
    assert [item["status"] for item in observations] == ["error", "error"], (
        "DIDNOTREFUSE: invalid private payload returned before prepare/replay",
        summary,
    )
    assert [item["error_type"] for item in observations] == ["ValueError", "ValueError"], summary
    assert all(item["original_traceback"] for item in observations), summary
    assert observations[0]["cause_type"] is None, summary
    assert observations[1]["cause_type"] in ("ValueError", "RuntimeError"), summary
    cause_message = observations[1]["cause_message"]
    assert cause_message and not any(
        word in cause_message.lower() for word in ("timeout", "connection", "eof")
    )


def _assert_private_roundtrips(item: PayloadObservation) -> None:
    saved, loaded = item["saved"], item["loaded"]
    assert loaded is not None
    _assert_tree_equal(loaded, saved)
    live = serialize_rng_state()
    try:
        for original, restored in zip(
            [saved["continuation_rng"], *saved["window"]["forward_rng"]],
            [loaded["continuation_rng"], *loaded["window"]["forward_rng"]],
            strict=True,
        ):
            deserialize_rng_state(original)
            expected = torch.rand(3)
            deserialize_rng_state(restored)
            _assert_tree_equal(torch.rand(3), expected)
    finally:
        deserialize_rng_state(live)
    original_augmentation, restored_augmentation = _augmentation(11), _augmentation(13)
    original_augmentation.load_state_dict(saved["augmentation"])
    restored_augmentation.load_state_dict(loaded["augmentation"])
    _assert_tree_equal(restored_augmentation.state_dict(), loaded["augmentation"])
    image = torch.full((1, 3, 8, 8), 0.5)
    _assert_tree_equal(
        restored_augmentation.observation({"observation.images.image": image.clone()}),
        original_augmentation.observation({"observation.images.image": image.clone()}),
    )
    _assert_tree_equal(restored_augmentation.state_dict(), original_augmentation.state_dict())
    _assert_tree_equal(serialize_rng_state(), live)


@pytest.mark.parametrize(
    "case",
    [PayloadCase.EXPLICIT, PayloadCase.IMPLICIT, PayloadCase.GRID_SHORTHAND],
    ids=lambda case: case.value,
)
def test_valid_private_payload_preserves_randomness_without_sampling(
    tmp_path: Path, case: PayloadCase
) -> None:
    observations = _launch_payload(tmp_path / case.value, case)
    assert [item["status"] for item in observations] == ["state", "state"], _payload_summary(observations)
    for item in observations:
        _assert_observation_preserved(item, case)
        _assert_private_roundtrips(item)
        loaded = item["loaded"]
        assert loaded is not None
        normalizer = _normalizer()
        live = serialize_rng_state()
        assert count_training_inputs(loaded["window"]["inputs"], normalizer) == item["local_counts"]
        _assert_tree_equal(serialize_rng_state(), live)
        for state in loaded["window"]["inputs"]:
            if case == PayloadCase.IMPLICIT:
                assert ISAAC_FLOW_TIMESTEPS_KEY not in state and ISAAC_FLOW_NOISE_KEY not in state
            else:
                assert ISAAC_FLOW_TIMESTEPS_KEY in state and ISAAC_FLOW_NOISE_KEY in state
            assert state["rank_payload"].item() == item["rank"]
