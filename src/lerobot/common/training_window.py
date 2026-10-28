"""Bounded host staging and replay plans for the stock training loop.

One window retains prepared inputs, never graphs or parameter-gradient buffers. A pending
prefix is reconstructed through update_policy after normal model/optimizer restoration.
Counters, scheduler, EMA and publication remain owned by the stock microbatch loop.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Final

import torch

from lerobot.policies.training_loss_normalization import (
    TrainingLossNormalization,
    TrainingLossNormalizer,
    TrainingObjectiveSchema,
    training_state_to_device,
)
from lerobot.utils.random_utils import deserialize_rng_state, serialize_rng_state

if TYPE_CHECKING:
    from accelerate import Accelerator


TRAINING_WINDOW_VERSION: Final = 1


@dataclass(frozen=True)
class TrainingWindowResumePlan:
    """Shared replay plan only; rank-private inputs, RNG and augmentation never enter agreement."""

    metadata_version: int | None
    sidecar_version: int | None
    step: int
    read_step: int
    saved_world_size: int | None
    saved_batch_size: int | None
    saved_backward_divisor: int | None
    start_step: int | None
    input_length: int | None
    consumed: int | None
    schema: tuple[str, ...] | None
    global_totals: tuple[int, ...] | None
    requested_endpoint: int
    requested_backward_divisor: int


def agree_training_window_error(error: Exception | None, accelerator: Accelerator, *, phase: str) -> None:
    """Finish local replay validation before peers enter any subsequent collective phase."""
    failed = torch.tensor(int(error is not None), device=accelerator.device, dtype=torch.int64)
    if bool(accelerator.reduce(failed, reduction="sum")):
        raise ValueError(f"Training window resume {phase} failed on at least one rank.") from error


def validate_training_window_plan(plan: TrainingWindowResumePlan, accelerator: Accelerator) -> None:
    """Compare a fixed-size digest of the shared replay plan, never private payload state."""
    digest = hashlib.sha256(json.dumps(asdict(plan)).encode()).digest()
    local = torch.tensor(list(digest), dtype=torch.uint8, device=accelerator.device)
    plans = accelerator.gather(local).reshape(accelerator.num_processes, -1)
    if not bool((plans == plans[0]).all()):
        raise ValueError("Training window resume shared plan differs across ranks.")


def count_training_inputs(
    inputs: list[dict[str, Any]], normalizer: TrainingLossNormalizer
) -> tuple[int, ...]:
    """Validate every retained input and its local counts without entering collectives."""
    totals = [0] * len(normalizer.schema.names)
    for state in inputs:
        counts = normalizer.local_counts(normalizer.decode_batch(state, torch.device("cpu")))
        if len(counts) != len(totals) or any(type(count) is not int or count < 0 for count in counts):
            raise ValueError(
                "Training normalizer local counts must match its schema and be nonnegative integers."
            )
        for index, count in enumerate(counts):
            totals[index] += count
    return tuple(totals)


def validate_training_normalizer(
    normalizer: TrainingLossNormalizer | None,
    accelerator: Accelerator,
    *,
    sample_weighting: bool,
    context_parallel: bool,
) -> None:
    """Every rank must declare the same schema, eligibility and actual backward divisor."""
    declaration = [
        None if normalizer is None else normalizer.schema.names,
        accelerator.gradient_accumulation_steps,
        sample_weighting,
        context_parallel,
    ]
    digest = hashlib.sha256(json.dumps(declaration).encode()).digest()
    local = torch.tensor(list(digest), dtype=torch.uint8, device=accelerator.device)
    declarations = accelerator.gather(local).reshape(accelerator.num_processes, -1)
    if not bool((declarations == declarations[0]).all()):
        raise ValueError("Training normalizer schema or collective eligibility differs across ranks.")
    if normalizer is not None and (sample_weighting or context_parallel):
        raise ValueError("Training objective windows require unweighted data-parallel training.")


def prepare_training_input(
    prepare: Callable[[], dict[str, Any]], normalizer: TrainingLossNormalizer, accelerator: Accelerator
) -> dict[str, Any]:
    """Reject invalid real inputs on every rank before any rank starts a model collective."""
    error: ValueError | TypeError | KeyError | None = None
    state = None
    try:
        state = normalizer.encode_batch(prepare())
        count_training_inputs([state], normalizer)
    except (ValueError, TypeError, KeyError) as invalid:
        error = invalid
    failed = torch.tensor(int(error is not None), device=accelerator.device, dtype=torch.int64)
    if bool(accelerator.reduce(failed, reduction="sum")):
        raise ValueError("Training prepared input is invalid on at least one rank.") from error
    assert state is not None
    return state


@dataclass
class PreparedTrainingWindow:
    """Rank-local prepared prefix/suffix and immutable global plan; consumed tracks relative phase."""

    start_step: int
    inputs: list[dict[str, Any]]
    normalization: TrainingLossNormalization
    forward_rng: list[dict[str, torch.Tensor]] = field(default_factory=list)

    def __post_init__(self) -> None:
        divisor = self.normalization.backward_divisor
        if type(self.start_step) is not int or self.start_step < 0:
            raise ValueError("Training window start must be a nonnegative integer.")
        if not 0 < len(self.inputs) <= divisor or len(self.forward_rng) > len(self.inputs):
            raise ValueError("Training window inputs and forward RNG must fit one accumulation window.")

    @property
    def consumed(self) -> int:
        return len(self.forward_rng)

    @property
    def read_step(self) -> int:
        return self.start_step + len(self.inputs)

    def record_forward_rng(self) -> None:
        assert self.consumed < len(self.inputs)
        self.forward_rng.append(serialize_rng_state())

    def state_dict(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "inputs": self.inputs,
            "schema": self.normalization.schema.names,
            "totals": self.normalization.totals,
            "backward_divisor": self.normalization.backward_divisor,
            # First entry is the window-entry model RNG; later entries preserve interleaved eval RNG.
            "forward_rng": self.forward_rng,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> PreparedTrainingWindow:
        if (
            not isinstance(state["inputs"], list)
            or not all(isinstance(item, dict) for item in state["inputs"])
            or not isinstance(state["forward_rng"], list)
            or not isinstance(state["schema"], (list, tuple))
            or not isinstance(state["totals"], (list, tuple))
        ):
            raise ValueError(
                "Training window checkpoint requires prepared-input, RNG and objective sequences."
            )
        window = cls(
            start_step=state["start_step"],
            inputs=state["inputs"],
            normalization=TrainingLossNormalization(
                TrainingObjectiveSchema(tuple(state["schema"])),
                tuple(state["totals"]),
                state["backward_divisor"],
            ),
            forward_rng=state["forward_rng"],
        )
        if not 0 < window.consumed <= len(window.inputs) <= window.normalization.backward_divisor:
            raise ValueError("Training window checkpoint has invalid prepared-input or prefix lengths.")
        if window.consumed >= window.normalization.backward_divisor:
            raise ValueError("Training window checkpoint must not replay a synchronized optimizer update.")
        return window


def plan_training_window(
    start_step: int,
    inputs: list[dict[str, Any]],
    normalizer: TrainingLossNormalizer,
    accelerator: Accelerator,
    *,
    forward_rng: list[dict[str, torch.Tensor]] | None = None,
) -> PreparedTrainingWindow:
    """SUM integer objective counts once over actual retained inputs across all ranks."""
    assert 0 < len(inputs) <= accelerator.gradient_accumulation_steps
    error = None
    local = None
    try:
        local = count_training_inputs(inputs, normalizer)
    except Exception as invalid:
        error = invalid
    agree_training_window_error(error, accelerator, phase="input counting")
    assert local is not None
    totals = accelerator.reduce(
        torch.tensor(local, dtype=torch.int64, device=accelerator.device), reduction="sum"
    )
    context = TrainingLossNormalization(
        normalizer.schema, tuple(totals.tolist()), accelerator.gradient_accumulation_steps
    )
    return PreparedTrainingWindow(start_step, inputs, context, [] if forward_rng is None else forward_rng)


@dataclass(frozen=True)
class TrainingWindowCheckpoint:
    """Relative rank-local sidecar, including RNG even at an optimizer boundary."""

    step: int
    read_step: int
    world_size: int
    process_index: int
    backward_divisor: int
    window: PreparedTrainingWindow | None
    continuation_rng: dict[str, torch.Tensor]
    augmentation: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for value in (
                self.step,
                self.read_step,
                self.world_size,
                self.process_index,
                self.backward_divisor,
            )
        ):
            raise ValueError("Training window checkpoint cursors and topology must be integers.")
        if (
            self.step < 0
            or self.read_step < self.step
            or self.world_size < 1
            or not 0 <= self.process_index < self.world_size
            or self.backward_divisor < 1
            or self.read_step - self.step >= self.backward_divisor
        ):
            raise ValueError("Training window checkpoint cursors or topology are invalid.")

    def state_dict(self) -> dict[str, Any]:
        return training_state_to_device(
            {
                "version": TRAINING_WINDOW_VERSION,
                "step": self.step,
                "read_step": self.read_step,
                "world_size": self.world_size,
                "process_index": self.process_index,
                "backward_divisor": self.backward_divisor,
                "window": None if self.window is None else self.window.state_dict(),
                "continuation_rng": self.continuation_rng,
                "augmentation": self.augmentation,
            },
            torch.device("cpu"),
        )

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TrainingWindowCheckpoint:
        if type(state["version"]) is not int or state["version"] != TRAINING_WINDOW_VERSION:
            raise ValueError("Training window checkpoint version is unsupported.")
        # Check all safe containers, including the suffix that replay has not consumed yet.
        training_state_to_device(state, torch.device("cpu"))
        raw_window = state["window"]
        result = cls(
            state["step"],
            state["read_step"],
            state["world_size"],
            state["process_index"],
            state["backward_divisor"],
            None if raw_window is None else PreparedTrainingWindow.from_state_dict(raw_window),
            state["continuation_rng"],
            state["augmentation"],
        )
        if result.window is None:
            if result.read_step != result.step:
                raise ValueError("Training window checkpoint lost a pending accumulation prefix.")
        elif (
            result.read_step != result.window.read_step
            or result.step != result.window.start_step + result.window.consumed
            or result.window.normalization.backward_divisor != result.backward_divisor
        ):
            raise ValueError("Training window checkpoint cursors disagree with its accumulation plan.")
        rng_states = [result.continuation_rng]
        if result.window is not None:
            rng_states.extend(result.window.forward_rng)
        live_rng = serialize_rng_state()
        try:
            for rng in rng_states:
                if not isinstance(rng, dict) or not all(
                    isinstance(value, torch.Tensor) for value in rng.values()
                ):
                    raise ValueError("Training window checkpoint RNG must contain tensors.")
                deserialize_rng_state(rng)
        finally:
            deserialize_rng_state(live_rng)
        augmentation = result.augmentation
        if not isinstance(augmentation, dict) or (
            augmentation
            and (
                set(augmentation) != {"generator_state"}
                or not isinstance(augmentation["generator_state"], torch.Tensor)
                or augmentation["generator_state"].dtype != torch.uint8
                or augmentation["generator_state"].ndim != 1
                or augmentation["generator_state"].numel() == 0
            )
        ):
            raise ValueError("Training window checkpoint augmentation generator state is invalid.")
        return result
