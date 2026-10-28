"""Optional objective-count normalization for the stock microbatch update path."""

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class TrainingObjectiveSchema:
    """Ordered objective names are identical on every participating rank."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.names
            or any(not isinstance(name, str) or not name for name in self.names)
            or len(set(self.names)) != len(self.names)
        ):
            raise ValueError("Training objective schema requires unique, nonempty names.")


@dataclass(frozen=True)
class TrainingLossNormalization:
    """Global integer counts and the actual Accelerator backward divisor, not window length."""

    schema: TrainingObjectiveSchema
    totals: tuple[int, ...]
    backward_divisor: int

    def __post_init__(self) -> None:
        if len(self.totals) != len(self.schema.names) or any(
            type(count) is not int or count < 0 for count in self.totals
        ):
            raise ValueError("Training objective totals must be nonnegative integers matching the schema.")
        if not any(self.totals):
            raise ValueError("Training objective window is globally empty.")
        if type(self.backward_divisor) is not int or self.backward_divisor < 1:
            raise ValueError("Training objective backward divisor must be positive.")


def training_state_to_device(value: Any, device: torch.device) -> Any:
    """Copy safe prepared-input containers; reject objects requiring arbitrary pickle globals."""
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise ValueError("Training prepared inputs must not retain computation graphs.")
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError("Training prepared inputs must contain only finite values.")
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Training prepared-input state requires string dictionary keys.")
        return {key: training_state_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [training_state_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(training_state_to_device(item, device) for item in value)
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Training prepared inputs must contain only finite values.")
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Training prepared-input state contains unsupported type {type(value).__name__}.")


class TrainingLossNormalizer(Protocol):
    """Parameter-free policy capability; prepared state contains only tensors and primitives.

    Transport must not mutate the retained host input. Decode is policy-owned so the stock
    trainer never imports a policy-specific packed payload or unpickles arbitrary classes.
    """

    @property
    def schema(self) -> TrainingObjectiveSchema: ...

    def local_counts(self, batch: dict[str, Any]) -> tuple[int, ...]: ...

    def encode_batch(self, batch: dict[str, Any]) -> dict[str, Any]: ...

    def decode_batch(self, state: dict[str, Any], device: torch.device) -> dict[str, Any]: ...
