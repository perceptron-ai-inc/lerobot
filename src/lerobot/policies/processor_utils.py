"""Dependency-light input coercion shared by policy processor implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

# Optional batch protocol for processors that drop individual training samples.
# The trainer uses these original-batch indices to keep per-sample weights aligned
# with a policy's reduced ``forward(..., reduction="none")`` output.
RETAINED_SAMPLE_INDICES_KEY = "retained_sample_indices"


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like processor input to NumPy without changing NumPy inputs."""
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def coerce_rgb_uint8(
    value: Any,
    *,
    shape_error: str,
    squeeze_leading_singletons: bool = False,
    contiguous: bool = False,
) -> np.ndarray:
    """Coerce one grayscale/RGB(A), channel-first/last image to channel-last RGB uint8."""
    image = to_numpy(value)
    if squeeze_leading_singletons:
        while image.ndim > 3 and int(image.shape[0]) == 1:
            image = image[0]
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in {3, 4}:
        raise ValueError(shape_error.format(shape=image.shape))
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype in (np.float16, np.float32, np.float64) and image.size and float(np.nanmax(image)) <= 1.0:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image) if contiguous else image


def coerce_batch_text(
    value: Any,
    batch_size: int,
    *,
    item_label: str = "task strings",
    normalizer: Callable[[str], str] | None = None,
) -> list[str]:
    """Stringify scalar or batched text input, broadcasting a singleton to the batch."""
    if value is None:
        texts = [""] * batch_size
    elif isinstance(value, str):
        texts = [value] * batch_size
    elif torch.is_tensor(value) or isinstance(value, np.ndarray):
        values = value.detach().cpu() if torch.is_tensor(value) else value
        if values.ndim == 0:
            texts = [str(values.item())] * batch_size
        else:
            texts = [str(item) for item in values.reshape(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        texts = [str(item) for item in value]
    else:
        texts = [str(value)]

    if len(texts) == 1 and batch_size != 1:
        texts *= batch_size
    if len(texts) != batch_size:
        raise ValueError(f"Expected {batch_size} {item_label}, got {len(texts)}.")
    return [normalizer(text) for text in texts] if normalizer is not None else texts


def flatten_feature_names(raw_names: Any, *, empty_as_none: bool = False) -> list[str] | None:
    """Flatten grouped feature names in declaration order without returning group keys."""
    if raw_names is None:
        return None
    groups = raw_names.values() if isinstance(raw_names, dict) else (raw_names,)
    names: list[str] = []
    for group in groups:
        if isinstance(group, (list, tuple)):
            names.extend(str(name) for name in group)
        elif group is not None:
            names.append(str(group))
    return (names or None) if empty_as_none else names
