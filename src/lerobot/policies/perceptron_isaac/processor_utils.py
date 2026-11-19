"""Dependency-light input coercion and image geometry for the Perceptron Isaac processors.

These helpers are Isaac-owned: the processor pipeline and the mharmony packer are their
only callers, so they live in the policy package rather than in shared LeRobot modules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from torchvision.transforms import InterpolationMode

# Batch key carrying the original-batch indices of the samples a processor kept. Isaac's
# processor drops individual training samples, and this records which ones survived.
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


def letterbox_resize_image(
    image: torch.Tensor,
    size: Sequence[int],
    *,
    fill: float = 0.0,
    interpolation: InterpolationMode | None = None,
    antialias: bool = True,
) -> torch.Tensor:
    """Resize with one scale factor, then center-pad to an exact ``(height, width)`` canvas.

    Accepts ``[..., C, H, W]`` tensors, so it serves single frames and temporal/batched
    stacks alike. Training and deployment share this one implementation so their image
    geometry cannot drift. ``interpolation`` defaults to bilinear.
    """
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tvf

    if len(size) != 2:
        raise ValueError(f"Letterbox size must be (height, width), got {tuple(size)}.")
    target_height, target_width = (int(value) for value in size)
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Letterbox size must be positive, got {(target_height, target_width)}.")
    if image.ndim < 3:
        raise ValueError(f"Letterbox expects [..., C, H, W], got {tuple(image.shape)}.")

    source_height, source_width = (int(value) for value in image.shape[-2:])
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Letterbox source dimensions must be positive, got {tuple(image.shape[-2:])}.")
    if (source_height, source_width) == (target_height, target_width):
        return image

    scale = min(target_height / source_height, target_width / source_width)
    resized_height = min(target_height, max(1, int(round(source_height * scale))))
    resized_width = min(target_width, max(1, int(round(source_width * scale))))
    resized = tvf.resize_image(
        image,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR if interpolation is None else interpolation,
        antialias=antialias,
    )

    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return torch.nn.functional.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=fill)
