"""Public ``mharmony`` boundary and ISAAC RenderedStream lowering.

The standalone package owns Harmony conversations, tokenization, and media
preprocessing. ISAAC deliberately owns the final conversion into its local
``TensorStream`` value type so neither package depends on the other's runtime
types.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .mharmony_contract import SUPPORTED_MHARMONY_VERSION
from .tensor_stream import Event, TensorStream, TextType, VectorType, VisionType, create_stream

PUBLIC_MHARMONY_PACKAGE = "mharmony"

_ISAAC_PRIORITY = [
    TextType.padding,
    TextType.timestamp,
    TextType.text,
    TextType.action,
    TextType.action_c,
    TextType.control,
    VectorType.vector,
    VisionType.I,
    VisionType.P,
]
_CONTROL_MODALITIES = frozenset({"control", "special_token", "role_marker", "modality_tag"})
_TEXT_TYPE_TAG = "genesis_text_type"
_TEXT_TYPES = {
    "text": TextType.text,
    "timestamp": TextType.timestamp,
    "action": TextType.action,
    "action_c": TextType.action_c,
}
_ROLE_ALIASES = {
    "assistant": "agent",
    "agent": "agent",
    "developer": "system",
    "system": "system",
    "tool": "tool_output",
    "tool_output": "tool_output",
    "user": "user",
}
_REQUIRED_PUBLIC_API = (
    "Author",
    "Conversation",
    "HarmonyEncodingName",
    "ImageContent",
    "MediaRef",
    "Message",
    "Role",
    "TextContent",
    "TokensContent",
    "VectorContent",
    "create_qwen35_image_processor",
    "load_harmony_encoding",
)


def import_mharmony_module(module_name: str):
    """Import only the standalone package or one of its public submodules."""
    if module_name != PUBLIC_MHARMONY_PACKAGE and not module_name.startswith(f"{PUBLIC_MHARMONY_PACKAGE}."):
        raise ImportError(
            f"Perceptron ISAAC may only import {PUBLIC_MHARMONY_PACKAGE} or "
            f"{PUBLIC_MHARMONY_PACKAGE}.*, got {module_name!r}."
        )
    return importlib.import_module(module_name)


@functools.cache
def _validate_installed_mharmony_version() -> str:
    """Require the exact distribution version covered by ISAAC parity tests."""
    try:
        installed = importlib.metadata.version(PUBLIC_MHARMONY_PACKAGE)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Perceptron ISAAC requires mharmony=={SUPPORTED_MHARMONY_VERSION}; "
            "install the perceptron_isaac extra."
        ) from exc
    if installed != SUPPORTED_MHARMONY_VERSION:
        raise RuntimeError(
            f"Perceptron ISAAC requires mharmony=={SUPPORTED_MHARMONY_VERSION}, but {installed} is installed."
        )
    return installed


def load_mharmony():
    _validate_installed_mharmony_version()
    return import_mharmony_module(PUBLIC_MHARMONY_PACKAGE)


def create_qwen35_image_processor(**kwargs: Any):
    """Create the canonical root-exported Qwen3.5 image processor."""
    factory = getattr(load_mharmony(), "create_qwen35_image_processor", None)
    if factory is None:
        raise RuntimeError("The installed mharmony package does not expose create_qwen35_image_processor.")
    return factory(**kwargs)


def _encoding_name_value(encoding_name: Any) -> str:
    value = getattr(encoding_name, "value", encoding_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"mharmony encoding name must be a non-empty string, got {value!r}.")
    return value


def _reserved_groups_cache_key(reserved_token_groups: Sequence[Mapping[str, Any]] | None) -> str:
    if reserved_token_groups is None:
        return "null"
    try:
        return json.dumps(
            list(reserved_token_groups),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("mharmony reserved token groups must be JSON-serializable mappings.") from exc


def _qwen35_vocab_cache_key(encoding_name: str) -> str | None:
    if encoding_name not in {"QWEN35_HARMONY", "Qwen35Harmony"}:
        return None
    vocab_path = os.environ.get("QWEN35_VOCAB_PATH")
    if not vocab_path:
        return None
    return str(Path(vocab_path).expanduser().resolve())


@functools.lru_cache(maxsize=16)
def _load_mharmony_encoding_cached(
    encoding_name: str,
    reserved_groups_key: str,
    _qwen35_vocab_path: str | None,
):
    # qwen35_vocab_path is deliberately part of the cache identity. The native
    # loader reads QWEN35_VOCAB_PATH itself, but two authenticated checkpoints
    # can provide different checkpoint-local vocabularies in one process.
    groups = json.loads(reserved_groups_key)
    return load_mharmony().load_harmony_encoding(
        encoding_name,
        reserved_token_groups=groups,
    )


def load_mharmony_encoding(
    encoding_name: Any,
    *,
    reserved_token_groups: Sequence[Mapping[str, Any]] | None = None,
):
    """Load and cache one standalone Harmony encoding configuration."""
    encoding_name_value = _encoding_name_value(encoding_name)
    return _load_mharmony_encoding_cached(
        encoding_name_value,
        _reserved_groups_cache_key(reserved_token_groups),
        _qwen35_vocab_cache_key(encoding_name_value),
    )


def assert_mharmony_available() -> None:
    """Fail early when the standalone package lacks ISAAC's public surface."""
    try:
        mharmony = load_mharmony()
        missing = [name for name in _REQUIRED_PUBLIC_API if not hasattr(mharmony, name)]
        if missing:
            raise AttributeError(f"missing public API: {', '.join(missing)}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Perceptron ISAAC requires the standalone mharmony package with its public "
            "conversation, rendering, and Qwen3.5 image-processing APIs. "
            f"{exc}"
        ) from exc


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _local_role(role: Any) -> str | None:
    value = _enum_value(role)
    if value is None:
        return None
    value = str(value)
    return _ROLE_ALIASES.get(value, value)


def _event_time(raw_time: Any) -> tuple[float, float]:
    if not isinstance(raw_time, Sequence) or isinstance(raw_time, str | bytes) or len(raw_time) != 2:
        raise ValueError(f"mharmony event time must contain exactly two values, got {raw_time!r}.")
    return (float(raw_time[0]), float(raw_time[1]))


def _event_tags(raw_event: Mapping[str, Any]) -> dict[str, Any]:
    raw_tags = raw_event.get("tags")
    if raw_tags is None:
        tags: dict[str, Any] = {}
    elif isinstance(raw_tags, Mapping):
        tags = dict(raw_tags)
    else:
        raise ValueError(f"mharmony event tags must be a mapping, got {type(raw_tags).__name__}.")
    channel = raw_event.get("channel")
    if channel is not None:
        tags["channel"] = str(_enum_value(channel))
    return tags


def _event_type(modality: Any, tags: Mapping[str, Any]):
    modality_name = str(_enum_value(modality))
    if modality_name == "text":
        text_type = str(_enum_value(tags.get(_TEXT_TYPE_TAG, "text")))
        try:
            return _TEXT_TYPES[text_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported ISAAC text event type {text_type!r}.") from exc
    if modality_name in _TEXT_TYPES:
        return _TEXT_TYPES[modality_name]
    if modality_name in _CONTROL_MODALITIES:
        return TextType.control
    if modality_name == "vector":
        return VectorType.vector
    if modality_name == "image_frame":
        return VisionType.I
    raise ValueError(f"Unsupported ISAAC mharmony modality {modality_name!r}.")


def _event_data(raw_event: Mapping[str, Any]) -> Mapping[str, Any]:
    data = raw_event.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("mharmony event data must be a mapping.")
    return data


def _tokens_event(
    raw_event: Mapping[str, Any],
    *,
    event_type: TextType,
    time: tuple[float, float],
    role: str | None,
    tags: dict[str, Any],
) -> Event:
    data = _event_data(raw_event)
    if "Tokens" not in data:
        raise ValueError("mharmony text/control event is missing its Tokens payload.")
    tokens = torch.as_tensor(data["Tokens"], dtype=torch.long).reshape(-1)
    return Event.from_text_tokens(tokens, time=time, type=event_type, role=role, tags=tags)


def _shaped_payload(
    raw_event: Mapping[str, Any],
    payload_name: str,
    *,
    allow_scalar: bool,
) -> torch.Tensor:
    data = _event_data(raw_event)
    payload = data.get(payload_name)
    if not isinstance(payload, Mapping) or "data" not in payload or "shape" not in payload:
        raise ValueError(f"mharmony {payload_name} event is missing data or shape.")
    raw_shape = payload["shape"]
    if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, str | bytes):
        raise ValueError(f"mharmony {payload_name} shape must be a sequence.")
    shape = [int(dim) for dim in raw_shape]
    if (not shape and not allow_scalar) or any(dim <= 0 for dim in shape):
        raise ValueError(f"mharmony {payload_name} shape must contain positive dimensions, got {shape}.")
    tensor = torch.as_tensor(payload["data"], dtype=torch.float32)
    expected_values = math.prod(shape)
    if tensor.numel() != expected_values:
        raise ValueError(
            f"mharmony {payload_name} contains {tensor.numel()} values for declared shape {shape}."
        )
    return tensor.reshape(shape)


def _vector_event(
    raw_event: Mapping[str, Any],
    *,
    time: tuple[float, float],
    role: str | None,
    tags: dict[str, Any],
) -> Event:
    vector = _shaped_payload(raw_event, "Vector", allow_scalar=True)
    if vector.ndim == 0:
        vector = vector.reshape(1, 1)
    elif vector.ndim == 1:
        vector = vector.unsqueeze(0)
    elif vector.ndim > 2:
        vector = vector.reshape(-1, vector.shape[-1])
    token_count = int(vector.shape[0])
    return Event(
        data=vector,
        time=time,
        type=VectorType.vector,
        role=role,
        dims_virtual=[token_count],
        dims_real=list(vector.shape),
        idx_range=(0, token_count),
        tags=tags,
    )


def _image_event(
    raw_event: Mapping[str, Any],
    *,
    time: tuple[float, float],
    role: str | None,
    tags: dict[str, Any],
    pixel_shuffle_scale: int,
) -> Event:
    image = _shaped_payload(raw_event, "Image", allow_scalar=False)
    if image.ndim != 3:
        raise ValueError(
            "mharmony Image shape must be [patch_rows, patch_columns, patch_dimension], "
            f"got {list(image.shape)}."
        )
    dims_real = [1, int(image.shape[0]), int(image.shape[1])]
    if any(dim % pixel_shuffle_scale for dim in dims_real[1:]):
        raise ValueError(
            f"mharmony Image patch grid {dims_real[1:]} is not divisible by "
            f"pixel_shuffle_scale={pixel_shuffle_scale}."
        )
    dims_virtual = [
        dims_real[0],
        dims_real[1] // pixel_shuffle_scale,
        dims_real[2] // pixel_shuffle_scale,
    ]
    return Event(
        data=image.reshape(-1, image.shape[-1]),
        time=time,
        type=VisionType.I,
        role=role,
        dims_virtual=dims_virtual,
        dims_real=dims_real,
        idx_range=(0, math.prod(dims_virtual[1:])),
        tags=tags,
    )


def _pixel_shuffle_scale(rendered: Mapping[str, Any]) -> int:
    config = rendered.get("config") or {}
    if not isinstance(config, Mapping):
        raise ValueError("mharmony RenderedStream config must be a mapping.")
    image_config = config.get("image") or {}
    if not isinstance(image_config, Mapping):
        raise ValueError("mharmony RenderedStream image config must be a mapping.")
    raw_scale = image_config.get("pixel_shuffle_scale") or 1
    if isinstance(raw_scale, bool):
        raise ValueError("mharmony pixel_shuffle_scale must be a positive integer.")
    scale = int(raw_scale)
    if scale <= 0 or scale != raw_scale:
        raise ValueError("mharmony pixel_shuffle_scale must be a positive integer.")
    return scale


def rendered_stream_to_local_tensor_stream(
    rendered: Mapping[str, Any],
    *,
    device: Any,
    dtype: Any,
) -> TensorStream:
    """Lower the public RenderedStream dictionary into ISAAC's local tensor IR."""
    raw_stream = rendered.get("stream")
    if not isinstance(raw_stream, Mapping):
        raise ValueError("mharmony RenderedStream is missing its stream mapping.")
    raw_events = raw_stream.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, str | bytes):
        raise ValueError("mharmony RenderedStream events must be a sequence.")

    pixel_shuffle_scale = _pixel_shuffle_scale(rendered)
    events: list[Event] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            raise ValueError("mharmony RenderedStream events must be mappings.")
        tags = _event_tags(raw_event)
        event_type = _event_type(raw_event.get("modality"), tags)
        time = _event_time(raw_event.get("time"))
        role = _local_role(raw_event.get("role"))
        if isinstance(event_type, TextType):
            event = _tokens_event(
                raw_event,
                event_type=event_type,
                time=time,
                role=role,
                tags=tags,
            )
        elif event_type == VectorType.vector:
            event = _vector_event(raw_event, time=time, role=role, tags=tags)
        elif event_type == VisionType.I:
            event = _image_event(
                raw_event,
                time=time,
                role=role,
                tags=tags,
                pixel_shuffle_scale=pixel_shuffle_scale,
            )
        else:  # pragma: no cover - the exhaustive mapper rejects this first
            raise ValueError(f"Unsupported ISAAC event type {event_type!r}.")
        events.append(event)

    stream = create_stream(events, priority=list(_ISAAC_PRIORITY), schedule=False)
    return TensorStream([stream]).to(device=device, dtype=dtype, non_blocking=True)
