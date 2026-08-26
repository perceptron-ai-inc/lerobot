"""Temporary import boundary between ISAAC and in-tree Genesis mharmony.

Only this module may directly import ``genesis.data.mharmony.*``.  The
standalone mharmony package planned by Genesis PR #3075 is not released yet,
so those imports intentionally tolerate mharmony's own transitive Genesis
dependencies.  Callers elsewhere in the ISAAC package use this local adapter;
they do not inspect ``sys.modules`` and do not depend on Genesis runtime APIs.
"""

from __future__ import annotations

import importlib
from typing import Any

from .tensor_stream import ALL_TYPES, Event, TensorStream, create_stream

ALLOWED_GENESIS_MHARMONY_PREFIX = "genesis.data.mharmony"


def import_mharmony_module(module_name: str):
    """Import one module from the temporary in-tree mharmony namespace."""
    allowed = module_name == ALLOWED_GENESIS_MHARMONY_PREFIX or module_name.startswith(
        f"{ALLOWED_GENESIS_MHARMONY_PREFIX}."
    )
    if not allowed:
        raise ImportError(
            f"Perceptron ISAAC may only import {ALLOWED_GENESIS_MHARMONY_PREFIX}.*, got {module_name!r}."
        )
    return importlib.import_module(module_name)


def load_mharmony_types():
    return import_mharmony_module("genesis.data.mharmony.types")


def load_mharmony_encoding_cache():
    return import_mharmony_module("genesis.data.mharmony.encoding_cache")


def load_mharmony_stream_adapter():
    return import_mharmony_module("genesis.data.mharmony.stream_adapter")


def load_qwen35_image_processor():
    return import_mharmony_module("genesis.data.mharmony.qwen35_image_processor")


def assert_mharmony_available() -> None:
    """Fail early with a boundary-specific message when mharmony is absent."""
    try:
        load_mharmony_types()
        load_mharmony_encoding_cache()
        load_mharmony_stream_adapter()
        load_qwen35_image_processor()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Perceptron ISAAC requires the temporary in-tree "
            "genesis.data.mharmony.* package and its mharmony binding."
        ) from exc


def _to_local_modality_type(modality_type: Any):
    value = int(getattr(modality_type, "value", modality_type))
    try:
        return ALL_TYPES[value]
    except IndexError as exc:
        raise ValueError(f"Unsupported ISAAC TensorStream modality type value {value}.") from exc


def _to_local_stream(stream: Any):
    priority = [_to_local_modality_type(modality_type) for modality_type in getattr(stream, "priority", [])]
    events = []
    for event in stream:
        events.append(
            Event(
                data=event.data,
                time=tuple(event.time),
                type=_to_local_modality_type(event.type),
                role=getattr(event, "role", None),
                dims_virtual=(
                    None if getattr(event, "dims_virtual", None) is None else list(event.dims_virtual)
                ),
                dims_real=(None if getattr(event, "dims_real", None) is None else list(event.dims_real)),
                idx_range=getattr(event, "idx_range", None),
                tags=dict(getattr(event, "tags", None) or {}),
            )
        )
    return create_stream(events, priority=priority, schedule=False)


def to_local_tensor_stream(stream: Any, *, device: Any, dtype: Any) -> TensorStream:
    """Convert mharmony's transitive Genesis stream into ISAAC's local value type."""
    return TensorStream([_to_local_stream(stream)]).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
