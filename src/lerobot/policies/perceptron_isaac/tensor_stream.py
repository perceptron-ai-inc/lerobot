# ruff: noqa
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import (
    Any,
    TypeAlias,
    TypeVar,
)

import torch
from torch.profiler import record_function


class ModalityType(Enum):
    """
    Base class for modality-type enumerations.
    Each derived class (VisionType, AudioType, etc.) holds
    an integer value that identifies a specific modality.

    Example usage:
        If you have an object `my_event` of class `Event`,
        you might write:
            if my_event.type == AudioType.waveform:
                # process an audio waveform

    The methods below implement ordering and hashing
    based on the integer `.value` of each enum member.
    """

    @property
    def modality(self):
        # AudioType.spectrogram.modality = AudioType
        # TODO: AudioType.modality = AudioType
        return self.__class__

    def __lt__(self, other):
        if isinstance(other, ModalityType):
            return self.value < other.value
        raise NotImplementedError()

    def __eq__(self, other):
        if isinstance(other, ModalityType):
            return self.value == other.value
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.value)


# NOTE: modality types need to be unique
class VisionType(ModalityType):
    """
    Enum for vision modalities such as video frames.
    Typically used in video processing or image sequences.

    Members:
        I: An I-frame in a video (intra-coded, more complete data).
        P: A P-frame in a video (predicted frame, partial data).
    """

    I = 0  # noqa: E741
    P = 1


class AudioType(ModalityType):
    """
    Enum for audio-related modalities.

    Members:
        waveform: Raw time-domain audio samples.
        spectrogram: Frequency-domain representation of audio.
        encodec: Some compressed audio representation (e.g. EnCodec).
    """

    waveform = 2
    spectrogram = 3
    encodec = 4


class SyntheticType(ModalityType):
    """
    Enum for "synthetic" or derived modalities, such as
    automatically generated annotations.

    Members:
        audio_transcript: Text transcript derived from audio.
        caption: Caption derived from an image/video.
        segmentation: (e.g.) semantic or instance segmentation map.
    """

    audio_transcript = 5
    caption = 6
    segmentation = 7


class TextType(ModalityType):
    """
    Enum for text or text-like tokens (e.g. subtitles, timestamps, etc.).

    Members:
        text: Actual textual tokens.
        timestamp: Special tokens representing time boundaries or intervals.
        padding: Padding tokens, often used in NLP or other sequence tasks.
    """

    text = 8
    timestamp = 9
    padding = 10
    eval = 11
    control = 12
    action = 14
    action_c = 15


FLOW_ACTION_TEXT_TYPE = TextType.action_c


# NOTE: modality types need to be unique
class VectorType(ModalityType):
    """
    Enum for vector-valued modalities (e.g. state vectors).
    """

    vector = 13


# maps idx -> type (sorted by value to maintain ALL_TYPES[t.value] == t invariant)
ALL_TYPES = sorted(
    [
        tp
        for types in [
            list(VisionType),
            list(AudioType),
            list(SyntheticType),
            list(TextType),
            list(VectorType),
        ]
        for tp in types
    ],
    key=lambda t: t.value,
)
assert all(ALL_TYPES[t.value] == t for t in ALL_TYPES), "ALL_TYPES must preserve enum value -> type lookup"


def _prod_dims(dims: list[int]) -> int:
    total = 1
    for dim in dims:
        total *= dim
    return total


_LONG_TOKEN_MODALITIES = frozenset({TextType, SyntheticType})
_BULK_LONG_TRANSFER_MIN_TENSORS = 4


# @dataclass
@dataclass(slots=True)
class Event:
    """
    Represents a single data occurrence (with a specific type, time interval, and data payload).

    Attributes:
        data (Any): The actual data payload (e.g. a torch.Tensor, a string, etc.).
        type (ModalityType): The modality type of the data (e.g., VisionType.I).
        time (Tuple[float, float]): (start_time, end_time) indicating when this Event occurs.
        role (Optional[str]): The role associated with this event (e.g., "user", "agent", "system").
            If None, the event is always included in loss calculation.

    Example usage:
        evt = Event(data=torch.randn(1, 16000),  # e.g. 1-second audio waveform
                    type=AudioType.waveform,
                    time=(0.0, 1.0),
                    role="user")
    """

    # Descriptors
    data: Any
    time: tuple[float, float]
    type: ModalityType
    role: str | None = None

    # Structure
    dims_virtual: list[int] | None = None  # virtual/processed dimensions (e.g., pixel-shuffled)
    dims_real: list[int] | None = None  # real/actual tensor dimensions
    idx_range: tuple[int, int] | None = None

    # Misc Tags (data source, shard idx, etc.)
    tags: dict = field(default_factory=dict)

    def dims(self, virtual: bool = True) -> list[int] | None:
        """
        Get the dimensions of this event.

        Args:
            virtual: If True (default), return virtual/processed dimensions (e.g., pixel-shuffled).
                    If False, return real/actual tensor dimensions.

        Returns:
            Dimensions list or None if not measured.
        """
        if virtual:
            return self.dims_virtual
        else:
            return self.dims_real

    @property
    def is_measured(self):
        return self.dims_virtual is not None

    def slice_tokens(self, start: int | None = None, end: int | None = None):
        """
        Converts into a partial event where the only valid data is between start and end indices of the flattened data
        """
        assert self.is_measured
        assert self.idx_range is not None
        assert start is not None and end is not None
        assert self.idx_range[0] <= start <= end <= self.idx_range[1]
        dims = self.dims()
        assert dims is not None
        self.idx_range = (start or 0, end or _prod_dims(dims))

    def num_tokens(self, partial=True, virtual=True) -> int:
        if not virtual:
            assert partial is False and isinstance(self.data, torch.Tensor)
            dims = self.dims(virtual=False)
            assert dims is not None
            return _prod_dims(dims)
        if partial:
            assert self.idx_range is not None
            return self.idx_range[1] - self.idx_range[0]
        dims = self.dims()
        assert dims is not None
        return _prod_dims(dims)

    def shallow_copy(self) -> Event:
        return replace(
            self,
            dims_virtual=list(self.dims_virtual) if self.dims_virtual is not None else None,
            dims_real=list(self.dims_real) if self.dims_real is not None else None,
            tags=dict(self.tags),
        )

    @classmethod
    def from_text_tokens(
        cls,
        tokens: torch.Tensor,
        *,
        time: tuple[float, float],
        type: TextType = TextType.text,
        role: str | None = None,
        tags: dict | None = None,
    ) -> Event:
        """
        Construct a text event from integer token ids.

        Contract:
        - tokens must be a torch.Tensor with an integer dtype.
        - type must be a TextType variant.
        - tokens must be 1D or 2D; 1D tensors are normalized to (n_tokens, 1).
        """
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if not isinstance(type, TextType):
            raise ValueError("type must be a TextType")
        if not isinstance(time, tuple) or len(time) != 2:
            raise ValueError("time must be a tuple of (start, end)")

        int_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        for dtype_name in ("uint16", "uint32", "uint64"):
            dtype = getattr(torch, dtype_name, None)
            if dtype is not None:
                int_dtypes.add(dtype)
        if tokens.dtype not in int_dtypes:
            raise ValueError("tokens must use an integer dtype")

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(1)
        elif tokens.dim() == 2:
            if tokens.shape[1] != 1:
                raise ValueError("2D token tensors must have shape (n_tokens, 1)")
        else:
            raise ValueError("tokens must be 1D or 2D")

        dims = list(tokens.shape)
        idx_range = (0, math.prod(dims))
        assert idx_range[0] <= idx_range[1]

        if tags is None:
            tags = {}
        else:
            tags = dict(tags)

        return cls(
            data=tokens,
            time=time,
            type=type,
            role=role,
            tags=tags,
            dims_virtual=dims,
            dims_real=dims,
            idx_range=idx_range,
        )

    def __hash__(self) -> int:
        """Hash Event based on structure, excluding data."""

        def make_hashable(obj):
            """Convert any object to hashable form."""
            if obj is None:
                return None
            elif isinstance(obj, str | int | float | bool | tuple):
                return obj
            elif isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj) if obj else None
            elif isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items())) if obj else None
            elif hasattr(obj, "value"):  # Enum types
                return obj.value
            else:
                return str(obj)  # Fallback for other types

        hash_values = []
        for fld in fields(self):
            if fld.name == "data":
                continue  # Skip tensor data

            value = getattr(self, fld.name)
            hash_values.append(make_hashable(value))

        return hash(tuple(hash_values))

    def __eq__(self, other) -> bool:
        """
        Compares two Event objects for strict equality,
        allowing for float tolerances in torch.Tensors (via torch.allclose).
        """
        if not isinstance(other, Event):
            return False

        for fld in fields(self):
            self_value = getattr(self, fld.name)
            other_value = getattr(other, fld.name)

            if fld.name == "data":
                # Special handling for tensor data with float tolerance
                if isinstance(self_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
                    if not torch.allclose(self_value, other_value):
                        return False
                else:
                    if self_value != other_value:
                        return False
            elif fld.name == "role":
                # Special handling for role: both must be None or both must be set and equal
                if (self_value is None) != (other_value is None):
                    return False
                if self_value is not None and self_value != other_value:
                    return False
            else:
                # Standard equality for all other fields
                if self_value != other_value:
                    return False

        return True


@dataclass
class Stream:
    """
    Represents an ordered sequence of Event objects, each with
    a specific ModalityType and a time range.

    Attributes:
        events (List[Event]): The list of Event objects in the stream.
        priority (List[ModalityType]): A list of modality types that define
            how we might want to reorder or prioritize events if scheduling is needed.

    Example usage:
        # Create two events of different types
        evt1 = Event(torch.zeros((3, 224, 224)), VisionType.I, (0.0, 0.04))
        evt2 = Event(torch.randn((16000,)), AudioType.waveform, (0.0, 1.0))

        # Make a stream with a given priority
        s = Stream(events=[evt1, evt2],
                   priority=[VisionType.I, AudioType.waveform])

        print(s)
    """

    events: list[Event]
    priority: list[ModalityType]  # priority of stream ordering

    def __len__(self):
        """Returns the number of Event objects in this Stream."""
        return len(self.events)

    def __getitem__(self, key: int) -> Stream | Event:
        return self.events[key]

    def __iter__(self):
        """
        Yields each Event in the Stream, enabling iteration like:
            for event in my_stream:
                ...
        """
        yield from self.events

    # --- after ------------------------------------------------------------
    @record_function("Stream.map")
    def map(
        self,
        func: Callable[[Event], dict[str, Any]],
        *,
        copy_unchanged: bool = False,  # opt-in if you really need isolation
    ) -> Stream:
        """
        Apply *func* to every event and return a new Stream.

        *func* must return a **dict of fields that actually change**.
        We create **one shallow copy** only when something changes;
        unchanged events are reused directly, which is inexpensive and
        keeps autograd graphs intact.
        """
        mapped: list[Event] = []
        for ev in self.events:
            delta = func(ev)
            if not delta:  # fast-path: nothing changes
                mapped.append(ev if not copy_unchanged else ev.shallow_copy())
                continue

            new_ev = ev.shallow_copy()  # ⚡ no tensor clone
            for k, v in delta.items():
                setattr(new_ev, k, v)
            mapped.append(new_ev)

        return create_stream(mapped, priority=self.priority, schedule=False)

    @record_function("Stream.compact")
    def compact(self) -> torch.Tensor:
        assert all([(isinstance(ev.data, torch.Tensor) and ev.is_measured) for ev in self.events]), (
            "Stream.compact only works for streams with events that have measured tensor data"
        )
        return torch.cat([ev.data for ev in self.events]).contiguous()

    def flatten(self) -> Stream:
        return self.map(lambda ev: {"data": ev.data.reshape(-1, ev.data.shape[-1])})

    def shallow_copy(self) -> Stream:
        events_copy = [ev.shallow_copy() for ev in self.events]
        return create_stream(events=events_copy, priority=self.priority, schedule=False)

    def __hash__(self) -> int:
        """Hash Stream based on structure."""
        return hash(
            (
                tuple(p.value for p in self.priority),  # Convert enums to values
                tuple(hash(event) for event in self.events),  # Use Event.__hash__
            )
        )

    def __eq__(self, other) -> bool:
        """Compare Streams structurally."""
        if not isinstance(other, Stream):
            return False

        return (
            self.priority == other.priority
            and len(self.events) == len(other.events)
            and all(e1 == e2 for e1, e2 in zip(self.events, other.events, strict=False))
        )


# TODO: implement all types of cool indexing which can happen since TensorStream assuems Event.data = Tensor
@dataclass
class TensorStream:
    streams: list[Stream]
    _device: torch.device | None = None

    def __post_init__(self):
        for stream in self.streams:
            for event in stream.events:
                assert isinstance(event.data, torch.Tensor)
                if self._device is None:
                    self._device = torch.device(event.data.device)

    # TODO: implement non-strict compaction modes
    @record_function("TensorStream.compact")
    def compact(self, mode="strict") -> torch.Tensor:
        compact_tensor_stream = torch.stack([stream.compact() for stream in self.streams]).contiguous()
        return compact_tensor_stream

    @record_function("TensorStream.map")
    def map(self, event_tf: Callable[[Event], dict[str, Any]]) -> TensorStream:
        mapped_streams = [stream.map(event_tf) for stream in self.streams]
        return TensorStream(mapped_streams)

    def flat_stream(self) -> Stream:
        if not self.streams:
            return create_stream([], priority=[], schedule=False)
        return create_stream(
            [event for stream in self.streams for event in stream],
            priority=self.streams[0].priority,
            schedule=False,
        )

    @property
    def device(self):
        return self._device

    def _bulk_move_long_token_events(
        self,
        *,
        target_device: torch.device,
        non_blocking: bool,
    ) -> set[int]:
        if target_device.type != "cuda":
            return set()

        long_token_events: list[Event] = []
        total_numel = 0
        for stream in self.streams:
            for ev in stream:
                if (
                    ev.type.modality in _LONG_TOKEN_MODALITIES
                    and isinstance(ev.data, torch.Tensor)
                    and ev.data.device.type == "cpu"
                    and ev.data.ndim == 1
                    and ev.data.is_contiguous()
                ):
                    long_token_events.append(ev)
                    total_numel += ev.data.numel()

        if len(long_token_events) < _BULK_LONG_TRANSFER_MIN_TENSORS or total_numel == 0:
            return set()

        # Many tiny token copies show up as TensorStream.to overhead in steady-state
        # profiling. Flattening only the 1D long-token case keeps semantics unchanged
        # while collapsing those H2D copies into one pinned transfer.
        flat_cpu = torch.empty(total_numel, dtype=torch.long, pin_memory=True)
        offset = 0
        lengths: list[int] = []
        original_shapes: list[torch.Size] = []
        for ev in long_token_events:
            length = ev.data.numel()
            flat_cpu[offset : offset + length].copy_(ev.data.reshape(-1))
            lengths.append(length)
            original_shapes.append(ev.data.shape)
            offset += length

        flat_gpu = flat_cpu.to(device=target_device, non_blocking=non_blocking)

        moved_event_ids: set[int] = set()
        offset = 0
        for ev, length, original_shape in zip(long_token_events, lengths, original_shapes, strict=False):
            ev.data = flat_gpu.narrow(0, offset, length).view(original_shape)
            moved_event_ids.add(id(ev))
            offset += length

        return moved_event_ids

    @property
    def shape(self):
        seq_lens = [sum([ev.num_tokens() for ev in stream]) for stream in self.streams]
        assert all([sl == seq_lens[0] for sl in seq_lens]), (
            f"each stream must have same token count to have a shape: {seq_lens}"
        )
        return (len(seq_lens), seq_lens[0])

    @record_function("TensorStream.to")
    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        non_blocking: bool = True,
    ) -> TensorStream:
        """
        Move **all** `Event.data` tensors to *device*.

        We send each tensor individually instead of the
        flatten → unflatten round-trip:

        * one async H2D copy per tensor (still overlapped when
          `pin_memory=True` is set on the DataLoader),
        * no extra host-side concat, no extra device allocation,
        * `requires_grad` flags are preserved.

        NOTE: textual & synthetic modalities are always cast
        to `torch.long`; everything else keeps its original
        dtype unless an explicit *dtype* argument is supplied.
        """
        target_device = torch.device(device)
        bulk_moved_event_ids = self._bulk_move_long_token_events(
            target_device=target_device,
            non_blocking=non_blocking,
        )

        for stream in self.streams:
            for ev in stream:
                if id(ev) in bulk_moved_event_ids:
                    continue

                # ------------------------------------------------------------------
                # Decide the dtype for *this* event.
                # ------------------------------------------------------------------
                if ev.type.modality in _LONG_TOKEN_MODALITIES:
                    tgt_dtype = torch.long
                else:
                    tgt_dtype = dtype or ev.data.dtype

                # ------------------------------------------------------------------
                # Perform the device / dtype move.
                # ------------------------------------------------------------------
                # We clone no tensor here; torch will reuse storage
                # if `dtype` and `device` are unchanged.
                moved = ev.data.to(
                    device=target_device,
                    dtype=tgt_dtype,
                    non_blocking=non_blocking,
                )

                if ev.data.requires_grad:
                    moved.requires_grad_(True)

                ev.data = moved

        # Remember where the whole TensorStream lives now.
        self._device = target_device
        return self

    @record_function("TensorStream.pin_memory")
    def pin_memory(self, non_blocking: bool = True) -> TensorStream:
        """
        Page-lock (aka *pin*) all **CPU** tensors contained in this
        `TensorStream`.  Pinned tensors make subsequent asynchronous
        H2D copies (e.g. inside `TensorStream.to("cuda")`) faster and,
        when used together with a `DataLoader(pin_memory=True)`,
        enable overlap of host-to-device transfers with GPU execution.

        The call is a no-op for tensors that are already on a CUDA /
        MPS / other non-CPU device.

        Parameters
        ----------
        non_blocking : bool, default = True
            Forwarded to `Tensor.pin_memory()`; should almost always
            stay *True* so later `to(device, non_blocking=True)` calls
            can overlap.

        Returns
        -------
        self : TensorStream
            The same object (mutated in-place) to allow call chaining.
        """
        for stream in self.streams:
            for ev in stream:
                if ev.data.device.type == "cpu":
                    # `pin_memory()` clones only when needed
                    pinned = ev.data.pin_memory()  # noqa: F841
                    # NB: pin_memory() preserves dtype/shape/grad/etc.
                    if not non_blocking:
                        # ensure the pinning work is done now
                        torch.cuda.current_stream().synchronize()  # safe on CPU too
                    ev.data = pinned

        # `_device` **stays** the same (still CPU) – no change needed
        return self

    def __hash__(self) -> int:
        """Hash TensorStream based on structure."""
        return hash(
            (
                tuple(hash(stream) for stream in self.streams),  # Use Stream.__hash__
                str(self._device) if self._device else None,
                self.shape,
            )
        )

    def __eq__(self, other) -> bool:
        """Compare TensorStreams structurally."""
        if not isinstance(other, TensorStream):
            return False

        return (
            self._device == other._device
            and self.shape == other.shape
            and len(self.streams) == len(other.streams)
            and all(s1 == s2 for s1, s2 in zip(self.streams, other.streams, strict=False))
        )


def _schedule_stream(stream: Stream) -> Stream:
    """
    Internal function that reorders (schedules) the events in a Stream
    based on the stream's priority.

    By default, this calls schedule_events(...) and reorders the events accordingly.
    The new ordering is assigned in-place to stream.events.

    Example usage (indirect):
        new_stream = _schedule_stream(old_stream)
    """
    scheduled_inds = schedule_events(stream, priority=stream.priority)
    stream.events = [stream.events[i] for i in scheduled_inds]
    return stream


def create_stream(events: list[Event], priority: list[ModalityType], schedule: bool = True) -> Stream:
    """
    Creates a new Stream with the given events and priority.
    If 'schedule' is True, the events are reordered by calling _schedule_stream.

    The events list is shallow-copied so that the returned Stream owns its
    own events container. Without this, downstream code that appends to
    `stream.events` would mutate the caller's list, leaking state across
    independent operations.

    Example usage:
        evt1 = Event(torch.zeros(10), AudioType.waveform, (0.0, 1.0))
        evt2 = Event(torch.ones(10), AudioType.waveform, (1.0, 2.0))
        my_stream = create_stream(events=[evt1, evt2],
                                  priority=[AudioType.waveform],
                                  schedule=False)
        print(my_stream)
    """
    stream = Stream(list(events), priority)
    if schedule:
        stream = _schedule_stream(stream)
    return stream


GroupKeyT = TypeVar("GroupKeyT", bound=Hashable)


def group_streams(
    stream: Stream,
    group_fn: Callable[[Event], GroupKeyT],
    schedule=True,
) -> dict[GroupKeyT, Stream]:
    """
    Splits a single Stream into multiple sub-Streams, grouped by the output of group_fn(event).

    For example, group_fn could be:
        - lambda ev: ev.type
        - lambda ev: ev.type.modality
        - lambda ev: (ev.type.modality, ev.data.shape)

    Returns:
        A dictionary mapping each group key to a Stream of events belonging to that group.
        If 'schedule' is True, each sub-Stream is scheduled via create_stream(..., schedule=True).

    Example usage:
        substreams = group_streams(my_stream, lambda ev: ev.type)
    """
    split_streams: defaultdict[GroupKeyT, list[Event]] = defaultdict(list)
    for ev in stream:
        group = group_fn(ev)
        split_streams[group].append(ev)
    for g, events in split_streams.items():
        split_streams[g] = create_stream(events, stream.priority, schedule=schedule)
    return dict(split_streams)  # type: ignore[no-matching-overload]


# Define Category for clarity
Category: TypeAlias = Any


def schedule_events(stream: Stream, priority: list[Category]) -> list[int]:
    """
    Schedule events based on their start time and priority using a topological sort algorithm.

    The priority list defines the ordering of categories.

    This function:
      1. Pairs each event with its original index.
      2. Sorts events by start time.
      3. Builds a dependency graph based on overlapping events.
      4. Uses a heap to perform a deterministic topological sort with tie-breakers.

    Raises:
        ValueError: If a cycle is detected in the events (i.e., no valid ordering exists).

    Returns:
        List[int]: A list of original indices representing the scheduled order of events.
    """
    priority_index: dict[Category, int] = {category: idx for idx, category in enumerate(priority)}

    # Pair each event metadata with its original index
    events = []
    for i, event in enumerate(stream.events):
        events.append(
            (
                i,
                event.time[0],
                event.time[1],
                event.type,
            )
        )

    sorted_events = sorted(events, key=lambda e: e[1])  # sort by start time
    num_events = len(sorted_events)

    # Build dependency graph
    graph = defaultdict(set)
    indegree = {i: 0 for i in range(num_events)}

    for i in range(num_events):
        idx_i, start_i, end_i, category_i = sorted_events[i]
        prio_i = priority_index[category_i]
        for j in range(i + 1, num_events):
            idx_j, start_j, end_j, category_j = sorted_events[j]
            if start_j >= end_i:
                break
            if end_i > start_j and end_j > start_i:
                prio_j = priority_index[category_j]
                if prio_i < prio_j:
                    graph[i].add(j)
                    indegree[j] += 1
                elif prio_i > prio_j:
                    graph[j].add(i)
                    indegree[i] += 1

    # Use heap for deterministic tie-breakers: (start_time, priority, original_index)
    heap = [
        (
            sorted_events[i][1],
            priority_index[sorted_events[i][3]],
            sorted_events[i][0],
            i,
        )
        for i in range(num_events)
        if indegree[i] == 0
    ]
    heapq.heapify(heap)
    resolved_order = []

    while heap:
        _, _, _, u = heapq.heappop(heap)
        resolved_order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(
                    heap,
                    (
                        sorted_events[v][1],
                        priority_index[sorted_events[v][3]],
                        sorted_events[v][0],
                        v,
                    ),
                )

    if len(resolved_order) != num_events:
        raise ValueError("Cycle detected in events, cannot resolve order")

    return [sorted_events[i][0] for i in resolved_order]
