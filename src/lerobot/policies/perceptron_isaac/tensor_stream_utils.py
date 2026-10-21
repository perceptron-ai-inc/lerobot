# ruff: noqa
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from .tensor_stream_mrope import compute_mrope_pos_tensor_common
from .tensor_stream import (
    FLOW_ACTION_TEXT_TYPE,
    Event,
    ModalityType,
    Stream,
    TensorStream,
    TextType,
    create_stream,
    group_streams,
)

ACTION_CONTEXT_EXCLUDED_TYPES = frozenset({TextType.action, FLOW_ACTION_TEXT_TYPE})
ACTION_CONTEXT_EXCLUDED_TYPE_VALUES = tuple(
    action_type.value for action_type in ACTION_CONTEXT_EXCLUDED_TYPES
)


@dataclass(frozen=True)
class TensorStreamMetadata:
    modality_mask: torch.Tensor
    role_mask: torch.Tensor
    interleave_partial_masks: dict[ModalityType, torch.Tensor]

    def partial_mask(self, mod_type: ModalityType) -> torch.Tensor:
        mask = self.interleave_partial_masks.get(mod_type)
        if mask is not None:
            return mask
        return torch.empty((0,), dtype=torch.bool, device=self.modality_mask.device)

    def _map_tensors(self, tensor_map: Callable[[torch.Tensor], torch.Tensor]) -> "TensorStreamMetadata":
        return TensorStreamMetadata(
            modality_mask=tensor_map(self.modality_mask),
            role_mask=tensor_map(self.role_mask),
            interleave_partial_masks={
                mod_type: tensor_map(mask) for mod_type, mask in self.interleave_partial_masks.items()
            },
        )

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = True,
    ) -> "TensorStreamMetadata":
        target_device = torch.device(device)
        if self.modality_mask.device == target_device:
            return self
        return self._map_tensors(lambda tensor: tensor.to(device=target_device, non_blocking=non_blocking))

    def pin_memory(self) -> "TensorStreamMetadata":
        if self.modality_mask.device.type != "cpu":
            return self
        return self._map_tensors(lambda tensor: tensor.pin_memory())


def _compute_event_mask_uncached(
    ts: TensorStream,
    tag_fn: Callable[[Event], int | None],
    default: int = -1,
) -> torch.Tensor:
    batch_size, seq_len = ts.shape
    device = ts.device or torch.device("cpu")
    mask = torch.full((batch_size, seq_len), default, dtype=torch.long, device=device)

    for batch_idx, stream in enumerate(ts.streams):
        seq_offset = 0
        for event in stream:
            token_count = event.num_tokens()
            if token_count == 0:
                continue
            label = tag_fn(event)
            mask[batch_idx, seq_offset : seq_offset + token_count] = default if label is None else label
            seq_offset += token_count
        if seq_offset != seq_len:
            raise ValueError(
                f"TensorStream stream {batch_idx} covered {seq_offset} tokens while shape expects {seq_len}."
            )

    return mask


def _spans_to_partial_mask(
    spans: list[tuple[int, int, int]], total: int, device: torch.device
) -> torch.Tensor:
    partial_mask = torch.zeros(total, dtype=torch.bool, device=device)
    offset = 0
    for full_token_count, idx_start, idx_end in spans:
        if idx_end > idx_start:
            partial_mask[offset + idx_start : offset + idx_end] = True
        offset += full_token_count
    return partial_mask


def _compute_interleave_partial_mask_uncached(ts: TensorStream, mod_type: ModalityType) -> torch.Tensor:
    device = ts.device or torch.device("cpu")
    spans: list[tuple[int, int, int]] = []
    total = 0
    for stream in ts.streams:
        for ev in stream:
            if ev.type != mod_type:
                continue
            full_token_count = ev.num_tokens(partial=False)
            spans.append((full_token_count, ev.idx_range[0], ev.idx_range[1]))
            total += full_token_count
    return _spans_to_partial_mask(spans, total, device)


def interleave_partial_mask(
    ts: TensorStream,
    mod_type: ModalityType,
    metadata: TensorStreamMetadata | None = None,
) -> torch.Tensor:
    if metadata is not None:
        return metadata.partial_mask(mod_type)
    return _compute_interleave_partial_mask_uncached(ts, mod_type)


def compute_tensor_stream_metadata(ts: TensorStream) -> TensorStreamMetadata:
    batch_size, seq_len = ts.shape
    device = ts.device or torch.device("cpu")
    modality = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
    roles = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
    partial_mask_spans: dict[ModalityType, list[tuple[int, int, int]]] = {}

    for batch_idx, stream in enumerate(ts.streams):
        seq_offset = 0
        for event in stream:
            full_token_count = event.num_tokens(partial=False)
            idx_start, idx_end = event.idx_range
            partial_mask_spans.setdefault(event.type, []).append((full_token_count, idx_start, idx_end))

            token_count = event.num_tokens()
            if token_count == 0:
                continue

            modality[batch_idx, seq_offset : seq_offset + token_count] = event.type.value
            roles[batch_idx, seq_offset : seq_offset + token_count] = ROLE_TO_IDX.get(event.role, -1)
            seq_offset += token_count
        if seq_offset != seq_len:
            raise ValueError(
                f"TensorStream stream {batch_idx} covered {seq_offset} tokens while shape expects {seq_len}."
            )

    partial_masks: dict[ModalityType, torch.Tensor] = {}
    for mod_type, spans in partial_mask_spans.items():
        total = sum(full_token_count for full_token_count, _, _ in spans)
        partial_masks[mod_type] = _spans_to_partial_mask(spans, total, device)

    return TensorStreamMetadata(
        modality_mask=modality,
        role_mask=roles,
        interleave_partial_masks=partial_masks,
    )


def resolve_tensor_stream_metadata(
    tensor_stream: TensorStream,
    precomputed: TensorStreamMetadata | None,
    *,
    force_recompute: bool = False,
) -> TensorStreamMetadata:
    """Return precomputed metadata (moved to the TensorStream device) or compute fresh."""
    if force_recompute or precomputed is None:
        return compute_tensor_stream_metadata(tensor_stream)
    return precomputed.to(device=tensor_stream.device, non_blocking=True)


def compute_mrope_pos_tensor(
    ts: TensorStream,
    n_pos_dims: int = 3,
) -> torch.Tensor:
    """
    Create a (batch, T, n_pos_dims) position tensor in one sweep.
    The first dim is the running “time” index, the rest are spatial (or 1-fillers).

    Args:
        ts         : TensorStream
        n_pos_dims : total coordinate dimensions (default 3)

    Returns:
        torch.LongTensor  - shape (batch_size, seq_len, n_pos_dims)
    """

    return compute_mrope_pos_tensor_common(ts, n_pos_dims=n_pos_dims)


def modality_mask(ts: TensorStream, metadata: TensorStreamMetadata | None = None) -> torch.Tensor:
    if metadata is not None:
        return metadata.modality_mask
    return _compute_event_mask_uncached(ts, lambda ev: ev.type.value)


ACTION_CONTEXT_PROVENANCE_KEYS = ("source_stream_name", "shard_id", "shard_sample_id")


def first_event_start_indices(
    ts: TensorStream,
    event_type: ModalityType,
    *,
    fallback_start: int | None = None,
) -> list[int]:
    """Return the first token offset for `event_type` in each sub-stream.

    If a stream has no matching event, the fallback mirrors
    `build_action_context_mask` inference behavior: use the end of the model
    sequence so the whole non-action prefix remains visible.
    """
    if fallback_start is None:
        fallback_start = ts.shape[1]
    starts: list[int] = []
    for stream in ts.streams:
        seq_offset = 0
        event_start = int(fallback_start)
        for event in stream.events:
            if event.type == event_type:
                event_start = seq_offset
                break
            seq_offset += event.num_tokens()
        starts.append(event_start)
    return starts


def action_event_provenance_key(
    event: Event,
    provenance_keys: Iterable[str] = ACTION_CONTEXT_PROVENANCE_KEYS,
) -> tuple[object, ...] | None:
    tags = event.tags or {}
    keys = tuple(provenance_keys)
    if not any(key in tags for key in keys):
        return None
    return tuple(tags.get(key) for key in keys)


def build_action_context_mask(
    ts: TensorStream,
    metadata: TensorStreamMetadata | None = None,
    *,
    action_batch_indices: Iterable[int] | None = None,
    action_start_indices: Iterable[int] | None = None,
    action_provenance_keys: Iterable[tuple[object, ...] | None] | None = None,
    l_model: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    """Build per-action masks over VLM context tokens.

    Each row corresponds to one action query/chunk and marks non-action tokens before
    that action position. When a provenance key is supplied, only tokens from the same
    packed document are visible; without provenance, the mask falls back to same-stream
    prefix behavior. If no action positions are supplied, each stream gets one row with
    action_start=L, which is the standard single-sample inference path.
    """
    if l_model is None:
        l_model = metadata.modality_mask.shape[1] if metadata is not None else ts.shape[1]
    if device is None:
        device = metadata.modality_mask.device if metadata is not None else ts.device or torch.device("cpu")

    def _as_int_list(xs: Iterable[int] | None, default: Iterable[int]) -> list[int]:
        return list(default) if xs is None else [int(x) for x in xs]

    batch_indices = _as_int_list(action_batch_indices, range(len(ts.streams)))
    action_starts = _as_int_list(action_start_indices, [l_model] * len(batch_indices))
    provenance_keys = (
        list(action_provenance_keys) if action_provenance_keys is not None else [None] * len(batch_indices)
    )

    if not (len(batch_indices) == len(action_starts) == len(provenance_keys)):
        raise ValueError(
            "action_batch_indices, action_start_indices, and action_provenance_keys must have the same length"
        )

    if all(action_key is None for action_key in provenance_keys):
        if len(batch_indices) == 0:
            return torch.zeros(0, l_model, dtype=torch.bool, device=device), 0
        mod_mask = modality_mask(ts, metadata=metadata).to(device=device)
        mod_width = min(l_model, mod_mask.shape[1])
        batch_idx_tensor = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
        action_start_tensor = torch.as_tensor(action_starts, device=device, dtype=torch.long)
        position_idx = torch.arange(l_model, device=device).unsqueeze(0)
        prefix_mask = position_idx < action_start_tensor.unsqueeze(1)
        non_action_mask = torch.zeros(len(batch_indices), l_model, dtype=torch.bool, device=device)
        action_mask = mod_mask.index_select(0, batch_idx_tensor)[:, :mod_width]
        excluded_mask = torch.zeros_like(action_mask, dtype=torch.bool)
        for action_type_value in ACTION_CONTEXT_EXCLUDED_TYPE_VALUES:
            excluded_mask |= action_mask == action_type_value
        non_action_mask[:, :mod_width] = ~excluded_mask
        return non_action_mask & prefix_mask, len(batch_indices)

    context_mask = torch.zeros(len(batch_indices), l_model, dtype=torch.bool, device=device)
    context_without_provenance = 0
    for row_idx, (batch_idx, action_start, action_key) in enumerate(
        zip(batch_indices, action_starts, provenance_keys, strict=True)
    ):
        if action_key is None:
            context_without_provenance += 1
        seq_offset = 0
        for event in ts.streams[batch_idx].events:
            event_start = seq_offset
            event_end = seq_offset + event.num_tokens()
            seq_offset = event_end
            # Events are in stream order, so seq_offset is monotonically increasing.
            if event_start >= action_start or event_start >= l_model:
                break
            if event.type in ACTION_CONTEXT_EXCLUDED_TYPES:
                continue
            if action_key is not None and action_event_provenance_key(event) != action_key:
                continue
            span_start = max(event_start, 0)
            span_end = min(event_end, action_start, l_model)
            if span_end > span_start:
                context_mask[row_idx, span_start:span_end] = True

    return context_mask, context_without_provenance


ROLE_TO_IDX = {
    None: -1,
    "": -1,
    "agent": 0,
    "assistant": 0,  # mharmony uses "assistant" for agent role
    "user": 1,
    "system": 2,
    # … add more if you like
}


def role_mask(ts: TensorStream, metadata: TensorStreamMetadata | None = None) -> torch.Tensor:
    if metadata is not None:
        return metadata.role_mask
    return _compute_event_mask_uncached(ts, lambda ev: ROLE_TO_IDX.get(ev.role, -1))


def reconstruct_tensor_stream_from_compact_dict(
    ts: TensorStream, compact_dict: dict[ModalityType, torch.Tensor]
) -> TensorStream:
    streams = []
    for stream in ts.streams:
        event_list = []
        for event in stream:
            new_event = event.shallow_copy()
            new_event.data = compact_dict[event.type][event.idx_range[0] : event.idx_range[1]]
            compact_dict[event.type] = compact_dict[event.type][event.num_tokens(partial=False) :]
            event_list.append(new_event)
        streams.append(Stream(event_list, priority=stream.priority))
    return TensorStream(streams)


def set_data(
    tensor_stream: TensorStream,
    stream_types: Iterable[ModalityType],
    roles: Iterable[str | None] | None = None,
    metadata: TensorStreamMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gathers data from a TensorStream according to the given stream types
    and returns (data, mask) where 'data' has valid entries for
    each requested stream type and 'mask' indicates which elements
    in 'data' are valid.

    NOTE: Currently assumes stream_types are text-based types, but can be extended.

    Args:
        tensor_stream (TensorStream):
            The input TensorStream which contains data for multiple modalities.
        stream_types (Iterable[ModalityType]):
            A list or iterable of modality types (e.g., TextType, VisionType, etc.)
            to retrieve from the TensorStream.
        roles (Iterable[str | None] | None, optional):
            Roles to include. If None, all roles in ROLE_TO_IDX are included.
        metadata (TensorStreamMetadata | None, optional):
            Pre-computed metadata for fast lookups.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - data: A tensor of the same shape as the internal metadata shape,
              containing valid entries from the given stream types.
            - mask: A boolean tensor of the same shape, where True indicates
              the corresponding element in 'data' is valid/used.
    """
    # Retrieve indexing and shape metadata
    st_tensor = modality_mask(tensor_stream, metadata=metadata)  # (B, T) modality-ids
    roles_tensor = role_mask(tensor_stream, metadata=metadata)  # (B, T) role-ids

    # Create output data placeholders on the same device
    device = st_tensor.device
    data = torch.zeros_like(st_tensor, device=device)
    set_data_mask = torch.zeros_like(st_tensor, dtype=torch.bool, device=device)
    per_modality_stream = group_streams(
        tensor_stream.flat_stream(), group_fn=lambda ev: ev.type, schedule=False
    )
    requested_stream_types = tuple(stream_types)
    per_modality_compact_stream = {
        st: per_modality_stream[st].compact() for st in requested_stream_types if st in per_modality_stream
    }

    # Fill 'data' and 'set_data_mask' for each requested stream type
    role_filter = tuple(ROLE_TO_IDX.keys()) if roles is None else tuple(roles)

    for st in requested_stream_types:
        if st not in per_modality_stream:
            continue
        data_mask = st_tensor == st.value
        partial_mask = interleave_partial_mask(tensor_stream, st, metadata=metadata)
        data[data_mask] = per_modality_compact_stream[st].reshape(-1)[partial_mask]

        roles_mask = torch.zeros_like(st_tensor, dtype=torch.bool, device=device)
        for role in role_filter:
            roles_mask |= roles_tensor == ROLE_TO_IDX[role]
        data_mask = data_mask & roles_mask
        set_data_mask[data_mask] = True

    return data, set_data_mask


def slice(tensor_stream: TensorStream, start: int, end: int) -> TensorStream:
    """
    Return a new TensorStream that contains *only* the tokens in the
    half-open interval ``[start, end)`` (0-based, inclusive-exclusive).
    """
    B, T = tensor_stream.shape
    assert 0 <= start <= end <= T, f"slice [{start}, {end}) is out of bounds for sequence length {T}"

    sliced_streams: list[Stream] = []

    for stream in tensor_stream.streams:
        # current position in tensor stream token dims
        curr_global_index = 0
        new_events: list[Event] = []

        # iterate over each of the events in the stream only selecting
        # the events that fall within the range
        for ev in stream:
            ev_len = ev.num_tokens()

            # ev_start, ev_end are the start and end indices of the
            # event within the tensor stream token dim
            global_ev_start, global_ev_end = curr_global_index, curr_global_index + ev_len

            if global_ev_end <= start:
                # The event occurs before the start skip it and move the cursor
                # forward
                curr_global_index = global_ev_end
                continue
            if global_ev_start >= end:
                # event occurs after the end we can exit
                break

            # only consider the part of the event that falls within the range
            keep_from = max(0, start - global_ev_start)
            keep_to = min(ev_len, end - global_ev_start)
            part = ev.shallow_copy()

            if keep_from == 0 and keep_to == ev_len:
                # Event lies wholly inside the slice
                new_events.append(part)
            else:
                # Partial overlap → trim.
                assert ev.is_measured

                # update the local event ranges for the slices
                sliced_event_start = part.idx_range[0] + keep_from
                sliced_event_end = part.idx_range[0] + keep_to
                part.slice_tokens(sliced_event_start, sliced_event_end)
                new_events.append(part)

            curr_global_index = global_ev_end

        sliced_streams.append(create_stream(new_events, stream.priority, schedule=False))

    return TensorStream(sliced_streams)
