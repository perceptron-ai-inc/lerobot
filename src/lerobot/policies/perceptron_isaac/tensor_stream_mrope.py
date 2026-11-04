from collections.abc import Iterable, Sequence
from typing import Protocol

import torch


class _EventLike(Protocol):
    idx_range: tuple[int, int]

    def dims(self) -> list[int] | None: ...


class _TensorStreamLike(Protocol):
    streams: Sequence[Iterable[_EventLike]]
    shape: tuple[int, int]
    device: torch.device


def compute_mrope_pos_tensor_common(ts: _TensorStreamLike, n_pos_dims: int = 3) -> torch.Tensor:
    """
    Create a (batch, T, n_pos_dims) position tensor in one sweep.
    The first dim is the running "time" index, the rest are spatial (or 1-fillers).
    """
    if n_pos_dims < 1:
        raise ValueError(f"n_pos_dims must be >= 1, got {n_pos_dims}")

    bsz, seq_len = ts.shape
    positions = torch.empty((bsz, seq_len, n_pos_dims), dtype=torch.long, device=ts.device)

    for batch_idx, stream in enumerate(ts.streams):  # one stream == one batch sample
        cumulative_offset = 0
        seq_offset = 0

        for event in stream:
            raw_dims = event.dims() or [1]
            if len(raw_dims) > n_pos_dims:
                raise ValueError(
                    f"event dims length ({len(raw_dims)}) exceeds n_pos_dims ({n_pos_dims}); "
                    "higher-rank MRoPE events are unsupported"
                )
            dims = raw_dims + [1] * (n_pos_dims - len(raw_dims))
            start, end = event.idx_range
            token_count = end - start
            if token_count == 0:
                cumulative_offset += max(dims)
                continue

            token_indices = torch.arange(start, end, dtype=torch.long, device=ts.device)
            coords = torch.empty((token_count, n_pos_dims), dtype=torch.long, device=ts.device)

            stride = 1
            for dim_idx in range(n_pos_dims - 1, -1, -1):
                dim = dims[dim_idx]
                coords[:, dim_idx] = cumulative_offset + (token_indices // stride) % dim
                stride *= dim

            positions[batch_idx, seq_offset : seq_offset + token_count] = coords
            seq_offset += token_count
            cumulative_offset += max(dims)

    return positions
