"""Exclusive process locks for local accelerator deployment."""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch


def _physical_device_key(device_index: int) -> str:
    """Map a logical CUDA index onto a host-stable key.

    ``cuda:0`` means a different physical GPU under different ``CUDA_VISIBLE_DEVICES``
    mappings, and the same physical GPU can be spelled by index or by UUID, so the key must
    resolve to the device itself. The GPU UUID (via torch, which already applies the env
    mapping) is the only spelling-independent identity; the env-derived fallback covers
    CUDA-less hosts, where no real device contention exists.
    """
    try:
        device_uuid = torch.cuda.get_device_properties(device_index).uuid
    except Exception:
        device_uuid = None
    if device_uuid is not None:
        return f"GPU-{device_uuid}"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return str(device_index)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if device_index >= len(entries):
        return str(device_index)
    # Entries are physical indices or GPU-UUID strings; keep the filename well-formed.
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in entries[device_index])


@contextmanager
def exclusive_torch_device_lock(
    device: str | torch.device | None,
    *,
    enabled: bool = True,
    lock_dir: str | Path | None = None,
) -> Iterator[Path | None]:
    """Hold a non-blocking process lock for a CUDA device during rollout."""
    resolved = torch.device(device or "cpu")
    if not enabled or resolved.type != "cuda":
        yield None
        return

    device_index = resolved.index if resolved.index is not None else 0
    device_key = _physical_device_key(device_index)
    root = Path(lock_dir) if lock_dir is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    # v2: the un-versioned predecessor created files with mode 0o600, which permanently
    # locked other users out of the shared path; a new name sidesteps that residue.
    path = root / f"lerobot-cuda-v2-{device_key}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    # Clear the umask around creation: the file must be born 0o666, or a crash before the
    # fchmod below would leave a file other users cannot open.
    prior_umask = os.umask(0)
    try:
        fd = os.open(path, flags, 0o666)
    except PermissionError as exc:
        raise RuntimeError(
            f"Cannot open the CUDA device lock at {path}: {exc.strerror}. The file belongs to "
            f"another user and is not world-writable; on a sticky temp dir only its owner or "
            f"root can remove it. Ask them to remove it, pass --exclusive_device_lock=false, "
            f"or pass a private lock_dir (which forfeits cross-user exclusion)."
        ) from exc
    finally:
        os.umask(prior_umask)
    try:
        # The lock is host-wide, so a second user must be able to take it. Only the creator can
        # widen the mode, and a pre-existing file owned by someone else is already usable.
        with contextlib.suppress(PermissionError):
            os.fchmod(fd, 0o666)  # nosec B103  # host-wide advisory lock: every user must open it
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            holder = os.read(fd, 128).decode(errors="replace").strip() or "unknown"
            raise RuntimeError(
                f"CUDA device {device_index} is already owned by rollout process {holder}. "
                f"Pass --exclusive_device_lock=false to share the device between rollouts."
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        yield path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
