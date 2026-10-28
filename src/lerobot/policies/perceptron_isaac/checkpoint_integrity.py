"""Dependency-light primitives for checkpoint integrity verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self


@dataclass
class DirectorySnapshot:
    """Private directory copy retained until its owner calls ``cleanup``."""

    root: Path
    _temporary: tempfile.TemporaryDirectory

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        self._temporary.cleanup()


def is_hf_snapshot_blob_link(path: str | Path, *, snapshot_root: str | Path | None = None) -> bool:
    """Return whether a file symlink stays inside its HF snapshot repository's blob store."""
    candidate = Path(path)
    if not candidate.is_symlink():
        return False
    try:
        resolved = candidate.resolve(strict=True)
        expected_snapshot = Path(snapshot_root).resolve(strict=True) if snapshot_root is not None else None
    except (OSError, RuntimeError):
        return False
    if not resolved.is_file():
        return False
    for ancestor in candidate.parents:
        if ancestor.parent.name != "snapshots":
            continue
        if expected_snapshot is not None and ancestor.resolve() != expected_snapshot:
            return False
        blob_root = (ancestor.parent.parent / "blobs").resolve()
        return resolved.is_relative_to(blob_root)
    return False


def snapshot_directory(
    source: str | Path,
    *,
    prefix: str = "lerobot-checkpoint-",
    root_entries: Iterable[str] | None = None,
    directory_entries: Mapping[str, Iterable[str]] | None = None,
) -> DirectorySnapshot:
    """Copy a directory into private storage, optionally selecting root entries."""
    source_root = Path(source).resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError(f"Checkpoint snapshot source is not a directory: {source_root}.")
    selected = frozenset(root_entries) if root_entries is not None else None
    if selected is not None and any(Path(entry).name != entry or not entry for entry in selected):
        raise ValueError("Checkpoint snapshot root entries must be non-empty basenames.")
    selected_by_directory = {
        Path(relative).as_posix(): frozenset(entries)
        for relative, entries in (directory_entries or {}).items()
    }
    if any(
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or any(Path(entry).name != entry or not entry for entry in entries)
        for relative, entries in selected_by_directory.items()
    ):
        raise ValueError("Checkpoint snapshot directory selections must use safe relative basenames.")

    def ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory)
        relative = directory_path.relative_to(source_root).as_posix()
        selected_here = selected if relative == "." else selected_by_directory.get(relative)
        ignored = set(names) - selected_here if selected_here is not None else set()
        for name in set(names) - ignored:
            candidate = directory_path / name
            # Accepted Hub links are dereferenced into the private copy, whose
            # bytes are verified before use. Every other file/directory link is
            # rejected here, preventing cycles and unbounded external traversal.
            if candidate.is_symlink() and not is_hf_snapshot_blob_link(candidate):
                raise ValueError(f"Checkpoint snapshot rejects non-HF symlink: {candidate}.")
        return ignored

    temporary = tempfile.TemporaryDirectory(prefix=prefix)
    snapshot_root = Path(temporary.name) / "snapshot"
    try:
        shutil.copytree(source_root, snapshot_root, ignore=ignore)
    except BaseException:
        temporary.cleanup()
        raise
    return DirectorySnapshot(snapshot_root.resolve(), temporary)


def canonical_json_sha256(value: Any) -> str:
    """Hash the canonical compact JSON encoding used by checkpoint manifests."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object, rejecting duplicate keys at every depth."""
    json_path = Path(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{json_path} contains duplicate key {key!r} in JSON.")
            result[key] = value
        return result

    try:
        payload = json.loads(
            json_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint JSON {json_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint JSON {json_path} must contain an object.")
    return payload


def is_lowercase_sha256(value: Any) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_sha256(value: Any, *, context: str) -> str:
    """Return a validated lowercase SHA-256 digest."""
    if not is_lowercase_sha256(value):
        raise ValueError(f"{context} is not a lowercase SHA-256 digest.")
    return value


def verify_file_sha256(path: str | Path, expected: Any, *, context: str) -> str:
    """Verify a file against a declared digest and return the actual digest."""
    file_path = Path(path)
    expected_digest = require_sha256(expected, context=f"{context} digest for {file_path.name!r}")
    if not file_path.is_file():
        raise ValueError(f"{context} is missing {file_path.name!r} at {file_path}.")
    actual = file_sha256(file_path)
    if not hmac.compare_digest(actual, expected_digest):
        raise ValueError(
            f"{context} digest mismatch for {file_path.name}: expected {expected_digest}, found {actual}."
        )
    return actual
