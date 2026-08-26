"""Pinned FAST action-processor artifacts for native ISAAC training."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint_integrity import is_hf_snapshot_blob_link

DEFAULT_FAST_PROCESSOR_REPOSITORY = "physical-intelligence/fast"
DEFAULT_FAST_PROCESSOR_REVISION = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
DEFAULT_FAST_PROCESSOR_TREE_SHA256 = "127eb029e5acb8242c7bc2c8efcea6c5dd6ffb565353b5c689b8c8ac55f33a8a"
DEFAULT_FAST_PROCESSOR_FILES = (
    "processing_action_tokenizer.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class FastProcessorArtifact:
    path: Path
    tree_sha256: str
    file_count: int


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def fast_processor_tree_identity(path: str | Path) -> tuple[str, int]:
    """Hash every regular file in a self-contained processor tree."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"FAST processor root must be a real directory: {root}.")
    records: list[tuple[str, str, int]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink() and not is_hf_snapshot_blob_link(candidate):
            raise ValueError(f"FAST processor package must not contain symlinks: {candidate}.")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"Unsupported FAST processor artifact path: {candidate}.")
        digest = hashlib.sha256()
        size = 0
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        records.append((candidate.relative_to(root).as_posix(), digest.hexdigest(), size))
    if not records:
        raise ValueError(f"FAST processor artifact contains no files: {root}.")
    return hashlib.sha256(_canonical_bytes(records)).hexdigest(), len(records)


def materialize_fast_processor_artifact(source: str | Path, destination: str | Path) -> FastProcessorArtifact:
    """Copy a Hub snapshot (including file symlinks) into a movable real-file tree."""
    source_root = Path(source)
    destination_root = Path(destination)
    if not source_root.is_dir():
        raise FileNotFoundError(f"FAST processor source does not exist: {source_root}.")
    if destination_root.exists():
        raise FileExistsError(f"FAST processor destination already exists: {destination_root}.")
    destination_root.mkdir(parents=True)
    try:
        for candidate in sorted(source_root.rglob("*")):
            relative = candidate.relative_to(source_root)
            output = destination_root / relative
            if candidate.is_symlink():
                resolved = candidate.resolve(strict=True)
                if resolved.is_dir():
                    raise ValueError(f"FAST processor source contains a directory symlink: {candidate}.")
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(resolved, output)
            elif candidate.is_dir():
                output.mkdir(exist_ok=True)
            elif candidate.is_file():
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, output)
            else:
                raise ValueError(f"Unsupported FAST processor source path: {candidate}.")
        tree_sha256, file_count = fast_processor_tree_identity(destination_root)
        return FastProcessorArtifact(destination_root, tree_sha256, file_count)
    except Exception:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise


def materialize_pinned_fast_processor_snapshot(
    source: str | Path, destination: str | Path
) -> FastProcessorArtifact:
    """Package only the reviewed files from a potentially cache-expanded Hub snapshot."""
    source_root = Path(source)
    missing = [
        filename for filename in DEFAULT_FAST_PROCESSOR_FILES if not (source_root / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Pinned FAST processor snapshot is missing required files: {missing}.")
    filtered_root = Path(tempfile.mkdtemp(prefix="perceptron-isaac-fast-"))
    try:
        for filename in DEFAULT_FAST_PROCESSOR_FILES:
            source_file = source_root / filename
            resolved = source_file.resolve(strict=True) if source_file.is_symlink() else source_file
            shutil.copyfile(resolved, filtered_root / filename)
        return materialize_fast_processor_artifact(filtered_root, destination)
    finally:
        shutil.rmtree(filtered_root)


def resolve_pinned_fast_processor_snapshot(*, local_files_only: bool = False) -> Path:
    """Resolve only the immutable processor revision used by ISAAC checkpoints."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=DEFAULT_FAST_PROCESSOR_REPOSITORY,
            revision=DEFAULT_FAST_PROCESSOR_REVISION,
            local_files_only=local_files_only,
            allow_patterns=list(DEFAULT_FAST_PROCESSOR_FILES),
        )
    )


def load_fast_action_processor(path: str | Path, *, expected_tree_sha256: str):
    """Verify and load the exact pinned FAST processor artifact.

    Transformers 5.4's ``AutoProcessor`` no longer reconstructs this legacy
    remote processor correctly, so load its fully packaged tokenizer and
    hash-authenticated Python class explicitly. This remains remote-code
    execution, but only after the complete tree matches the checkpoint-owned
    digest and the processor metadata matches the reviewed artifact schema.
    """
    if not expected_tree_sha256 or len(expected_tree_sha256) != 64:
        raise ValueError("FAST processor loading requires a checkpoint-owned tree SHA-256.")
    if expected_tree_sha256 != DEFAULT_FAST_PROCESSOR_TREE_SHA256:
        raise ValueError(
            "FAST processor digest is not the reviewed pinned revision: "
            f"expected {DEFAULT_FAST_PROCESSOR_TREE_SHA256}, got {expected_tree_sha256}."
        )
    tree_sha256, _ = fast_processor_tree_identity(path)
    if tree_sha256 != expected_tree_sha256:
        raise ValueError(
            f"FAST processor artifact identity mismatch: expected {expected_tree_sha256}, got {tree_sha256}."
        )
    root = Path(path)
    required = {
        "processing_action_tokenizer.py",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(filename for filename in required if not (root / filename).is_file())
    if missing:
        raise ValueError(f"FAST processor artifact is missing required files: {missing}.")
    processor_config = json.loads((root / "processor_config.json").read_text())
    if processor_config.get("processor_class") != "UniversalActionProcessor" or processor_config.get(
        "auto_map"
    ) != {"AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"}:
        raise ValueError("FAST processor metadata does not name the pinned UniversalActionProcessor.")
    tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(root / "tokenizer.json"),
        clean_up_tokenization_spaces=bool(tokenizer_config.get("clean_up_tokenization_spaces", False)),
        model_max_length=int(tokenizer_config.get("model_max_length", int(1e30))),
    )
    module_name = f"lerobot_verified_fast_{tree_sha256}"
    source_path = root / "processing_action_tokenizer.py"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    # The package tree was authenticated immediately above. Execute those
    # exact bytes without importlib's SourceFileLoader, which otherwise writes
    # __pycache__ into the immutable checkpoint tree and invalidates its digest
    # after the first load.
    source_bytes = source_path.read_bytes()
    exec(compile(source_bytes, str(source_path), "exec"), module.__dict__)  # nosec B102 - deliberate remote-code load, gated on the checkpoint-owned tree digest verified above
    processor_class = getattr(module, "UniversalActionProcessor", None)
    if processor_class is None:
        raise ValueError("Verified FAST processor code does not define UniversalActionProcessor.")
    return processor_class(
        bpe_tokenizer=tokenizer,
        scale=float(processor_config["scale"]),
        vocab_size=int(processor_config["vocab_size"]),
        min_token=int(processor_config["min_token"]),
        action_dim=processor_config.get("action_dim"),
        time_horizon=processor_config.get("time_horizon"),
    )


def encode_fast_action_tokens(
    processor: Any, normalized_actions: Any, *, reserved_pool_size: int
) -> list[int]:
    """Encode one normalized chunk and enforce the mharmony reserved-group bounds."""
    output = processor(normalized_actions)
    if isinstance(output, dict):
        for key in ("tokens", "token_ids", "input_ids"):
            if key in output:
                output = output[key]
                break
    if hasattr(output, "tolist"):
        output = output.tolist()
    if isinstance(output, list | tuple) and output and isinstance(output[0], list | tuple):
        if len(output) != 1:
            raise ValueError(f"FAST processor returned {len(output)} token rows for one action chunk.")
        output = output[0]
    if not isinstance(output, list | tuple) or not output:
        raise ValueError("FAST processor returned no action tokens.")
    tokens = [int(token) for token in output]
    if any(token < 0 or token >= reserved_pool_size for token in tokens):
        raise ValueError(
            f"FAST action token IDs must be in [0, {reserved_pool_size}); "
            f"got min={min(tokens)}, max={max(tokens)}."
        )
    return tokens
