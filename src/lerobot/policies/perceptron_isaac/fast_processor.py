"""Pinned FAST action-processor artifacts for native ISAAC training."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from transformers import PreTrainedTokenizerFast

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


class NativeFastActionProcessor:
    """In-tree encode/decode for physical-intelligence/fast revision ec4d7aa.

    Independently expressed from the pinned UniversalActionProcessor's data
    contract: orthonormal DCT-II, nearest-even quantization, character BPE and
    inverse DCT. No fitting, checkpoint Python, or ProcessorMixin dispatch.
    Malformed decoded rows retain the original zero-coefficient fallback.
    Decode geometry overrides persist, matching the pinned processor cache.
    """

    def __init__(
        self,
        bpe_tokenizer: PreTrainedTokenizerFast,
        scale: float = 10,
        vocab_size: int = 1024,
        min_token: int = 0,
        *,
        action_dim: int | None = None,
        time_horizon: int | None = None,
    ) -> None:
        import math

        if not math.isfinite(scale) or scale <= 0 or vocab_size <= 0:
            raise ValueError("FAST scale and vocabulary size must be positive and finite.")
        self._validate_geometry(time_horizon, action_dim)
        self.bpe_tokenizer = bpe_tokenizer
        self.scale = scale
        self.vocab_size = vocab_size
        self.min_token = min_token
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.called_time_horizon = time_horizon
        self.called_action_dim = action_dim

    @staticmethod
    def _validate_geometry(time_horizon: int | None, action_dim: int | None) -> None:
        for size in (time_horizon, action_dim):
            if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
                raise ValueError("FAST geometry must contain positive integer dimensions.")

    def __call__(self, action_chunk: np.ndarray) -> list[list[int]]:
        import numpy as np
        from scipy.fft import dct

        actions = np.asarray(action_chunk)
        if actions.ndim not in (2, 3) or any(size == 0 for size in actions.shape):
            raise ValueError("FAST actions must have shape [time, dim] or [batch, time, dim].")
        if not np.issubdtype(actions.dtype, np.number) or np.iscomplexobj(actions):
            raise ValueError("FAST actions must be real finite numbers.")
        if not np.isfinite(actions).all():
            raise ValueError("FAST actions must be real finite numbers.")
        if actions.ndim == 2:
            actions = actions[None, :, :]
        self.called_time_horizon, self.called_action_dim = actions.shape[-2:]
        character_codes = np.maximum(np.around(dct(actions, axis=1, norm="ortho") * self.scale)
                                     - self.min_token, 0)
        if not np.isfinite(character_codes).all() or np.any(character_codes > 0x10FFFF):
            raise ValueError("FAST quantized coefficients exceed the Unicode character range.")
        return [
            self.bpe_tokenizer("".join(chr(int(code)) for code in row.ravel()))["input_ids"]
            for row in character_codes
        ]

    def decode(
        self,
        tokens: list[list[int]],
        *,
        time_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.ndarray:
        import numpy as np
        from scipy.fft import idct

        self._validate_geometry(time_horizon, action_dim)
        horizon = time_horizon or self.time_horizon or self.called_time_horizon
        dimension = action_dim or self.action_dim or self.called_action_dim
        if horizon is None or dimension is None:
            raise ValueError("FAST decode requires geometry: encode once or provide time_horizon and action_dim.")
        if not tokens:
            raise ValueError("FAST decode requires at least one token row.")
        self.time_horizon = self.called_time_horizon = horizon
        self.action_dim = self.called_action_dim = dimension
        rows = []
        for token_row in tokens:
            try:
                decoded = self.bpe_tokenizer.decode(token_row)
                coefficients = np.array([ord(character) + self.min_token for character in decoded])
                coefficients = coefficients.reshape(horizon, dimension)
            except Exception:
                # Compatibility: fallback is zero DCT coefficients, not min_token.
                coefficients = np.zeros((horizon, dimension))
            rows.append(idct(coefficients / self.scale, axis=0, norm="ortho"))
        return np.stack(rows)


def load_fast_action_processor(
    path: str | Path, *, expected_tree_sha256: str
) -> NativeFastActionProcessor:
    """Verify and load the exact pinned FAST processor artifact.

    Checkpoint files supply data only. The historical Python file remains part
    of the immutable tree identity, but is never compiled, imported or executed.
    Tokenizer JSON is loaded by the native tokenizer constructor below.
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
    return NativeFastActionProcessor(
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
