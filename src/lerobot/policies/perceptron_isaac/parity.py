"""Versioned parity artifacts for native ISAAC inference and training."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
from safetensors.torch import load_file, save_file

from .checkpoint_integrity import file_sha256

PARITY_SCHEMA = "perceptron_isaac_parity_v1"
PARITY_MANIFEST = "parity_manifest.json"
PARITY_TENSORS = "parity_tensors.safetensors"
FLOW_TAU_TENSOR = "fixture/flow_tau"
FLOW_NOISE_TENSOR = "fixture/flow_noise"
FLOW_LOSS_MASK_TENSOR = "fixture/flow_loss_mask"


class IsaacParityError(ValueError):
    """Raised when a parity artifact is invalid or fails its acceptance policy."""


class TensorComparison(TypedDict, total=False):
    mode: Literal["exact", "allclose", "gradient"]
    atol: float
    rtol: float
    min_cosine: float
    max_relative_l2: float
    zero_norm_threshold: float


def _validate_comparison(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    mode = spec.get("mode")
    if mode == "exact":
        if set(spec) != {"mode"}:
            raise IsaacParityError(f"Exact tensor {name!r} cannot declare tolerances.")
        return {"mode": "exact"}
    if mode == "allclose":
        if set(spec) != {"mode", "atol", "rtol"}:
            raise IsaacParityError(f"Allclose tensor {name!r} requires only atol and rtol.")
        atol, rtol = float(spec["atol"]), float(spec["rtol"])
        if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
            raise IsaacParityError(f"Allclose tolerances for {name!r} must be finite and nonnegative.")
        return {"mode": "allclose", "atol": atol, "rtol": rtol}
    if mode == "gradient":
        required = {"mode", "min_cosine", "max_relative_l2", "zero_norm_threshold"}
        if set(spec) != required:
            raise IsaacParityError(
                f"Gradient tensor {name!r} requires min_cosine, max_relative_l2, and zero_norm_threshold."
            )
        result = {key: float(spec[key]) for key in required - {"mode"}}
        if (
            not all(math.isfinite(value) for value in result.values())
            or not 0.0 <= result["min_cosine"] <= 1.0
            or result["max_relative_l2"] < 0.0
            or result["zero_norm_threshold"] < 0.0
        ):
            raise IsaacParityError(f"Gradient thresholds for {name!r} are invalid.")
        return {"mode": "gradient", **result}
    raise IsaacParityError(f"Tensor {name!r} has unsupported comparison mode {mode!r}.")


def _validate_training_fixtures(tensors: Mapping[str, torch.Tensor]) -> None:
    missing = {FLOW_TAU_TENSOR, FLOW_NOISE_TENSOR} - tensors.keys()
    if missing:
        raise IsaacParityError(
            f"Training parity requires explicit sampled tensors, missing {sorted(missing)}."
        )
    tau = tensors[FLOW_TAU_TENSOR]
    noise = tensors[FLOW_NOISE_TENSOR]
    if tau.ndim != 4 or tau.shape[-1] != 1:
        raise IsaacParityError("fixture/flow_tau must have shape [K, N, H, 1].")
    if noise.ndim != 4 or noise.shape[:3] != tau.shape[:3]:
        raise IsaacParityError("fixture/flow_noise must have shape [K, N, H, A] matching flow_tau.")
    if FLOW_LOSS_MASK_TENSOR in tensors and tensors[FLOW_LOSS_MASK_TENSOR].shape != noise.shape:
        raise IsaacParityError("fixture/flow_loss_mask must match fixture/flow_noise exactly in shape.")


def write_isaac_parity_artifact(
    output_path: str | Path,
    *,
    kind: Literal["inference", "training"],
    tensors: Mapping[str, torch.Tensor],
    comparisons: Mapping[str, TensorComparison],
    losses: Mapping[str, float] | None = None,
    records: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    loss_atol: float = 1e-6,
    loss_rtol: float = 1e-5,
) -> Path:
    """Write an immutable, self-authenticating parity artifact atomically."""
    if kind not in {"inference", "training"}:
        raise IsaacParityError(f"Unsupported parity artifact kind {kind!r}.")
    if not tensors or set(tensors) != set(comparisons):
        raise IsaacParityError("Parity tensors and comparison specs must have identical non-empty keys.")
    if kind == "training":
        _validate_training_fixtures(tensors)
    normalized_specs = {name: _validate_comparison(name, comparisons[name]) for name in sorted(comparisons)}
    for required in (FLOW_TAU_TENSOR, FLOW_NOISE_TENSOR, FLOW_LOSS_MASK_TENSOR):
        if required in normalized_specs and normalized_specs[required] != {"mode": "exact"}:
            raise IsaacParityError(f"Sampled training fixture {required!r} must use exact comparison.")

    normalized_tensors = {}
    tensor_metadata = {}
    for name in sorted(tensors):
        tensor = torch.as_tensor(tensors[name]).detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise IsaacParityError(f"Parity tensor {name!r} contains non-finite values.")
        normalized_tensors[name] = tensor
        tensor_metadata[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}

    normalized_losses = {name: float(value) for name, value in sorted((losses or {}).items())}
    if any(not math.isfinite(value) for value in normalized_losses.values()):
        raise IsaacParityError("Parity losses must be finite.")
    if not all(math.isfinite(value) and value >= 0.0 for value in (loss_atol, loss_rtol)):
        raise IsaacParityError("Loss tolerances must be finite and nonnegative.")

    output = Path(output_path).resolve()
    if output.exists():
        raise IsaacParityError(f"Parity output already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.parity-", dir=output.parent))
    try:
        tensor_path = staging / PARITY_TENSORS
        save_file(normalized_tensors, tensor_path)
        manifest = {
            "schema": PARITY_SCHEMA,
            "kind": kind,
            "tensors_file": PARITY_TENSORS,
            "tensors_sha256": file_sha256(tensor_path),
            "tensors": tensor_metadata,
            "comparisons": normalized_specs,
            "losses": normalized_losses,
            "loss_tolerance": {"atol": float(loss_atol), "rtol": float(loss_rtol)},
            "records": dict(records or {}),
            "provenance": dict(provenance or {}),
        }
        (staging / PARITY_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output)
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_isaac_parity_artifact(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    root = Path(path).resolve()
    manifest_path = root / PARITY_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise IsaacParityError(f"Invalid parity manifest at {manifest_path}.") from exc
    if manifest.get("schema") != PARITY_SCHEMA:
        raise IsaacParityError(f"Unsupported parity schema {manifest.get('schema')!r}.")
    tensor_path = root / str(manifest.get("tensors_file"))
    expected_sha = manifest.get("tensors_sha256")
    try:
        actual_sha = file_sha256(tensor_path)
    except OSError as exc:
        raise IsaacParityError(f"Missing parity tensor bundle at {tensor_path}.") from exc
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise IsaacParityError("Parity tensor bundle digest mismatch.")
    tensors = load_file(tensor_path)
    metadata = manifest.get("tensors")
    comparisons = manifest.get("comparisons")
    if not isinstance(metadata, dict) or not isinstance(comparisons, dict):
        raise IsaacParityError("Parity manifest is missing tensor metadata/comparisons.")
    if set(tensors) != set(metadata) or set(tensors) != set(comparisons):
        raise IsaacParityError("Parity tensor keys do not match the manifest.")
    for name, tensor in tensors.items():
        if metadata[name] != {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}:
            raise IsaacParityError(f"Parity tensor metadata mismatch for {name!r}.")
        _validate_comparison(name, comparisons[name])
        if not torch.isfinite(tensor).all():
            raise IsaacParityError(f"Parity tensor {name!r} contains non-finite values.")
    if manifest.get("kind") == "training":
        _validate_training_fixtures(tensors)
    return manifest, tensors


def _tensor_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            "passed": False,
            "reason": "shape_or_dtype_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
        }
    mode = spec["mode"]
    if mode == "exact":
        passed = torch.equal(reference, candidate)
        delta = (reference.float() - candidate.float()).abs()
        max_abs = float(delta.max().item()) if delta.numel() else 0.0
        return {"passed": bool(passed), "mode": mode, "max_abs_error": max_abs}

    left = reference.double().reshape(-1)
    right = candidate.double().reshape(-1)
    delta = right - left
    max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
    if mode == "allclose":
        passed = torch.allclose(reference, candidate, atol=spec["atol"], rtol=spec["rtol"])
        return {
            "passed": bool(passed),
            "mode": mode,
            "max_abs_error": max_abs,
            "atol": spec["atol"],
            "rtol": spec["rtol"],
        }

    reference_norm = float(torch.linalg.vector_norm(left).item())
    candidate_norm = float(torch.linalg.vector_norm(right).item())
    zero_threshold = float(spec["zero_norm_threshold"])
    if reference_norm <= zero_threshold:
        passed = candidate_norm <= zero_threshold
        cosine = 1.0 if passed else 0.0
        relative_l2 = 0.0 if passed else math.inf
    else:
        relative_l2 = float(torch.linalg.vector_norm(delta).item()) / reference_norm
        if candidate_norm == 0.0:
            cosine = 0.0
        else:
            cosine = float(torch.dot(left, right).item()) / (reference_norm * candidate_norm)
        passed = cosine >= spec["min_cosine"] and relative_l2 <= spec["max_relative_l2"]
    return {
        "passed": bool(passed),
        "mode": mode,
        "max_abs_error": max_abs,
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
        "cosine": cosine,
        "relative_l2": relative_l2,
        "min_cosine": spec["min_cosine"],
        "max_relative_l2": spec["max_relative_l2"],
        "zero_norm_threshold": zero_threshold,
    }


def compare_isaac_parity_artifacts(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Compare artifacts using only the committed reference acceptance policy."""
    reference_manifest, reference_tensors = load_isaac_parity_artifact(reference_path)
    candidate_manifest, candidate_tensors = load_isaac_parity_artifact(candidate_path)
    report: dict[str, Any] = {"schema": PARITY_SCHEMA, "passed": True, "tensors": {}, "losses": {}}
    if reference_manifest["kind"] != candidate_manifest["kind"]:
        report.update({"passed": False, "kind_mismatch": True})
    if reference_manifest["comparisons"] != candidate_manifest["comparisons"]:
        report.update({"passed": False, "comparison_policy_mismatch": True})
    if reference_manifest.get("records") != candidate_manifest.get("records"):
        report.update({"passed": False, "record_mismatch": True})
    if set(reference_tensors) != set(candidate_tensors):
        report.update({"passed": False, "tensor_key_mismatch": True})
    else:
        for name in sorted(reference_tensors):
            metrics = _tensor_metrics(
                reference_tensors[name],
                candidate_tensors[name],
                reference_manifest["comparisons"][name],
            )
            report["tensors"][name] = metrics
            report["passed"] = bool(report["passed"] and metrics["passed"])

    reference_losses = reference_manifest.get("losses") or {}
    candidate_losses = candidate_manifest.get("losses") or {}
    if set(reference_losses) != set(candidate_losses):
        report.update({"passed": False, "loss_key_mismatch": True})
    else:
        tolerance = reference_manifest["loss_tolerance"]
        for name in sorted(reference_losses):
            reference = float(reference_losses[name])
            candidate = float(candidate_losses[name])
            absolute_error = abs(candidate - reference)
            limit = float(tolerance["atol"]) + float(tolerance["rtol"]) * abs(reference)
            passed = math.isfinite(candidate) and absolute_error <= limit
            report["losses"][name] = {
                "passed": passed,
                "reference": reference,
                "candidate": candidate,
                "absolute_error": absolute_error,
                "limit": limit,
            }
            report["passed"] = bool(report["passed"] and passed)
    if raise_on_failure and not report["passed"]:
        raise IsaacParityError("ISAAC parity artifacts do not satisfy the reference acceptance policy.")
    return report
