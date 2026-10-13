from __future__ import annotations

import json

import pytest
import torch

from lerobot.policies.perceptron_isaac.parity import (
    FLOW_NOISE_TENSOR,
    FLOW_TAU_TENSOR,
    PARITY_MANIFEST,
    PARITY_TENSORS,
    IsaacParityError,
    compare_isaac_parity_artifacts,
    write_isaac_parity_artifact,
)


def _training_tensors() -> dict[str, torch.Tensor]:
    return {
        FLOW_TAU_TENSOR: torch.tensor([[[[0.25], [0.75]]]], dtype=torch.float32),
        FLOW_NOISE_TENSOR: torch.tensor([[[[0.1, -0.2], [0.3, -0.4]]]], dtype=torch.float32),
        "gradient/action_expert.weight": torch.tensor([1.0, -2.0], dtype=torch.float32),
        "intermediate/hidden_state": torch.tensor([0.5, -0.25], dtype=torch.float32),
    }


def _comparisons() -> dict[str, dict[str, float | str]]:
    return {
        FLOW_TAU_TENSOR: {"mode": "exact"},
        FLOW_NOISE_TENSOR: {"mode": "exact"},
        "gradient/action_expert.weight": {
            "mode": "gradient",
            "min_cosine": 0.9999,
            "max_relative_l2": 1e-3,
            "zero_norm_threshold": 1e-8,
        },
        "intermediate/hidden_state": {"mode": "allclose", "atol": 1e-5, "rtol": 1e-4},
    }


def _write_training_artifact(path, tensors=None, losses=None) -> None:
    write_isaac_parity_artifact(
        path,
        kind="training",
        tensors=tensors or _training_tensors(),
        comparisons=_comparisons(),
        losses=losses or {"flow_matching": 1.25, "text_ntp": 0.75, "total": 2.0},
        records={
            "event_sequence": ["analysis", "analysis", "final"],
            "token_ids": [100, 101, 102],
        },
        provenance={"implementation": "test"},
        loss_atol=1e-6,
        loss_rtol=1e-5,
    )


def test_training_parity_accepts_declared_numeric_tolerances(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_training_artifact(reference)
    tensors = _training_tensors()
    tensors["gradient/action_expert.weight"] += torch.tensor([5e-5, -5e-5])
    tensors["intermediate/hidden_state"] += 1e-6
    _write_training_artifact(
        candidate,
        tensors=tensors,
        losses={"flow_matching": 1.250005, "text_ntp": 0.750005, "total": 2.000005},
    )

    report = compare_isaac_parity_artifacts(reference, candidate)

    assert report["passed"]
    assert report["tensors"]["gradient/action_expert.weight"]["cosine"] >= 0.9999


def test_sampled_randomness_is_an_exact_parity_input(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_training_artifact(reference)
    tensors = _training_tensors()
    tensors[FLOW_TAU_TENSOR][0, 0, 0, 0] += 1e-7
    _write_training_artifact(candidate, tensors=tensors)

    report = compare_isaac_parity_artifacts(reference, candidate, raise_on_failure=False)

    assert not report["passed"]
    assert not report["tensors"][FLOW_TAU_TENSOR]["passed"]
    with pytest.raises(IsaacParityError, match="do not satisfy"):
        compare_isaac_parity_artifacts(reference, candidate)


def test_training_parity_requires_explicit_tau_and_noise(tmp_path) -> None:
    tensors = _training_tensors()
    tensors.pop(FLOW_NOISE_TENSOR)
    comparisons = _comparisons()
    comparisons.pop(FLOW_NOISE_TENSOR)

    with pytest.raises(IsaacParityError, match="explicit sampled tensors"):
        write_isaac_parity_artifact(
            tmp_path / "artifact",
            kind="training",
            tensors=tensors,
            comparisons=comparisons,
        )


def test_candidate_cannot_relax_reference_comparison_policy(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_training_artifact(reference)
    _write_training_artifact(candidate)
    manifest_path = candidate / PARITY_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["comparisons"]["intermediate/hidden_state"]["atol"] = 1.0
    manifest_path.write_text(json.dumps(manifest))

    report = compare_isaac_parity_artifacts(reference, candidate, raise_on_failure=False)

    assert not report["passed"]
    assert report["comparison_policy_mismatch"]


def test_parity_tensor_bundle_is_authenticated(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_training_artifact(artifact)
    tensor_path = artifact / PARITY_TENSORS
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")

    with pytest.raises(IsaacParityError, match="digest mismatch"):
        compare_isaac_parity_artifacts(artifact, artifact)


def test_empty_exact_tensor_is_supported(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    tensors = {"empty": torch.empty(0)}
    comparisons = {"empty": {"mode": "exact"}}
    write_isaac_parity_artifact(reference, kind="inference", tensors=tensors, comparisons=comparisons)
    write_isaac_parity_artifact(candidate, kind="inference", tensors=tensors, comparisons=comparisons)

    assert compare_isaac_parity_artifacts(reference, candidate)["passed"]
