"""Focused prepared-input/count/checkpoint contracts; stock behavior lives in the frozen gate."""

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="install lerobot[perceptron_isaac]")
pytest.importorskip("mharmony", reason="install lerobot[perceptron_isaac]")

from lerobot.common.training_window import PreparedTrainingWindow, TrainingWindowCheckpoint
from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import ISAAC_FLOW_LOSS_MASK_KEY
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
)
from lerobot.policies.perceptron_isaac.tensor_stream import TensorStream
from lerobot.policies.perceptron_isaac.training_perceptron_isaac import IsaacTrainingLossNormalizer
from lerobot.policies.training_loss_normalization import (
    TrainingLossNormalization,
    TrainingObjectiveSchema,
    training_state_to_device,
)
from lerobot.utils.random_utils import deserialize_rng_state, serialize_rng_state
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_modeling_training import _config


def test_window_rng_snapshot_preserves_cached_gaussian_draws() -> None:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(19)
        np.random.seed(23)
        random.gauss(0, 1)
        np.random.normal()
        state = serialize_rng_state()
        expected = (random.gauss(0, 1), float(np.random.normal()))
        deserialize_rng_state(state)
        assert (random.gauss(0, 1), float(np.random.normal())) == expected
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _normalizer() -> IsaacTrainingLossNormalizer:
    return IsaacTrainingLossNormalizer(_config(), 2, 2, PerceptronIsaacPolicy._aligned_flow_fixture_tensors)


def test_prepared_stream_roundtrip_owns_containers_and_preserves_counts(tmp_path: Path) -> None:
    normalizer = _normalizer()
    batch = {PERCEPTRON_ISAAC_STREAM_KEY: flow_stream()}
    before = normalizer.local_counts(batch)
    state = normalizer.encode_batch(batch)
    path = tmp_path / "prepared.pt"
    torch.save(state, path)
    loaded = torch.load(path, weights_only=True)
    first = normalizer.decode_batch(loaded, torch.device("cpu"))
    second = normalizer.decode_batch(loaded, torch.device("cpu"))
    assert normalizer.local_counts(first) == normalizer.local_counts(second) == before
    first[PERCEPTRON_ISAAC_STREAM_KEY].streams[0].events.clear()
    assert normalizer.local_counts(second) == before
    assert normalizer.local_counts(normalizer.decode_batch(state, torch.device("cpu"))) == before


def test_collective_dummy_has_no_primary_counts_after_safe_roundtrip() -> None:
    normalizer = _normalizer()
    batch = {
        PERCEPTRON_ISAAC_STREAM_KEY: TensorStream.collective_dummy(),
        PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY: [],
    }
    restored = normalizer.decode_batch(normalizer.encode_batch(batch), torch.device("cpu"))
    assert restored[PERCEPTRON_ISAAC_STREAM_KEY].collective_only
    assert normalizer.local_counts(restored) == (0, 0)


@pytest.mark.parametrize("weight", [0.0, -1.0, 2.0, float("nan")])
def test_real_chunk_counts_reject_invalid_flow_weights(weight: float) -> None:
    batch = {
        PERCEPTRON_ISAAC_STREAM_KEY: flow_stream(),
        ISAAC_FLOW_LOSS_MASK_KEY: torch.full((1, 1, 2, 2), weight),
    }
    with pytest.raises(ValueError, match="flow loss mask"):
        _normalizer().local_counts(batch)


@pytest.mark.parametrize("value", [torch.tensor(float("nan")), float("inf"), object()])
def test_prepared_input_rejects_nonfinite_or_unsafe_state(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        training_state_to_device({"input": value}, torch.device("cpu"))


@pytest.mark.parametrize("names", [(), ("flow", "flow"), ("",)])
def test_objective_schema_rejects_ambiguous_names(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="schema"):
        TrainingObjectiveSchema(names)


def test_normalization_requires_a_globally_nonempty_primary_window() -> None:
    with pytest.raises(ValueError, match="globally empty"):
        TrainingLossNormalization(TrainingObjectiveSchema(("flow", "text")), (0, 0), 2)
    context = TrainingLossNormalization(TrainingObjectiveSchema(("flow", "text")), (3, 0), 4)
    assert context.backward_divisor == 4  # Not the one available input in a short tail.


def test_pending_window_safe_checkpoint_preserves_prefix_suffix_and_read_cursor(tmp_path: Path) -> None:
    normalizer = _normalizer()
    inputs = [normalizer.encode_batch({PERCEPTRON_ISAAC_STREAM_KEY: flow_stream()}) for _ in range(2)]
    context = TrainingLossNormalization(normalizer.schema, (2, 4), 2)
    window = PreparedTrainingWindow(0, inputs, context)
    window.record_forward_rng()
    checkpoint = TrainingWindowCheckpoint(1, 2, 2, 0, 2, window, serialize_rng_state(), {})
    path = tmp_path / "rank-local.pt"
    torch.save(checkpoint.state_dict(), path)
    restored = TrainingWindowCheckpoint.from_state_dict(torch.load(path, weights_only=True))
    assert restored.step == 1 and restored.read_step == 2
    assert restored.window is not None and restored.window.consumed == 1
    assert restored.window.normalization == context
    assert len(restored.window.inputs) == 2
    assert restored.continuation_rng.keys() == checkpoint.continuation_rng.keys()
    for key in restored.continuation_rng:
        torch.testing.assert_close(
            restored.continuation_rng[key], checkpoint.continuation_rng[key], rtol=0, atol=0
        )
    malformed = copy.deepcopy(checkpoint.state_dict())
    malformed["read_step"] = 1
    with pytest.raises(ValueError, match="cursors"):
        TrainingWindowCheckpoint.from_state_dict(malformed)
