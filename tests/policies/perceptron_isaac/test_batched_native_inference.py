"""No-download B=2 smoke through the actual native vision/VLM/flow model.

This uses reduced, zero-valued checkpoint tensors, not trained weights or the
SDK renderer. It proves execution and batch shape, not conditioning quality.
"""

from pathlib import Path

import torch

from lerobot.policies.perceptron_isaac.mharmony_native import IsaacNativeMharmonyRenderer
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import load_mk1_vla_from_hf
from lerobot.policies.perceptron_isaac.tensor_stream import (
    ALL_TYPES,
    Event,
    TensorStream,
    TextType,
    VectorType,
    VisionType,
    create_stream,
)
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _write_checkpoint


def _native_sample(env: int) -> TensorStream:
    events = [
        Event.from_text_tokens(
            torch.tensor([3] * (env + 1)), time=(0.0, 0.0), type=TextType.text, role="user"
        ),
        Event(
            # Native 2x2 image grid: four RGB patch rows, merged into one virtual token.
            data=torch.full((4, 12), 0.1 * (env + 1)),
            time=(1.0, 1.0),
            type=VisionType.I,
            role="user",
            dims_virtual=[1, 1, 1],
            dims_real=[1, 2, 2],
            idx_range=(0, 1),
        ),
        Event(
            data=torch.tensor([[0.25 + env * 0.1, -0.5, 0.75, 0.0]]),
            time=(2.0, 2.0),
            type=VectorType.vector,
            role="user",
            dims_virtual=[1],
            dims_real=[1],
            idx_range=(0, 1),
        ),
        Event.from_text_tokens(torch.tensor([0]), time=(3.0, 3.0), type=TextType.action_c, role="assistant"),
        Event.from_text_tokens(torch.tensor([2]), time=(3.0, 3.0), type=TextType.control, role="assistant"),
    ]
    return TensorStream([create_stream(events, ALL_TYPES)])


def test_reduced_native_model_runs_batched_multimodal_inference(tmp_path: Path) -> None:
    model_dir = tmp_path / "tiny-mk1"
    _write_checkpoint(model_dir)
    model, _, _ = load_mk1_vla_from_hf(
        model_dir,
        dtype=torch.float32,
        device="cpu",
        allow_test_only_reduced_geometry=True,
    )
    model.eval()
    stream = IsaacNativeMharmonyRenderer.collate([_native_sample(env) for env in range(2)])
    assert stream.shape == (2, 6)
    assert any(event.type == TextType.padding for event in stream.streams[0].events)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        actions = model.sample_action(stream, num_steps=1, action_dim=2, num_action_steps=2)
    assert actions.shape == (2, 2, 2)
    assert torch.isfinite(actions).all()
    # With zero-valued weights this checks independent flow noise, not learned state dependence.
    assert not torch.equal(actions[0], actions[1])
