"""Offline native renderer -> reduced real vision/language/expert model, without token-ID rewriting."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.policies.perceptron_isaac.fast_processor import (
    DEFAULT_FAST_PROCESSOR_TREE_SHA256,
    load_fast_action_processor,
    materialize_pinned_fast_processor_snapshot,
    resolve_pinned_fast_processor_snapshot,
)
from lerobot.policies.perceptron_isaac.isaac_stats import isaac_stats_from_dict
from lerobot.policies.perceptron_isaac.mharmony_native import IsaacNativeMharmonyRenderer
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from lerobot.policies.perceptron_isaac.tensor_stream import TextType, VisionType
from tests.policies.perceptron_isaac.test_checkpoint_lifecycle import _write_stats
from tests.policies.perceptron_isaac.test_mharmony_training import _metadata
from tests.policies.perceptron_isaac.test_qwen35_training_lifecycle import reduced_qwen35_policy


def test_cached_native_image_stream_trains_and_samples_reduced_real_qwen35(tmp_path, monkeypatch):
    pytest.importorskip("mharmony")
    monkeypatch.delenv("QWEN35_VOCAB_PATH", raising=False)
    qwen = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / (
        "hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
    )
    if not (qwen / "config.json").is_file() or not (qwen / "vocab.json").is_file():
        pytest.skip("pinned local Qwen config/vocabulary assets are unavailable")
    try:
        fast_source = resolve_pinned_fast_processor_snapshot(local_files_only=True)
    except FileNotFoundError:
        pytest.skip("pinned local FAST artifacts are unavailable")
    artifact = materialize_pinned_fast_processor_snapshot(fast_source, tmp_path / "fast_processor")
    processor = load_fast_action_processor(
        artifact.path, expected_tree_sha256=DEFAULT_FAST_PROCESSOR_TREE_SHA256
    )
    metadata = replace(_metadata(), image_size=[256, 256], max_num_patches=256)
    # Retain the full base vocabulary and every declared FAST reserved slot, not only IDs observed here.
    base_vocab_size = json.loads((qwen / "config.json").read_text())["text_config"]["vocab_size"]
    vocab_size = base_vocab_size + sum(group["size"] for group in metadata.mharmony_reserved_token_groups)
    policy = reduced_qwen35_policy(tmp_path / "seed", vocab_size=vocab_size)
    stats_path = tmp_path / "seed/isaac_stats.json"
    _write_stats(stats_path)
    policy.config.native_stats_path = str(stats_path)
    policy.config.num_settle_steps = 0
    policy.config.num_inference_steps = 2
    renderer = IsaacNativeMharmonyRenderer(
        metadata=metadata, stats=isaac_stats_from_dict(json.loads(stats_path.read_text()))
    )
    inputs = {
        "observation_window": [
            {"images": {"image": np.zeros((256, 256, 3), dtype=np.uint8)}, "proprio": np.zeros(2)}
        ],
        "prompt": "pick up the block",
        "max_num_patches": 256,
        "device": "cpu",
        "dtype": torch.float32,
        "anchor_timestamp_seconds": 0.0,
    }
    training = renderer.build_training(
        **inputs,
        action_chunk=np.asarray([[0.25, -0.5], [0.75, 1.0]], dtype=np.float32),
        action_is_pad=[False, True],
        fast_processor=processor,
    )
    inference = renderer.build(**inputs)
    for stream in (training, inference):
        assert any(event.type == VisionType.I and event.data.numel() for event in stream.streams[0].events)
        assert any(event.type == TextType.action_c for event in stream.streams[0].events)
        print(
            "Native event geometry:",
            [(str(event.type), tuple(event.data.shape)) for event in stream.streams[0].events],
        )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=0.001)
    loss, metrics = policy({PERCEPTRON_ISAAC_STREAM_KEY: training})
    assert torch.isfinite(loss)
    loss.backward()
    assert policy._isaac_model is not None
    vision_gradients = [p.grad for p in policy._isaac_model.model.visual.parameters() if p.grad is not None]
    assert vision_gradients and all(torch.isfinite(gradient).all() for gradient in vision_gradients)
    assert any(gradient.abs().sum() > 0 for gradient in vision_gradients)
    optimizer.step()
    policy.eval()
    policy.reset()
    with torch.inference_mode():
        actions = policy.predict_action_chunk({PERCEPTRON_ISAAC_STREAM_KEY: inference})
    assert actions.shape == (1, 2, 2)
    assert torch.isfinite(actions).all()
    print("Native loss:", float(loss.detach()), "metrics:", metrics, "actions:", actions.tolist())
