"""P2 rollout regressions: real windowing/packing/queues, isolated SDK and model leaves.

The SDK leaf serializes an already-built real content plan into local TensorStream
records. It does not test Qwen vocabulary IDs, image processing, or model quality.
The sampler leaf derives different chunks from each stream's own proprio, with
independent horizon/feature values. Neither leaf implements batching or history.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacMharmonyContentPlan,
    IsaacNativeMharmonyRenderer,
)
from lerobot.policies.perceptron_isaac.tensor_stream import (
    ALL_TYPES,
    Event,
    TensorStream,
    TextType,
    VectorType,
    create_stream,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _write_rollout_stats(path: Path) -> None:
    block = {"q01": [-1.0, -1.0], "q99": [1.0, 1.0], "min": [-1.0, -1.0], "max": [1.0, 1.0]}
    path.write_text(
        json.dumps(
            {
                "schema": "flow_matching_stats_v1",
                "target_fps": 20.0,
                "action_horizon": 4,
                "action_dim": 2,
                "proprio_dim": 2,
                "action": block,
                "proprio": block,
            }
        )
    )


def _rollout_config(stats_path: Path, *, n_obs_steps: int = 3, settle: int = 0) -> PerceptronIsaacConfig:
    return PerceptronIsaacConfig(
        device="cpu",
        native_stats_path=str(stats_path),
        n_obs_steps=n_obs_steps,
        chunk_size=4,
        n_action_steps=2,
        action_dim=2,
        proprio_dim=2,
        image_size=(16, 16),
        camera_order=("image", "wrist_image"),
        num_settle_steps=settle,
        num_flow_samples=1,
        num_inference_steps=1,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            "observation.images.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


@pytest.fixture
def sdk_plans() -> Iterator[list[IsaacMharmonyContentPlan]]:
    """Replace only external tokenizer/media serialization; retain real plan/collate."""
    plans: list[IsaacMharmonyContentPlan] = []

    def serialize_at_sdk_boundary(
        renderer: IsaacNativeMharmonyRenderer,
        plan: IsaacMharmonyContentPlan,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None,
        patch_size: int,
        max_num_patches: int | None,
        min_num_patches: int | None,
        pixel_shuffle_scale: int,
        temporal_patch_size: int,
        rewrite_final_assistant_footer: bool = True,
    ) -> TensorStream:
        plans.append(plan)
        # These byte-valued text records are test transport, NOT Qwen token IDs.
        tokens = torch.tensor(list(plan.preamble.encode()), dtype=torch.long).reshape(-1, 1)
        vector = next(item for item in plan.user_content if item["kind"] == "vector")
        values = torch.tensor(vector["values"], dtype=torch.float32).reshape(1, -1)
        events = [
            Event(
                tokens,
                (0.0, 0.0),
                TextType.text,
                "user",
                [tokens.shape[0]],
                [tokens.shape[0]],
                (0, tokens.shape[0]),
            ),
            Event(values, (1.0, 1.0), VectorType.vector, "user", [1], [1], (0, 1)),
        ]
        for item in plan.assistant_content:
            action_tokens = torch.tensor(item["tokens"], dtype=torch.long).reshape(-1, 1)
            text_type = TextType.action_c if item["tags"]["token_group"] == "flow_action" else TextType.action
            length = action_tokens.shape[0]
            events.append(
                Event(
                    action_tokens,
                    (2.0, 2.0),
                    text_type,
                    "assistant",
                    [length],
                    [length],
                    (0, length),
                    item["tags"],
                )
            )
        return TensorStream([create_stream(events, ALL_TYPES)]).to(device=device, dtype=dtype)

    with patch.object(
        IsaacNativeMharmonyRenderer, "_render_plan_to_tensor_stream", serialize_at_sdk_boundary
    ):
        yield plans


def _assert_plan_history(plan: IsaacMharmonyContentPlan, env: int, ticks: list[int]) -> None:
    assert plan.preamble.splitlines()[0] == "move " + "the block " * env + "the block"
    images = [item for item in plan.user_content if item["kind"] == "image"]
    assert len(images) == len(ticks) * 2
    for index, tick in enumerate(ticks):
        for camera, base in enumerate((20, 40)):
            np.testing.assert_array_equal(images[index * 2 + camera]["image"], base + 60 * env + tick)
    assert not any(item["kind"] == "tokens" for item in plan.user_content)
