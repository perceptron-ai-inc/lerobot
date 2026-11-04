"""Small CPU fixtures for P2 data routing; only FAST tokenization is substituted."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.perceptron_isaac.configuration_perceptron_isaac import PerceptronIsaacConfig
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PerceptronIsaacMharmonyPackProcessorStep,
    make_perceptron_isaac_pre_post_processors,
)
from lerobot.policies.perceptron_isaac.tensor_stream import Event, TensorStream, TextType
from lerobot.utils.constants import ACTION, OBS_STATE


class ProfilePayload(TypedDict):
    profile_id: str
    profile_sha256: str
    action_stats_sha256: str
    proprio_stats_sha256: str
    clip_normalized_proprio: bool
    proprio_clip_normalized_max: float


class StatsBlockPayload(TypedDict):
    q01: list[float]
    q99: list[float]
    min: NotRequired[list[float]]
    max: NotRequired[list[float]]
    mask: NotRequired[list[bool]]


class StatsPayload(TypedDict):
    schema: str
    target_fps: float
    action_horizon: int
    action_dim: int
    proprio_dim: int
    action: StatsBlockPayload
    proprio: StatsBlockPayload
    _profile: NotRequired[ProfilePayload]


def data_stats_payload(*, gripper_upper: float = 100.0, passthrough: bool = False) -> StatsPayload:
    """Two channels: a quantile joint and an optional robot-unit gripper."""
    block: StatsBlockPayload = {"q01": [0.0, 0.0], "q99": [1.0, gripper_upper]}
    if passthrough:
        block["mask"] = [True, False]
    return {
        "schema": "flow_matching_stats_v1",
        "target_fps": 20.0,
        "action_horizon": 2,
        "action_dim": 2,
        "proprio_dim": 2,
        "action": block.copy(),
        "proprio": block.copy(),
    }


@dataclass
class FastInputSpy:
    """Capture numeric FAST inputs, not an authentic codec or tokenizer oracle."""

    inputs: list[np.ndarray] = field(default_factory=list)

    def __call__(self, actions: np.ndarray) -> list[list[int]]:
        self.inputs.append(np.asarray(actions, dtype=np.float32).copy())
        return [[7, 8]]


def data_config(
    tmp_path: Path, payload: StatsPayload, *, image_size: tuple[int, int] = (256, 256)
) -> PerceptronIsaacConfig:
    stats_path = tmp_path / "isaac_stats.json"
    stats_path.write_text(json.dumps(payload))
    height, width = image_size
    return PerceptronIsaacConfig(
        device="cpu",
        n_obs_steps=1,
        chunk_size=2,
        n_action_steps=2,
        action_dim=2,
        proprio_dim=2,
        camera_order=("image",),
        image_size=image_size,
        target_fps=20.0,
        native_stats_path=str(stats_path),
        clip_action_pose=False,
        gripper_binary_to_signed=False,
        num_settle_steps=0,
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


def data_pack(config: PerceptronIsaacConfig) -> tuple[PerceptronIsaacMharmonyPackProcessorStep, FastInputSpy]:
    preprocessor, _ = make_perceptron_isaac_pre_post_processors(config)
    pack = next(
        step for step in preprocessor.steps if isinstance(step, PerceptronIsaacMharmonyPackProcessorStep)
    )
    pack.dtype = "float32"
    fast = FastInputSpy()
    pack._fast_processor = fast
    return pack, fast


def data_batch(*, image_size: tuple[int, int] = (256, 256)) -> dict[str, Any]:
    """Already-batched dataset geometry; no resizing or history ambiguity in the fixture."""
    height, width = image_size
    return {
        "observation.images.image": torch.zeros(1, 1, 3, height, width, dtype=torch.uint8),
        OBS_STATE: torch.tensor([[[0.5, 25.0]]]),
        ACTION: torch.tensor([[[0.5, 25.0], [0.75, 75.0]]]),
        "action_is_pad": torch.tensor([[False, False]]),
        "frame_index": torch.tensor([3]),
        "episode_index": torch.tensor([0]),
        "timestamp": torch.tensor([0.3]),  # Deliberately not the 20 Hz model clock.
        "task": ["pick up the block"],
    }


def action_event(stream: TensorStream, event_type: TextType = TextType.action_c) -> Event:
    matches = [event for event in stream.streams[0].events if event.type == event_type]
    assert len(matches) == 1
    return matches[0]


def final_footer_id(stream: TensorStream) -> int:
    controls = [
        int(event.data.reshape(-1)[-1])
        for event in stream.streams[0].events
        if event.type == TextType.control and event.data.numel()
    ]
    assert controls
    return controls[-1]
