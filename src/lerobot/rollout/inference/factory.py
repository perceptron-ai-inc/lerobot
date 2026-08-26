# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc|temporal_ensemble``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from threading import Event

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import validate_perceptron_direct_chunk_inference
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine
from .temporal_ensemble import TemporalEnsembleInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("temporal_ensemble")
@dataclass
class TemporalEnsembleInferenceConfig(InferenceEngineConfig):
    """Async chunks selected by observation age and temporally ensembled."""

    ensemble_m: float = 0.5
    max_step: float | None = 15.0
    smooth_alpha: float = 1.0
    actions_per_chunk: int | None = None
    ring_capacity: int = 6
    max_consecutive_errors: int = 3


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _temporal_actions_per_chunk(
    policy: PreTrainedPolicy, requested_actions_per_chunk: int | None
) -> int | None:
    """Validate stateful policy semantics used by direct chunk inference."""
    policy_type = getattr(policy.config, "type", None) or getattr(policy, "type", None)
    return validate_perceptron_direct_chunk_inference(
        policy.config,
        requested_actions_per_chunk,
        mode="temporal",
        policy_type=policy_type,
    )


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    preflight_inferences: int = 0,
    shutdown_event: Event | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, RTCInferenceConfig):
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    if isinstance(config, TemporalEnsembleInferenceConfig):
        # torch.compile is applied to the policy upstream (rollout context); the
        # engine only needs enough discarded warmup predictions to flush
        # compilation before it reports ready, mirroring the RTC engine.
        compile_warmup = max(1, compile_warmup_inferences) if use_torch_compile else 0
        actions_per_chunk = _temporal_actions_per_chunk(policy, config.actions_per_chunk)
        return TemporalEnsembleInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_type=robot_wrapper.robot_type,
            hw_features=hw_features,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            fps=fps,
            device=device,
            ensemble_m=config.ensemble_m,
            max_step=config.max_step,
            smooth_alpha=config.smooth_alpha,
            actions_per_chunk=actions_per_chunk,
            ring_capacity=config.ring_capacity,
            warmup_predictions=max(preflight_inferences, compile_warmup),
            max_consecutive_errors=config.max_consecutive_errors,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
