"""Parameter-free ISAAC capability for actual post-pack stock training counts."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.training_loss_normalization import TrainingObjectiveSchema, training_state_to_device

from .configuration_perceptron_isaac import PerceptronIsaacConfig
from .loss_perceptron_isaac import (
    ISAAC_FLOW_LOSS_MASK_KEY,
    ISAAC_FLOW_NOISE_KEY,
    ISAAC_FLOW_TIMESTEPS_KEY,
    perceptron_isaac_objective_counts,
)
from .processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from .tensor_stream import TensorStream


@dataclass(frozen=True)
class IsaacTrainingLossNormalizer:
    """Count retained chunks and independent NTP labels, without sampling model randomness."""

    config: PerceptronIsaacConfig
    action_dim: int
    action_horizon: int
    align_flow_tensors: Callable[[dict[str, Any], int], dict[str, Tensor | None]]

    @property
    def schema(self) -> TrainingObjectiveSchema:
        names = ("flow", "text") if self.config.loss_plan == "text_ntp_fast_flow_action" else ("flow",)
        return TrainingObjectiveSchema(names)

    def local_counts(self, batch: dict[str, Any]) -> tuple[int, ...]:
        stream = batch[PERCEPTRON_ISAAC_STREAM_KEY]
        if not isinstance(stream, TensorStream):
            raise TypeError("ISAAC training counts require a prepared TensorStream.")
        count = 0 if stream.collective_only else len(stream.streams)
        fixtures = self.align_flow_tensors(batch, count)
        counts = perceptron_isaac_objective_counts(
            stream,
            loss_plan=str(self.config.loss_plan),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            train_samples_per_chunk=self.config.train_samples_per_chunk,
            flow_mask_padded_action_rows=self.config.flow_mask_padded_action_rows,
            exclude_non_agent_roles=self.config.exclude_non_agent_roles,
            flow_loss_mask=fixtures[ISAAC_FLOW_LOSS_MASK_KEY],
            flow_timesteps=fixtures[ISAAC_FLOW_TIMESTEPS_KEY],
            flow_noise=fixtures[ISAAC_FLOW_NOISE_KEY],
        )
        return tuple(counts[name] for name in self.schema.names)

    def encode_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        stream = batch[PERCEPTRON_ISAAC_STREAM_KEY]
        if not isinstance(stream, TensorStream):
            raise TypeError("ISAAC training staging requires a prepared TensorStream.")
        return training_state_to_device(
            {**batch, PERCEPTRON_ISAAC_STREAM_KEY: stream.training_state()}, torch.device("cpu")
        )

    def decode_batch(self, state: dict[str, Any], device: torch.device) -> dict[str, Any]:
        batch = training_state_to_device(state, device)
        batch[PERCEPTRON_ISAAC_STREAM_KEY] = TensorStream.from_training_state(
            batch[PERCEPTRON_ISAAC_STREAM_KEY]
        )
        return batch
