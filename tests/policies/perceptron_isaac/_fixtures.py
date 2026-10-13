from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.perceptron_isaac.tensor_stream import (
    Event,
    TensorStream,
    TextType,
    create_stream,
)


class ZeroExpert(nn.Module):
    def __init__(self, *, action_dim=2, action_horizon=2, clean_at_0=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.args = SimpleNamespace(
            action_dim=action_dim,
            action_horizon=action_horizon,
            clean_at_0=clean_at_0,
        )

    def sample_timesteps(self, batch_size, device, dtype):
        return torch.full((batch_size,), 0.5, device=device, dtype=dtype)

    def forward(self, vlm_activations, vlm_mask, x_tau, tau, **kwargs):
        del vlm_activations, vlm_mask, tau, kwargs
        return x_tau * self.scale


def _event(tokens, *, event_type, role, tags=None):
    return Event.from_text_tokens(
        torch.tensor(tokens, dtype=torch.long),
        time=(0.0, 0.0),
        type=event_type,
        role=role,
        tags=tags,
    )


def flow_stream(*, action_is_pad=None) -> TensorStream:
    events = [
        _event([3, 4], event_type=TextType.text, role="user"),
        _event(
            [0],
            event_type=TextType.action_c,
            role="assistant",
            tags={
                "action_target": [[1.0, 2.0], [3.0, 4.0]],
                "action_is_pad": action_is_pad or [False, False],
            },
        ),
        _event([8, 9], event_type=TextType.action, role="assistant"),
        _event([2], event_type=TextType.control, role="assistant"),
    ]
    return TensorStream([create_stream(events, list(TextType), schedule=False)])


class TinyIsaacModel(nn.Module):
    def __init__(self, action_expert: ZeroExpert | None = None):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Linear(4, 4, bias=False)
        self.model.language = nn.Linear(4, 4, bias=False)
        self.action_expert = action_expert or ZeroExpert()
        self.input_embeddings = nn.Embedding(32, 4)
        self.lm_head = nn.Linear(4, 32, bias=False)
        nn.init.zeros_(self.lm_head.weight)
        self.context_seed = nn.Parameter(torch.zeros(4))

    def get_input_embeddings(self):
        return self.input_embeddings

    def train_forward(self, stream):
        batch_size, sequence_length = stream.shape
        activations = self.context_seed.view(1, 1, -1).expand(batch_size, sequence_length - 1, -1)
        return SimpleNamespace(
            final_activations=activations,
            final_embedding=self.lm_head,
            flow_matching_expert=self.action_expert,
        )


def flow_output(stream: TensorStream, expert: ZeroExpert):
    return TinyIsaacModel(expert).train_forward(stream)
