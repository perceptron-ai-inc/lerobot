"""Per-step joint text/Flow objective through real Accelerator, DDP and update_policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as functional
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch import Tensor, nn

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")
pytest.importorskip("mharmony", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    ISAAC_FLOW_LOSS_MASK_KEY,
    ISAAC_FLOW_NOISE_KEY,
    ISAAC_FLOW_TIMESTEPS_KEY,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import PERCEPTRON_ISAAC_STREAM_KEY
from lerobot.policies.perceptron_isaac.tensor_stream import Event, TensorStream, TextType, create_stream
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from tests.policies.perceptron_isaac.test_modeling_training import _policy_with_tiny_model
from tests.policies.perceptron_isaac.test_p2_flow_regressions import (
    Chunk,
    _chunk_oracle,
    _chunks,
    _draws,
    _stream,
)


def _joint_stream(chunks: list[Chunk], labels: list[int]) -> TensorStream:
    samples = []
    for sample in _stream(chunks).streams:
        events = list(sample.events[:-1])
        events.append(
            Event.from_text_tokens(
                torch.tensor([8, *labels]), type=TextType.action, role="assistant", time=(0, 0)
            )
        )
        samples.append(create_stream(events, list(TextType)))
    return TensorStream(samples)


class _JointStepPolicy(nn.Module):
    """Expose the real tiny ISAAC forward through DDP without extra loss arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.policy = _policy_with_tiny_model()
        self.policy.config.train_samples_per_chunk = 2
        assert self.policy._isaac_model is not None
        with torch.no_grad():
            self.policy._isaac_model.action_expert.scale.fill_(0.3)
            self.policy._isaac_model.context_seed.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))
            self.policy._isaac_model.lm_head.weight.copy_(torch.linspace(-0.3, 0.5, 128).reshape(32, 4))

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict]:
        return self.policy(batch)


def _accelerator_worker(rank: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        accelerator = Accelerator(
            cpu=True,
            gradient_accumulation_steps=1,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )
        chunks = _chunks()
        partitions = (
            (([chunks[0]], [9]), ([chunks[2], chunks[0]], [9, 2, 10])),
            (([chunks[1], chunks[2]], [2, 9]), ([chunks[1], chunks[2]], [10])),
        )
        torch.manual_seed(83)
        model, reference = _JointStepPolicy(), _JointStepPolicy()
        reference.load_state_dict(model.state_dict())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        prepared, optimizer = accelerator.prepare(model, optimizer)
        tracker = MetricsTracker(
            1, 1, 1, {name: AverageMeter(name, ":.6f") for name in ("loss", "lr", "update_s")}, initial_step=0
        )
        for step in range(2):
            native = reference.policy._isaac_model
            assert native is not None
            flow = torch.cat(
                [_chunk_oracle(native.action_expert.scale, batches[step][0]) for batches in partitions]
            ).mean()
            logits = native.context_seed @ native.lm_head.weight.t()
            labels = torch.tensor(
                [label for batches in partitions for _ in batches[step][0] for label in batches[step][1]]
            )
            text = functional.cross_entropy(logits.expand(labels.numel(), -1), labels)
            text += reference.policy.config.softmax_auxiliary_loss_scale * logits.logsumexp(-1).square()
            expected_loss = flow + text
            reference_optimizer.zero_grad()
            expected_loss.backward()
            reference_optimizer.step()
            batch_chunks, batch_labels = partitions[rank][step]
            _, masks, noise, tau = _draws(batch_chunks)
            batch = {
                PERCEPTRON_ISAAC_STREAM_KEY: _joint_stream(batch_chunks, batch_labels),
                ISAAC_FLOW_LOSS_MASK_KEY: masks,
                ISAAC_FLOW_NOISE_KEY: noise,
                ISAAC_FLOW_TIMESTEPS_KEY: tau,
            }
            _, output = update_policy(tracker, prepared, batch, optimizer, 0.0, accelerator)
            assert output is not None
            torch.testing.assert_close(torch.tensor(output["loss"]), expected_loss.detach())
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo required")
def test_joint_per_step_update_uses_real_accelerator_ddp_and_update_policy(tmp_path: Path) -> None:
    mp.spawn(_accelerator_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
