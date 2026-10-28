"""Joint text/flow objective through real Accelerator, DDP and update_policy.

The wrapper supplies the existing update-wide denominator contract explicitly.
This tests the engine math, not automatic trainer lookahead (which has no hook yet).
"""

from __future__ import annotations

import os
from pathlib import Path

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
    perceptron_isaac_loss_denominators,
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


class _UpdateWindowPolicy(nn.Module):
    """Test adapter for the existing policy forward kwargs, not a production API."""

    def __init__(self, denominators: dict[str, Tensor]) -> None:
        super().__init__()
        self.policy = _policy_with_tiny_model()
        self.policy.config.train_samples_per_chunk = 2
        self.denominators = denominators
        assert self.policy._isaac_model is not None
        with torch.no_grad():
            self.policy._isaac_model.action_expert.scale.fill_(0.3)
            self.policy._isaac_model.context_seed.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))
            self.policy._isaac_model.lm_head.weight.copy_(torch.linspace(-0.3, 0.5, 128).reshape(32, 4))

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        return self.policy(batch, loss_denominators=self.denominators, loss_accumulation_scale=2)


def _accelerator_worker(rank: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        accelerator = Accelerator(
            cpu=True,
            gradient_accumulation_steps=2,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )
        chunks = _chunks()
        partitions = (
            (([chunks[0]], [9]), ([chunks[2], chunks[0]], [9, 2, 10])),
            (([chunks[1], chunks[2]], [2, 9]), ([chunks[1], chunks[2]], [10])),
        )
        batches = []
        totals: dict[str, Tensor] = {}
        for microbatch, labels in partitions[rank]:
            stream = _joint_stream(microbatch, labels)
            _, masks, noise, tau = _draws(microbatch)
            counts = perceptron_isaac_loss_denominators(
                stream,
                loss_plan="text_ntp_fast_flow_action",
                action_dim=2,
                action_horizon=2,
                train_samples_per_chunk=2,
                flow_mask_padded_action_rows=True,
                exclude_non_agent_roles=True,
                flow_loss_mask=masks,
            )
            for key, count in counts.items():
                totals[key] = totals.get(key, torch.tensor(0.0)) + count
            batches.append(
                {
                    PERCEPTRON_ISAAC_STREAM_KEY: stream,
                    ISAAC_FLOW_LOSS_MASK_KEY: masks,
                    ISAAC_FLOW_NOISE_KEY: noise,
                    ISAAC_FLOW_TIMESTEPS_KEY: tau,
                }
            )
        for count in totals.values():
            dist.all_reduce(count)
        torch.testing.assert_close(totals["flow"], torch.tensor(7.0))
        torch.manual_seed(83)
        model = _UpdateWindowPolicy(totals)
        reference = _UpdateWindowPolicy(totals)
        reference.load_state_dict(model.state_dict())
        native = reference.policy._isaac_model
        assert native is not None
        flow = torch.cat(
            [
                _chunk_oracle(native.action_expert.scale, microbatch)
                for rank_batches in partitions
                for microbatch, _ in rank_batches
            ]
        ).mean()
        logits = native.context_seed @ native.lm_head.weight.t()
        text_labels = torch.tensor(
            [
                label
                for rank_batches in partitions
                for microbatch, labels in rank_batches
                for _ in microbatch
                for label in labels
            ]
        )
        text = functional.cross_entropy(logits.expand(text_labels.numel(), -1), text_labels)
        text += reference.policy.config.softmax_auxiliary_loss_scale * logits.logsumexp(-1).square()
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        (flow + text).backward()
        reference_optimizer.step()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        prepared, optimizer = accelerator.prepare(model, optimizer)
        tracker = MetricsTracker(
            1, 1, 1, {name: AverageMeter(name, ":.6f") for name in ("loss", "lr", "update_s")}, initial_step=0
        )
        metrics = []
        for batch in batches:
            _, output = update_policy(tracker, prepared, batch, optimizer, 0.0, accelerator)
            assert output is not None
            metrics.append(output)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            torch.tensor(sum(item["loss"] for item in metrics) / 2), (flow + text).detach()
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo required")
def test_joint_equal_chunk_update_uses_real_accelerator_ddp_and_update_policy(tmp_path: Path) -> None:
    mp.spawn(_accelerator_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
