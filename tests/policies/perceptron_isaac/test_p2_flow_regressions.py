"""P2.08: equal-chunk optimizer-update objectives.

The reference averages weighted elements within a draw, draws within a chunk,
then chunks across the complete rank/accumulation window. Text stays token-weighted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    perceptron_isaac_loss_denominators,
    perceptron_isaac_training_loss,
)
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import ModelOutput
from lerobot.policies.perceptron_isaac.tensor_stream import Event, TensorStream, TextType, create_stream
from tests.policies.perceptron_isaac._fixtures import ZeroExpert


@dataclass(frozen=True)
class Chunk:
    target: list[list[float]]
    is_pad: list[bool]


def _stream(chunks: list[Chunk]) -> TensorStream:
    samples = []
    for chunk in chunks:
        events = [
            Event.from_text_tokens(torch.tensor([3, 4]), type=TextType.text, role="user", time=(0, 0)),
            Event.from_text_tokens(
                torch.tensor([0]),
                type=TextType.action_c,
                role="assistant",
                time=(0, 0),
                tags={"action_target": chunk.target, "action_is_pad": chunk.is_pad},
            ),
            Event.from_text_tokens(
                torch.tensor([8, 9, 2]), type=TextType.action, role="assistant", time=(0, 0)
            ),
        ]
        samples.append(create_stream(events, list(TextType)))
    return TensorStream(samples)


def _chunks() -> list[Chunk]:
    return [
        Chunk([[1.0]], [False]),
        Chunk([[2.0, 3.0], [900.0, 900.0]], [False, True]),
        Chunk([[4.0, 5.0], [6.0, 7.0]], [False, False]),
    ]


def _draws(chunks: list[Chunk], draws: int = 2) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    targets = torch.zeros(len(chunks), 2, 2)
    masks = torch.zeros(draws, len(chunks), 2, 2)
    for index, chunk in enumerate(chunks):
        raw = torch.tensor(chunk.target)
        horizon, width = raw.shape
        targets[index, :horizon, :width] = raw
        for row, padded in enumerate(chunk.is_pad):
            if not padded:
                masks[:, index, row, :width] = 1
    # Deliberately unequal weighted support, both between chunks and between draws.
    masks[-1, :, 0, 0] = 0.25
    targets *= masks[0].gt(0)
    noise = torch.arange(draws * len(chunks) * 4, dtype=torch.float32).reshape(draws, len(chunks), 2, 2) / 10
    noise *= masks[0].gt(0)
    tau = torch.linspace(0.2, 0.8, draws).view(draws, 1).expand(-1, len(chunks))
    return targets, masks, noise, tau


def _chunk_oracle(scale: Tensor, chunks: list[Chunk], draws: int = 2) -> Tensor:
    targets, masks, noise, tau = _draws(chunks, draws)
    interpolated = tau[:, :, None, None] * noise + (1 - tau[:, :, None, None]) * targets
    errors = (scale * interpolated - (targets - noise)).square() * masks
    return (errors.sum((-2, -1)) / masks.sum((-2, -1))).mean(0)


def _output(stream: TensorStream, expert: torch.nn.Module, width: int = 16) -> ModelOutput:
    batch, length = stream.shape
    return ModelOutput(
        final_activations=torch.linspace(-0.4, 0.6, batch * (length - 1) * width).reshape(
            batch, length - 1, width
        ),
        final_embedding=torch.nn.Linear(width, 32, bias=False),
        flow_matching_expert=expert,
    )


@pytest.mark.parametrize("draws", [1, 2])
def test_flow_mean_weights_chunks_equally_not_their_valid_elements(draws: int) -> None:
    chunks = _chunks()
    stream = _stream(chunks)
    expert = ZeroExpert()
    with torch.no_grad():
        expert.scale.fill_(0.3)
    _, masks, noise, tau = _draws(chunks, draws)
    result = perceptron_isaac_training_loss(
        _output(stream, expert),
        stream,
        loss_plan="flow_matching_action_prediction",
        train_samples_per_chunk=draws,
        flow_timesteps=tau,
        flow_noise=noise,
        flow_loss_mask=masks,
    )
    expected = _chunk_oracle(expert.scale, chunks, draws)
    torch.testing.assert_close(result.per_sample_loss, expected)
    torch.testing.assert_close(result.loss, expected.mean())
    torch.testing.assert_close(result.metrics["flow_loss"], expected.mean())
    actual_gradient = torch.autograd.grad(result.loss, expert.scale, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.mean(), expert.scale)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("draws", [1, 2])
def test_equal_full_support_preserves_existing_pooled_control(draws: int) -> None:
    chunks = [Chunk([[1.0, 2.0], [3.0, 4.0]], [False, False]), _chunks()[2]]
    target, _, noise, tau = _draws(chunks, draws)
    stream = _stream(chunks)
    result = perceptron_isaac_training_loss(
        _output(stream, ZeroExpert()),
        stream,
        loss_plan="flow_matching_action_prediction",
        train_samples_per_chunk=draws,
        flow_timesteps=tau,
        flow_noise=noise,
    )
    torch.testing.assert_close(result.loss, (target - noise).square().mean())
    torch.testing.assert_close(result.loss, result.per_sample_loss.mean())


@pytest.mark.parametrize("draws", [1, 2, 8])
def test_precomputed_update_denominator_counts_chunks_not_elements_or_draws(draws: int) -> None:
    chunks = _chunks()
    _, masks, _, _ = _draws(chunks, draws)
    counts = perceptron_isaac_loss_denominators(
        _stream(chunks),
        loss_plan="text_ntp_fast_flow_action",
        action_dim=2,
        action_horizon=2,
        train_samples_per_chunk=draws,
        flow_mask_padded_action_rows=True,
        exclude_non_agent_roles=True,
        flow_loss_mask=masks,
    )
    torch.testing.assert_close(counts["flow"], torch.tensor(3.0))
    # Each assistant sequence [8,9,2] has two valid next-token labels.
    torch.testing.assert_close(counts["text"], torch.tensor(6.0))


def _global_chunk_worker(rank: int, rendezvous: str) -> None:
    """Real SUM collectives plus explicit DDP/Accelerate backward averaging, no GPU."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # Unequal chunks per rank/microbatch, but identical collective order.
        chunks = _chunks()
        partitions = (
            ([chunks[0]], [chunks[2], chunks[0]]),
            ([chunks[1], chunks[2]], [chunks[1], chunks[2]]),
        )
        expert = ZeroExpert()
        with torch.no_grad():
            expert.scale.fill_(0.3)
        local_count = torch.tensor(0.0)
        for microbatch in partitions[rank]:
            _, masks, _, _ = _draws(microbatch)
            local_count += perceptron_isaac_loss_denominators(
                _stream(microbatch),
                loss_plan="flow_matching_action_prediction",
                action_dim=2,
                action_horizon=2,
                train_samples_per_chunk=2,
                flow_mask_padded_action_rows=True,
                exclude_non_agent_roles=True,
                flow_loss_mask=masks,
            )["flow"]
        dist.all_reduce(local_count)
        metric = torch.tensor(0.0)
        for microbatch in partitions[rank]:
            stream = _stream(microbatch)
            _, masks, noise, tau = _draws(microbatch)
            result = perceptron_isaac_training_loss(
                _output(stream, expert),
                stream,
                loss_plan="flow_matching_action_prediction",
                train_samples_per_chunk=2,
                flow_timesteps=tau,
                flow_noise=noise,
                flow_loss_mask=masks,
                global_denominators={"flow": local_count},
                accumulation_scale=2,
            )
            (result.loss / 2).backward()  # Accelerate's accumulation division.
            metric += result.metrics["flow_loss"] / 2
        assert expert.scale.grad is not None
        dist.all_reduce(expert.scale.grad)
        expert.scale.grad /= 2  # DDP's rank average.
        oracle_scale = torch.tensor(0.3, requires_grad=True)
        # Draws are generated per microbatch, not regenerated as one concatenated RNG batch.
        oracle = torch.cat(
            [
                _chunk_oracle(oracle_scale, microbatch)
                for rank_batches in partitions
                for microbatch in rank_batches
            ]
        ).mean()
        expected_gradient = torch.autograd.grad(oracle, oracle_scale)[0]
        torch.testing.assert_close(expert.scale.grad, expected_gradient)
        torch.testing.assert_close(metric, oracle.detach())
        torch.testing.assert_close(local_count, torch.tensor(7.0))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo required")
def test_two_rank_two_microbatch_loss_matches_global_chunk_objective(tmp_path: Path) -> None:
    mp.spawn(_global_chunk_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize("defect", ["all_pad", "empty_draw", "padded_row", "padded_dim", "nonfinite"])
def test_equal_chunk_objective_keeps_invalid_support_fail_closed(defect: str) -> None:
    chunks = _chunks()
    _, masks, noise, tau = _draws(chunks)
    if defect == "all_pad":
        chunks[0] = Chunk([[1.0]], [True])
    elif defect == "empty_draw":
        masks[1, 1].zero_()
    elif defect == "padded_row":
        masks[0, 1, 1, 0] = 1
    elif defect == "padded_dim":
        masks[0, 0, 0, 1] = 1
    else:
        masks[0, 0, 0, 0] = float("nan")
    stream = _stream(chunks)
    with pytest.raises(ValueError, match="(?i)(pad|mask|unpadded)"):
        perceptron_isaac_training_loss(
            _output(stream, ZeroExpert()),
            stream,
            loss_plan="flow_matching_action_prediction",
            train_samples_per_chunk=2,
            flow_timesteps=tau,
            flow_noise=noise,
            flow_loss_mask=masks,
        )
