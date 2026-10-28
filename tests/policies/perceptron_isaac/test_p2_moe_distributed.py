"""Real Gloo controls for global logical-null routing auxiliaries, including padding."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import perceptron_isaac_training_loss
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    Mk1Qwen36VLAForActionGeneration,
    _build_config,
)
from lerobot.policies.perceptron_isaac.modeling_qwen36_moe import GenesisNullSparseMoeBlock
from lerobot.policies.perceptron_isaac.tensor_stream import Event, TensorStream, TextType, create_stream
from tests.policies.perceptron_isaac._fixtures import flow_stream
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config
from tests.policies.perceptron_isaac.test_p2_moe_regressions import (
    _capture_router_logits,
    _reference_auxiliary,
)


def _global_moe_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        raw = _config(test_geometry=True, mtp_present=False)
        raw["text_config"]["layer_types"] = ["full_attention", "full_attention"]
        raw["text_config"]["full_attention_interval"] = 1
        contract = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
        torch.manual_seed(47)
        model = Mk1Qwen36VLAForActionGeneration(_build_config(raw, contract)).float().train()
        blocks = [module for module in model.modules() if isinstance(module, GenesisNullSparseMoeBlock)]
        with torch.no_grad():
            for layer, block in enumerate(blocks):
                block.gate.weight.zero_()
                block.gate.weight[0, 0] = 0.7 * layer
        stream = flow_stream()
        if rank:
            events = list(stream.streams[0].events)
            events.append(
                Event.from_text_tokens(
                    torch.zeros(3, dtype=torch.long), type=TextType.padding, role=None, time=(0, 0)
                )
            )
            stream = TensorStream([create_stream(events, list(TextType))])
        captures: list[Tensor] = []
        handles = [
            block.register_forward_pre_hook(
                lambda module, inputs: _capture_router_logits(module, inputs, captures)
            )
            for block in blocks
        ]
        try:
            output = model.train_forward(stream)
        finally:
            for handle in handles:
                handle.remove()
        gathered: list[list[Tensor] | None] = [None, None]
        dist.all_gather_object(gathered, [logits.detach() for logits in captures])
        oracles = []
        counts = []
        for layer, local in enumerate(captures):
            global_logits = []
            for source_rank, source in enumerate(gathered):
                assert source is not None
                global_logits.append(local if source_rank == rank else source[layer])
            logits = torch.cat(global_logits)
            assert logits.shape[0] == 13  # Includes all forwarded pads after NTP truncation.
            oracles.append(_reference_auxiliary(logits, real=2, null=2, top_k=2))
            logical = torch.cat((logits[:, :2], logits[:, 2:].expand(-1, 2)), -1)
            selected = logical.detach().argsort(dim=-1, descending=True, stable=True)[:, :2]
            counts.append(torch.bincount(selected.flatten(), minlength=4).float())
        balance = torch.stack([oracle.balance for oracle in oracles]).mean()
        z = torch.stack([oracle.z for oracle in oracles]).mean()
        parameters = tuple(block.gate.weight for block in blocks)
        for balance_coeff, z_coeff in ((0.0, 0.0), (0.02, 0.0), (0.0, 0.001), (0.02, 0.001)):
            kwargs = {
                "loss_plan": "flow_matching_action_prediction",
                "flow_timesteps": torch.tensor([[0.5]]),
                "flow_noise": torch.zeros(1, 1, 3, 4),
            }
            baseline = perceptron_isaac_training_loss(
                output, stream, **kwargs, moe_load_balancing_loss_coeff=0.0, moe_z_loss_coeff=0.0
            )
            result = perceptron_isaac_training_loss(
                output,
                stream,
                **kwargs,
                moe_load_balancing_loss_coeff=balance_coeff,
                moe_z_loss_coeff=z_coeff,
            )
            auxiliary = balance_coeff * balance + z_coeff * z
            # Each rank differentiates its slice of the concatenation; DDP averages slices.
            expected = torch.autograd.grad(baseline.loss + 2 * auxiliary, parameters, retain_graph=True)
            actual = torch.autograd.grad(result.loss, parameters, retain_graph=True)
            for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
                dist.all_reduce(actual_gradient)
                dist.all_reduce(expected_gradient)
                torch.testing.assert_close(actual_gradient / 2, expected_gradient / 2, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(result.metrics["loss"], baseline.metrics["loss"] + auxiliary.detach())
            for actual_count, expected_count in zip(
                output.additional_info["router_dispatch_counts"], counts, strict=True
            ):
                torch.testing.assert_close(actual_count, expected_count)
                assert actual_count.sum() == 26  # Top-k lane counts SUM, never mean.
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo required")
def test_two_rank_native_auxiliary_matches_global_scalar_gradient_and_dispatch_counts(tmp_path: Path) -> None:
    mp.spawn(_global_moe_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
