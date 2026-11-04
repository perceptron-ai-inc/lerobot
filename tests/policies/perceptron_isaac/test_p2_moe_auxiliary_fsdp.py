"""Tiny CPU FSDP2 + recomputation control for the all-null auxiliary side channel.

This is not another storage/optimizer/Accelerator qualification suite. Both ranks
use the same eight forwarded positions so only wrapped-layer backward scheduling
is under test; no primary-objective or accumulation-window behavior is changed.
"""

from __future__ import annotations

from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn

pytest.importorskip("transformers", reason="requires lerobot[perceptron_isaac]")

from lerobot.policies.perceptron_isaac.tensor_stream import TensorStream
from tests.policies.perceptron_isaac.test_p2_moe_followup import (
    _all_null_native_model,
    _capture_native_forward,
    _logical_reference,
    _native_blocks,
    _padded_stream,
    _record_layer_visit,
)


class NativeAuxiliaryResult(NamedTuple):
    loss: Tensor
    dispatch_counts: Tensor
    token_counts: Tensor


class NativeAuxiliaryOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native = _all_null_native_model()

    def forward(self, stream: TensorStream) -> NativeAuxiliaryResult:
        output = self.native.train_forward(stream)
        info = output.additional_info
        assert len(info["load_balancing_loss"]) == len(info["router_z_loss"]) == 2
        loss = 0.02 * torch.stack(info["load_balancing_loss"]).mean()
        loss = loss + 0.001 * torch.stack(info["router_z_loss"]).mean()
        return NativeAuxiliaryResult(
            loss,
            torch.stack(info["router_dispatch_counts"]),
            torch.stack(info["router_token_count"]),
        )


def _all_null_fsdp_worker(rank: int, rendezvous: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        stream = _padded_stream(3)
        reference = _all_null_native_model()
        reference_capture = _capture_native_forward(reference, stream)
        oracle = _logical_reference(reference_capture)
        reference_parameters = (
            reference.model.language_model.embed_tokens.weight,
            *tuple(block.gate.weight for block in _native_blocks(reference)),
        )
        expected = torch.autograd.grad(oracle.weighted(), reference_parameters)
        objective = NativeAuxiliaryOnly()
        objective.native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        assert objective.native.is_gradient_checkpointing
        visits: list[list[int]] = [[], []]
        handles = [
            block.register_forward_pre_hook(partial(_record_layer_visit, visits=visits[layer]))
            for layer, block in enumerate(_native_blocks(objective.native))
        ]
        mesh = init_device_mesh("cpu", (2,))
        for layer in objective.native.model.language_model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
        fully_shard(objective, mesh=mesh, reshard_after_forward=True)
        try:
            result = objective(stream)
            assert all(len(layer) == 1 for layer in visits)
            torch.testing.assert_close(result.loss, oracle.weighted())
            torch.testing.assert_close(
                result.dispatch_counts, torch.tensor([[0.0, 0.0, 16.0, 16.0]]).expand(2, -1)
            )
            torch.testing.assert_close(result.token_counts, torch.tensor([16.0, 16.0]))
            result.loss.backward()
            assert all(len(layer) >= 2 for layer in visits), "FSDP-wrapped decoder layers did not recompute"
        finally:
            for handle in handles:
                handle.remove()
        parameters = (
            objective.native.model.language_model.embed_tokens.weight,
            *tuple(block.gate.weight for block in _native_blocks(objective.native)),
        )
        for parameter, expected_gradient in zip(parameters, expected, strict=True):
            assert bool(expected_gradient.abs().max() > 1e-6)
            gradient = parameter.grad
            assert isinstance(gradient, DTensor), "auxiliary-only backward lost an FSDP-owned gradient"
            torch.testing.assert_close(gradient.full_tensor(), expected_gradient, atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo required")
def test_cpu_fsdp2_all_null_auxiliary_only_backward_survives_resharding_and_recompute(tmp_path: Path) -> None:
    mp.spawn(_all_null_fsdp_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
