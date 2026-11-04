"""Per-microbatch communication must not change accumulation or update boundaries."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch import nn
from torch.multiprocessing.spawn import spawn
from torch.optim import SGD

from lerobot.configs.accelerator import GradientAccumulationConfig


def _gloo_sync_each_batch_worker(rank: int, rendezvous: str, sync_each_batch: bool) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
        OMP_NUM_THREADS="1",
    )
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        accelerator = Accelerator(
            cpu=True,
            gradient_accumulation_plugin=GradientAccumulationConfig(
                steps=8, sync_each_batch=sync_each_batch
            ).build_plugin(),
        )
        assert accelerator.num_processes == 2 and accelerator.use_distributed
        model = nn.Linear(1, 1, bias=False)
        nn.init.ones_(model.weight)
        prepared, optimizer = accelerator.prepare(model, SGD(model.parameters(), lr=0.1))
        for step in range(1, 17):
            with accelerator.accumulate(prepared):
                # Rank zero contributes nothing. Rank averaging cancels rank one's factor two.
                loss = prepared(torch.tensor([[float(step)]])).sum() * (2 * rank)
                accelerator.backward(loss)
                boundary = step % 8 == 0
                assert accelerator.sync_gradients is boundary
                start = (step - 1) // 8 * 8 + 1
                prefix = sum(range(start, step + 1)) / 8
                factor = 1 if sync_each_batch or boundary else 2 * rank
                assert model.weight.grad is not None
                torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, prefix * factor))
                optimizer.step()
                optimizer.zero_grad()
                # Independent linear-loss oracle: only completed windows may change weights.
                expected = 1 - 0.1 * sum(range(1, step // 8 * 8 + 1)) / 8
                torch.testing.assert_close(model.weight, torch.full_like(model.weight, expected))
                assert (model.weight.grad is None) is boundary
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sync_each_batch", [False, True])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo required")
def test_sync_each_batch_communicates_without_reaveraging_or_early_updates(
    tmp_path: Path, sync_each_batch: bool
) -> None:
    spawn(
        _gloo_sync_each_batch_worker,
        args=((tmp_path / "rendezvous").as_uri(), sync_each_batch),
        nprocs=2,
        join=True,
    )
