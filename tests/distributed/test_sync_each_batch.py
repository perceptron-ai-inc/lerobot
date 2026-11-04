"""Real Accelerator and stock update_policy must retain the eight-microbatch cadence."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch import Tensor, nn
from torch.multiprocessing.spawn import spawn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs.accelerator import GradientAccumulationConfig
from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import _global_objective_mean
from lerobot.policies.training_loss_normalization import TrainingLossNormalization, TrainingObjectiveSchema
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig, DummyCheckpointPolicy


@dataclass
class CadenceTrace:
    """Observe actual optimizer mutations, not calls to the accelerated wrapper."""

    microstep: int = 0
    optimizer_steps: list[int] = field(default_factory=list)
    zero_grad_steps: list[int] = field(default_factory=list)
    clip_steps: list[int] = field(default_factory=list)
    gradients: list[dict[str, Tensor]] = field(default_factory=list)


class ObservedSGD(SGD):
    """Delegate real gradient clearing while recording when the wrapper permits it."""

    def __init__(self, parameters: Iterable[nn.Parameter], trace: CadenceTrace) -> None:
        self.trace = trace
        super().__init__(parameters, lr=0.05)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.trace.zero_grad_steps.append(self.trace.microstep)
        super().zero_grad(set_to_none=set_to_none)


class ObservedAccelerator(Accelerator):
    """Only observe clipping; all accumulation/backward/optimizer behavior stays native."""

    def __init__(self, config: GradientAccumulationConfig, trace: CadenceTrace) -> None:
        self.trace = trace
        super().__init__(
            cpu=True,
            gradient_accumulation_plugin=config.build_plugin(),
            step_scheduler_with_optimizer=False,
        )

    def clip_grad_norm_(self, parameters: Iterable[Tensor], max_norm: float, norm_type: int = 2) -> Tensor:
        self.trace.clip_steps.append(self.trace.microstep)
        norm = super().clip_grad_norm_(parameters, max_norm, norm_type)
        assert isinstance(norm, Tensor)
        return norm


class ObservedPolicy(DummyCheckpointPolicy):
    """The stock post-update callback must also run only after real optimizer steps."""

    def __init__(self) -> None:
        super().__init__(DummyCheckpointConfig(device="cpu", hidden=4, push_to_hub=False))
        self.applied_updates = 0

    def update(self) -> None:
        self.applied_updates += 1


@pytest.mark.parametrize("sync_each_batch", [False, True])
def test_sync_each_batch_preserves_real_stock_optimizer_zero_clip_scheduler_cadence(
    sync_each_batch: bool,
) -> None:
    trace = CadenceTrace()
    accelerator = ObservedAccelerator(
        GradientAccumulationConfig(steps=8, sync_each_batch=sync_each_batch), trace
    )
    assert accelerator.device.type == "cpu" and accelerator.num_processes == 1
    assert accelerator.gradient_accumulation_steps == 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1997)
        policy = ObservedPolicy()
    reference = copy.deepcopy(policy)
    reference_optimizer = SGD(reference.parameters(), lr=0.05)
    raw_optimizer = ObservedSGD(policy.parameters(), trace)

    def capture_real_optimizer_step(
        optimizer: Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del optimizer, args, kwargs
        trace.optimizer_steps.append(trace.microstep)
        gradients: dict[str, Tensor] = {}
        for name, parameter in policy.named_parameters():
            assert parameter.grad is not None
            assert parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad).all()
            gradients[name] = parameter.grad.detach().clone()
        trace.gradients.append(gradients)

    raw_optimizer.register_step_pre_hook(capture_real_optimizer_step)
    raw_scheduler = LambdaLR(raw_optimizer, lr_lambda=lambda _step: 1.0)
    prepared, optimizer, scheduler = accelerator.prepare(policy, raw_optimizer, raw_scheduler)
    tracker = MetricsTracker(
        1,
        16,
        16,
        {name: AverageMeter(name, ":.6f") for name in ("loss", "grad_norm", "lr", "update_s")},
        initial_step=0,
    )
    batches = [{"observation.state": torch.full((1, 4), step / 16)} for step in range(1, 17)]
    boundary_weights = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    for microstep, batch in enumerate(batches, start=1):
        trace.microstep = microstep
        # Native scheduling intentionally advances before the first accumulated optimizer update.
        if microstep == 1:
            with pytest.warns(UserWarning, match="Detected call of.*lr_scheduler.step"):
                update_policy(tracker, prepared, batch, optimizer, 0.1, accelerator, scheduler)
        else:
            update_policy(tracker, prepared, batch, optimizer, 0.1, accelerator, scheduler)
        assert accelerator.sync_gradients is (microstep % 8 == 0)
        assert raw_scheduler.last_epoch == microstep
        assert len(trace.optimizer_steps) == microstep // 8
        assert policy.applied_updates == microstep // 8
        if microstep % 8:
            for name, parameter in policy.named_parameters():
                torch.testing.assert_close(parameter, boundary_weights[name], rtol=0, atol=0)
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            # An independent whole-window update catches premature zeroing or extra averaging.
            window = batches[microstep - 8 : microstep]
            objective = torch.stack([reference(item)[0] for item in window]).mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
            for name, parameter in reference.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                torch.testing.assert_close(trace.gradients[-1][name], parameter.grad, atol=1e-7, rtol=1e-6)
            reference_optimizer.step()
            reference_optimizer.zero_grad()
            for (name, actual), (_, expected) in zip(
                policy.named_parameters(), reference.named_parameters(), strict=True
            ):
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
                assert actual.grad is None
                assert not torch.equal(actual, boundary_weights[name])
                boundary_weights[name] = actual.detach().clone()
    assert trace.optimizer_steps == trace.zero_grad_steps == trace.clip_steps == [8, 16]
    assert policy.applied_updates == 2
    assert raw_scheduler.last_epoch == 16


class GlobalWindowPolicy(ObservedPolicy):
    """Use the real global-objective derivative with a deliberately empty local rank."""

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        training_loss_normalization: TrainingLossNormalization | None = None,
    ) -> tuple[Tensor, None]:
        assert training_loss_normalization is not None
        assert training_loss_normalization.schema.names == ("linear",)
        enabled = batch["enabled"]
        numerator = self.net(batch["observation.state"]).mean() * enabled
        loss, _, _ = _global_objective_mean(
            numerator,
            enabled,
            global_denominator=torch.tensor(training_loss_normalization.totals[0]),
            accumulation_scale=training_loss_normalization.backward_divisor,
            allow_empty=True,
        )
        return loss, None


def _gloo_sync_each_batch_worker(rank: int, rendezvous: str, directory: str, sync_each_batch: bool) -> None:
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
        trace = CadenceTrace()
        accelerator = ObservedAccelerator(
            GradientAccumulationConfig(steps=8, sync_each_batch=sync_each_batch), trace
        )
        assert accelerator.num_processes == 2 and accelerator.use_distributed
        torch.manual_seed(1997)
        policy = GlobalWindowPolicy()
        reference = copy.deepcopy(policy)
        reference_optimizer = SGD(reference.parameters(), lr=0.05)
        raw_optimizer = ObservedSGD(policy.parameters(), trace)

        def capture_real_step(optimizer: Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del optimizer, args, kwargs
            trace.optimizer_steps.append(trace.microstep)
            gradients: dict[str, Tensor] = {}
            for name, parameter in policy.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                gradients[name] = parameter.grad.detach().clone()
            trace.gradients.append(gradients)

        raw_optimizer.register_step_pre_hook(capture_real_step)
        raw_scheduler = LambdaLR(raw_optimizer, lr_lambda=lambda _step: 1.0)
        prepared, optimizer, scheduler = accelerator.prepare(policy, raw_optimizer, raw_scheduler)
        tracker = MetricsTracker(
            1,
            16,
            16,
            {name: AverageMeter(name, ":.6f") for name in ("loss", "grad_norm", "lr", "update_s")},
            initial_step=0,
        )
        normalization = TrainingLossNormalization(TrainingObjectiveSchema(("linear",)), (8,), 8)
        pending_reads: list[dict[str, int | bool]] = []
        for microstep in range(1, 17):
            trace.microstep = microstep
            values = torch.full((1, 4), microstep / 16)
            batch = {"observation.state": values, "enabled": torch.tensor(float(rank == 1))}
            # Only rank one contributes; this reference has no collectives or Accelerator divisor.
            (reference.net(values).mean() / 8).backward()
            if microstep == 1:
                with pytest.warns(UserWarning, match="Detected call of.*lr_scheduler.step"):
                    update_policy(
                        tracker,
                        prepared,
                        batch,
                        optimizer,
                        0.1,
                        accelerator,
                        scheduler,
                        training_loss_normalization=normalization,
                    )
            else:
                update_policy(
                    tracker,
                    prepared,
                    batch,
                    optimizer,
                    0.1,
                    accelerator,
                    scheduler,
                    training_loss_normalization=normalization,
                )
            assert accelerator.sync_gradients is (microstep % 8 == 0)
            assert raw_scheduler.last_epoch == microstep
            if microstep % 8:
                for actual, expected in zip(policy.parameters(), reference.parameters(), strict=True):
                    assert actual.grad is not None and expected.grad is not None
                    assert torch.isfinite(actual.grad).all()
                    # Default no_sync keeps local derivatives; True must expose the global prefix now.
                    factor = 1 if sync_each_batch else 2 * int(rank == 1)
                    torch.testing.assert_close(actual.grad, expected.grad * factor, atol=2e-7, rtol=2e-6)
                    if rank == 0:
                        assert bool(actual.grad.abs().sum() > 0) is sync_each_batch
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                pending_reads.append({"microstep": microstep, "global_prefix": sync_each_batch})
            else:
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
                for name, expected in reference.named_parameters():
                    assert expected.grad is not None and torch.isfinite(expected.grad).all()
                    torch.testing.assert_close(trace.gradients[-1][name], expected.grad, atol=2e-7, rtol=2e-6)
                reference_optimizer.step()
                reference_optimizer.zero_grad()
                for actual, expected in zip(policy.parameters(), reference.parameters(), strict=True):
                    assert actual.grad is None
                    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
            assert len(trace.optimizer_steps) == policy.applied_updates == microstep // 8
        assert trace.optimizer_steps == trace.zero_grad_steps == trace.clip_steps == [8, 16]
        assert len(pending_reads) == 14
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "sync_each_batch": sync_each_batch,
                    "pending_reads": pending_reads,
                    "optimizer_steps": trace.optimizer_steps,
                    "zero_grad_steps": trace.zero_grad_steps,
                    "clip_steps": trace.clip_steps,
                    "scheduler_microsteps": raw_scheduler.last_epoch,
                    "independent_window_oracle": True,
                }
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sync_each_batch", [False, True])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo required")
def test_sync_each_batch_communicates_every_gloo_microbatch_without_reaveraging_old_gradients(
    tmp_path: Path, sync_each_batch: bool
) -> None:
    spawn(
        _gloo_sync_each_batch_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path), sync_each_batch),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        receipt = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert receipt["optimizer_steps"] == receipt["zero_grad_steps"] == receipt["clip_steps"] == [8, 16]
        assert receipt["scheduler_microsteps"] == 16
        assert len(receipt["pending_reads"]) == 14
        assert receipt["independent_window_oracle"] is True
