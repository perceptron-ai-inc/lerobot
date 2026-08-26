from __future__ import annotations

import weakref

import pytest
import torch
from torch import nn

# accelerate lives in the `training` extra and lerobot_train needs `datasets`; neither is
# installed in the base fast_tests tier (see .github/workflows/fast_tests.yml).
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from accelerate import Accelerator  # noqa: E402
from accelerate.utils import GradientAccumulationPlugin  # noqa: E402

from lerobot.policies.processor_utils import RETAINED_SAMPLE_INDICES_KEY  # noqa: E402
from lerobot.scripts.lerobot_train import (  # noqa: E402
    _align_sample_weights_with_loss,
    _compute_update_loss_denominators,
    _run_optimizer_update_microbatches,
    update_policy,
)


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0
        self.update_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        prediction = self.weight * batch["x"]
        loss = (prediction - batch["y"]).square().mean()
        return loss, {"tiny_loss": loss.detach()}

    def update(self):
        self.update_calls += 1


class _CountedTinyPolicy(_TinyPolicy):
    """Policy fixture that exposes a valid-element denominator to the trainer."""

    def get_loss_denominators(self, batch):
        return {"elements": torch.tensor(float(batch["x"].numel()))}

    def forward(self, batch, loss_denominators=None, loss_accumulation_scale=1):
        self.forward_calls += 1
        prediction = self.weight * batch["x"]
        numerator = (prediction - batch["y"]).square().sum()
        if loss_denominators is None:
            loss = numerator / batch["x"].numel()
        else:
            loss = numerator * loss_accumulation_scale / loss_denominators["elements"]
        return loss, {"tiny_loss": loss.detach()}


class _WeightedTinyPolicy(_TinyPolicy):
    def forward(self, batch, reduction="mean"):
        self.forward_calls += 1
        x, y = batch["x"], batch["y"]
        retained = batch.get(RETAINED_SAMPLE_INDICES_KEY)
        if retained is not None:
            indices = torch.as_tensor(retained, dtype=torch.long)
            x, y = x.index_select(0, indices), y.index_select(0, indices)
        prediction = self.weight * x
        per_sample = (prediction - y).square()
        loss = per_sample if reduction == "none" else per_sample.mean()
        return loss, {"tiny_loss": per_sample.mean().detach()}


class _FixedSampleWeighter:
    def __init__(self, weights):
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def compute_batch_weights(self, _batch):
        return self.weights, {"mean_weight": self.weights.mean().item()}


class _CollectiveFreeAccelerator:
    """Small denominator/unwrap stand-in for microbatch ordering tests."""

    device = torch.device("cpu")

    def __init__(self):
        self.unwrap_calls = 0

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        self.unwrap_calls += 1
        return policy

    @staticmethod
    def reduce(value, reduction="sum"):
        return value


class _Tracker:
    def __init__(self):
        self.policy_metrics = []

    def update_metrics(self, metrics):
        self.policy_metrics.append(metrics)


class _Scheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def _make_training_case(
    *,
    accumulation_steps: int = 1,
    policy_cls: type[_TinyPolicy] = _TinyPolicy,
) -> tuple[Accelerator, _TinyPolicy, torch.optim.Optimizer, _Tracker]:
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=accumulation_steps,
            sync_with_dataloader=False,
        ),
    )
    policy = policy_cls()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    return accelerator, policy, optimizer, _Tracker()


def test_update_policy_accumulates_before_optimizer_scheduler_and_policy_update():
    accelerator, policy, optimizer, tracker = _make_training_case(accumulation_steps=2)
    scheduler = _Scheduler()
    batch = {"x": torch.ones(1), "y": torch.ones(1)}

    update_policy(tracker, policy, batch, optimizer, 10.0, accelerator, scheduler)

    assert accelerator.sync_gradients is False
    assert accelerator.unwrap_model(policy).weight.item() == 0.0
    assert scheduler.step_calls == 0
    assert accelerator.unwrap_model(policy).update_calls == 0
    assert not hasattr(tracker, "grad_norm")

    update_policy(tracker, policy, batch, optimizer, 10.0, accelerator, scheduler)

    assert accelerator.sync_gradients is True
    assert accelerator.unwrap_model(policy).weight.item() == pytest.approx(0.2)
    assert scheduler.step_calls == 1
    assert accelerator.unwrap_model(policy).update_calls == 1
    assert tracker.grad_norm == pytest.approx(2.0)
    assert len(tracker.policy_metrics) == 2


def test_filtered_sample_weights_select_retained_rows_even_for_one_loss():
    weights = torch.tensor([1.0, 20.0, 300.0])
    batch = {RETAINED_SAMPLE_INDICES_KEY: [1]}

    selected = _align_sample_weights_with_loss(weights, torch.tensor([4.0]), batch)

    torch.testing.assert_close(selected, torch.tensor([20.0]))


def test_filtered_sample_indices_are_validated_on_cpu_before_selection(monkeypatch):
    import lerobot.scripts.lerobot_train as train_module

    weights = torch.tensor([1.0, 20.0, 300.0])
    loss = torch.tensor([4.0])
    real_as_tensor = torch.as_tensor
    requested_devices = []

    def recording_as_tensor(*args, **kwargs):
        requested_devices.append(kwargs.get("device"))
        return real_as_tensor(*args, **kwargs)

    monkeypatch.setattr(train_module.torch, "as_tensor", recording_as_tensor)

    selected = _align_sample_weights_with_loss(
        weights,
        loss,
        {RETAINED_SAMPLE_INDICES_KEY: [1]},
    )

    assert [torch.device(device).type for device in requested_devices] == ["cpu"]
    torch.testing.assert_close(selected, torch.tensor([20.0]))


def test_update_policy_aligns_weights_after_processor_filters_samples():
    accelerator, policy, optimizer, tracker = _make_training_case(policy_cls=_WeightedTinyPolicy)
    batch = {
        "x": torch.ones(3),
        "y": torch.tensor([1.0, 100.0, 3.0]),
        RETAINED_SAMPLE_INDICES_KEY: [0, 2],
    }

    update_policy(
        tracker,
        policy,
        batch,
        optimizer,
        10.0,
        accelerator,
        sample_weighter=_FixedSampleWeighter([1.0, 1000.0, 3.0]),
    )

    # Only rows 0 and 2 remain. Their weighted gradient at w=0 is
    # (1 * -2 + 3 * -6) / 4 = -5, so SGD(lr=.1) moves the weight to .5.
    assert accelerator.unwrap_model(policy).weight.item() == pytest.approx(0.5)


def test_generic_policy_microbatches_are_preprocessed_and_updated_streamingly():
    events = []
    raw_batches = iter([{"x": torch.ones(1)}, {"x": torch.ones(1)}])

    def load_batch():
        events.append("load")
        return next(raw_batches)

    def update_batch(_batch, denominators):
        assert denominators is None
        events.append("update")

    _run_optimizer_update_microbatches(
        policy=_TinyPolicy(),
        accelerator=_CollectiveFreeAccelerator(),
        gradient_accumulation_steps=2,
        sample_weighter=None,
        load_batch=load_batch,
        update_batch=update_batch,
    )

    assert events == ["load", "update", "load", "update"]


def test_denominator_policy_buffers_the_update_before_first_backward():
    events = []
    raw_batches = iter([{"x": torch.ones(1)}, {"x": torch.ones(3)}])
    accelerator = _CollectiveFreeAccelerator()

    def load_batch():
        events.append("load")
        return next(raw_batches)

    def update_batch(_batch, denominators):
        torch.testing.assert_close(denominators["elements"], torch.tensor(4.0))
        events.append("update")

    _run_optimizer_update_microbatches(
        policy=_CountedTinyPolicy(),
        accelerator=accelerator,
        gradient_accumulation_steps=2,
        sample_weighter=None,
        load_batch=load_batch,
        update_batch=update_batch,
    )

    assert events == ["load", "load", "update", "update"]
    assert accelerator.unwrap_calls == 1


def test_batched_denominator_validation_preserves_the_invalid_field_name():
    class InvalidDenominatorPolicy(_TinyPolicy):
        def get_loss_denominators(self, _batch):
            return {"valid": torch.tensor(1.0), "invalid": torch.tensor(float("nan"))}

    with pytest.raises(ValueError, match="denominator 'invalid' must be one positive finite scalar"):
        _compute_update_loss_denominators(
            InvalidDenominatorPolicy(),
            [{"x": torch.ones(1)}],
            _CollectiveFreeAccelerator(),
        )


def _one_update(*, accumulation_steps, batches, policy_cls=_TinyPolicy, compute_denominators=False):
    accelerator, policy, optimizer, tracker = _make_training_case(
        accumulation_steps=accumulation_steps,
        policy_cls=policy_cls,
    )
    denominators = (
        _compute_update_loss_denominators(policy, batches, accelerator) if compute_denominators else None
    )
    for batch in batches:
        update_policy(tracker, policy, batch, optimizer, 10.0, accelerator, loss_denominators=denominators)
    return accelerator.unwrap_model(policy).weight.detach().clone()


def test_accumulated_microbatches_match_one_large_deterministic_batch():
    first = {"x": torch.tensor([1.0, 2.0]), "y": torch.tensor([1.0, -1.0])}
    second = {"x": torch.tensor([3.0, 4.0]), "y": torch.tensor([0.5, 2.0])}
    large = {key: torch.cat((first[key], second[key])) for key in first}

    large_batch_weight = _one_update(accumulation_steps=1, batches=[large])
    accumulated_weight = _one_update(accumulation_steps=2, batches=[first, second])

    torch.testing.assert_close(accumulated_weight, large_batch_weight, rtol=0.0, atol=1e-7)


def test_counted_accumulation_matches_large_batch_with_unequal_microbatch_sizes():
    first = {"x": torch.tensor([1.0]), "y": torch.tensor([1.0])}
    second = {
        "x": torch.tensor([2.0, 3.0, 4.0]),
        "y": torch.tensor([-1.0, 0.5, 2.0]),
    }
    large = {key: torch.cat((first[key], second[key])) for key in first}

    large_batch_weight = _one_update(
        accumulation_steps=1, batches=[large], policy_cls=_CountedTinyPolicy, compute_denominators=True
    )
    accumulated_weight = _one_update(
        accumulation_steps=2,
        batches=[first, second],
        policy_cls=_CountedTinyPolicy,
        compute_denominators=True,
    )

    torch.testing.assert_close(accumulated_weight, large_batch_weight, rtol=0.0, atol=1e-7)


def test_update_policy_rebuilds_graph_after_transient_autograd_assertion(monkeypatch, caplog):
    accelerator, policy, optimizer, tracker = _make_training_case()
    batch = {"x": torch.ones(1), "y": torch.ones(1)}
    real_backward = accelerator.backward
    backward_calls = 0
    first_loss_ref = None
    real_forward = policy.forward

    def checked_forward(*args, **kwargs):
        nonlocal first_loss_ref
        if first_loss_ref is not None:
            assert first_loss_ref() is None, "failed backward graph survived into retry forward"
        loss, output = real_forward(*args, **kwargs)
        if first_loss_ref is None:
            first_loss_ref = weakref.ref(loss)
        return loss, output

    def flaky_backward(loss):
        nonlocal backward_calls
        backward_calls += 1
        if backward_calls == 1:
            raise RuntimeError(
                'isDifferentiableType(grad.scalar_type()) INTERNAL ASSERT FAILED at "engine.cpp"'
            )
        real_backward(loss)

    monkeypatch.setattr(accelerator, "backward", flaky_backward)
    monkeypatch.setattr(policy, "forward", checked_forward)

    update_policy(
        tracker,
        policy,
        batch,
        optimizer,
        10.0,
        accelerator,
    )

    unwrapped = accelerator.unwrap_model(policy)
    assert backward_calls == 2
    assert unwrapped.forward_calls == 2
    assert unwrapped.weight.item() == pytest.approx(0.2)
    assert "retrying this batch (2/3)" in caplog.text


def test_update_policy_retries_transient_forward_dtype_validator_error(monkeypatch, caplog):
    accelerator, policy, optimizer, tracker = _make_training_case()
    real_forward = policy.forward
    forward_calls = 0

    def flaky_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        if forward_calls == 1:
            raise RuntimeError("Autograd not support dtype: Byte")
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(policy, "forward", flaky_forward)

    update_policy(
        tracker,
        policy,
        {"x": torch.ones(1), "y": torch.ones(1)},
        optimizer,
        10.0,
        accelerator,
    )

    unwrapped = accelerator.unwrap_model(policy)
    assert forward_calls == 2
    assert unwrapped.forward_calls == 1
    assert unwrapped.weight.item() == pytest.approx(0.2)
    assert "transient forward-pass dtype validator error" in caplog.text
    assert "retrying this batch (2/3)" in caplog.text


def test_update_policy_does_not_retry_other_runtime_errors(monkeypatch):
    accelerator, policy, optimizer, tracker = _make_training_case()

    def oom_backward(_loss):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(accelerator, "backward", oom_backward)

    with pytest.raises(torch.OutOfMemoryError, match="CUDA out of memory"):
        update_policy(
            tracker,
            policy,
            {"x": torch.ones(1), "y": torch.ones(1)},
            optimizer,
            10.0,
            accelerator,
        )

    assert accelerator.unwrap_model(policy).forward_calls == 1


def test_update_policy_does_not_discard_accumulated_gradients_to_retry(monkeypatch):
    accelerator, policy, optimizer, tracker = _make_training_case(accumulation_steps=2)
    backward_calls = 0

    def broken_backward(_loss):
        nonlocal backward_calls
        backward_calls += 1
        raise RuntimeError('isDifferentiableType(grad.scalar_type()) INTERNAL ASSERT FAILED at "engine.cpp"')

    monkeypatch.setattr(accelerator, "backward", broken_backward)

    with pytest.raises(RuntimeError, match="isDifferentiableType"):
        update_policy(
            tracker,
            policy,
            {"x": torch.ones(1), "y": torch.ones(1)},
            optimizer,
            10.0,
            accelerator,
        )

    assert backward_calls == 1
    assert accelerator.unwrap_model(policy).forward_calls == 1


def _train_stopped_after_accelerator_guard(cfg, accelerator, monkeypatch):
    """Run train() up to the caller-supplied-Accelerator guard, stopping right after it."""
    import lerobot.scripts.lerobot_train as train_module

    def sentinel(*args, **kwargs):
        raise RuntimeError("reached-post-guard")

    monkeypatch.setattr(train_module, "init_logging", sentinel)
    cfg.job.is_remote = False
    train_module.train.__wrapped__(cfg, accelerator=accelerator)


def test_vanilla_accelerator_accepted_at_accumulation_one(monkeypatch):
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.gradient_accumulation_steps = 1
    # A plain Accelerator() keeps accelerate's default sync_with_dataloader=True; at an
    # accumulation window of 1 every step syncs, so the guard must not reject it.
    with pytest.raises(RuntimeError, match="reached-post-guard"):
        _train_stopped_after_accelerator_guard(cfg, Accelerator(cpu=True), monkeypatch)


def test_supplied_accelerator_rejected_when_sync_desynchronizes_accumulation(monkeypatch):
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.gradient_accumulation_steps = 2
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=2,
            sync_with_dataloader=True,
        ),
    )
    with pytest.raises(ValueError, match="sync_with_dataloader"):
        _train_stopped_after_accelerator_guard(cfg, accelerator, monkeypatch)
