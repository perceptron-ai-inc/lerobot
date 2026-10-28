"""Focused native modality schedule preservation; execution belongs to the parent CPU gate.

The accepted vision-block profile supplies actual Accelerate/FSDP2 preparation. These
additional cases keep real nonempty streams, both vision kinds, stock counting and
update_policy, and compare against the same unsharded native objective before wrapping.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from datetime import timedelta
from enum import Enum
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
from accelerate.utils.fsdp_utils import fsdp2_prepare_model
from torch import Tensor, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from lerobot.common.training_window import plan_training_window, prepare_training_input
from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    ISAAC_FLOW_NOISE_KEY,
    ISAAC_FLOW_TIMESTEPS_KEY,
)
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import Mk1Qwen36VLAModel
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import Qwen35VLAModel, _ordered_modality_input
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
)
from lerobot.policies.perceptron_isaac.tensor_stream import (
    ALL_TYPES,
    Event,
    TensorStream,
    TextType,
    VectorType,
    VisionType,
    create_stream,
)
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import deserialize_rng_state, serialize_rng_state
from tests.policies.perceptron_isaac.test_p2_stock_sharded_modalities import (
    CpuFsdpPreparation,
    CpuFsdpState,
    CpuMeshTopology,
    ModalityWrap,
    _assert_master_ownership,
    _measured_modality_stream,
    _supported_wrap_config,
    _tiny_native_policy,
)


class ScheduleCase(str, Enum):
    MISSING_VISION = "nonempty-missing-vision"
    MISSING_VECTOR = "nonempty-missing-vector"
    LOCAL_ORDER = "different-local-type-order"
    EMPTY = "empty-with-both-vision-kinds"
    FROZEN = "frozen-modality-encoders"


def _schedule_stream(rank: int, case: ScheduleCase) -> TensorStream:
    if rank == 1 and case == ScheduleCase.EMPTY:
        return TensorStream.collective_dummy()
    events = _measured_modality_stream(ModalityWrap.VISION_BLOCK).streams[0].events
    image_i = events[1]
    image_p = image_i.shallow_copy()
    image_p.type = VisionType.P
    image_p.data = image_p.data.flip(0).clone()
    # Vector payload fixture is independent of the unqualified vector-wrap probes.
    vector = Event(
        data=torch.tensor([[0.25, -0.5, 0.75, 0.0]]),
        time=(0.0, 0.0),
        type=VectorType.vector,
        role="user",
        dims_virtual=[1],
        dims_real=[1],
        idx_range=(0, 1),
    )
    vector.data[0, -1] = -0.0
    modalities = [vector, image_i, image_p]
    if rank == 1:
        if case == ScheduleCase.MISSING_VISION:
            modalities = [vector]
        elif case == ScheduleCase.MISSING_VECTOR:
            modalities = [image_i, image_p]
        else:
            modalities.reverse()
    padding = Event.from_text_tokens(torch.tensor([0, 0]), time=(0.0, 0.0), type=TextType.padding)
    return TensorStream([create_stream([events[0], *modalities, *events[2:], padding], ALL_TYPES)])


def _assert_rng_equal(actual: dict[str, Tensor], expected: dict[str, Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        assert torch.equal(value, expected[name]), name


def _schedule_worker(rank: int, directory: str, case: ScheduleCase) -> None:
    root = Path(directory)
    handles: list[RemovableHandle] = []
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
        OMP_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            "gloo",
            init_method=(root / "gloo").as_uri(),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=45),
        )
        accelerator = Accelerator(cpu=True)
        torch.manual_seed(71)
        policy, model = _tiny_native_policy(ModalityWrap.VISION_BLOCK)
        if case == ScheduleCase.FROZEN:
            for encoder in (model.get_input_embeddings(), model.model.vector_embedding, model.model.visual):
                encoder.requires_grad_(False)
        stream = _schedule_stream(rank, case)
        original_bytes = [event.data.contiguous().view(torch.uint8).clone() for event in stream.flat_stream()]
        batch = {
            PERCEPTRON_ISAAC_STREAM_KEY: stream,
            PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY: [] if stream.collective_only else [0],
        }
        if not stream.collective_only:
            batch[ISAAC_FLOW_TIMESTEPS_KEY] = torch.tensor([[0.4]])
            batch[ISAAC_FLOW_NOISE_KEY] = torch.zeros(1, 1, 2, 2)
        normalizer = policy.get_training_loss_normalizer()
        staged = prepare_training_input(lambda: batch, normalizer, accelerator)
        window = plan_training_window(0, [staged], normalizer, accelerator)
        assert normalizer.local_counts(batch) == ((0, 0) if stream.collective_only else (1, 2))
        rng = serialize_rng_state()
        with torch.no_grad():
            expected_embeds = model.model.embed_stream(stream)
            expected_loss, expected_metrics = policy(batch, training_loss_normalization=window.normalization)
        deserialize_rng_state(rng)

        config = _supported_wrap_config(ModalityWrap.VISION_BLOCK, model)
        host = CpuFsdpPreparation(
            state=CpuFsdpState(config.build_plugin(), TorchDynamoPlugin()),
            torch_device_mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard_cp",)),
            parallelism_config=CpuMeshTopology(),
            is_main_process=accelerator.is_main_process,
        )
        prepared = fsdp2_prepare_model(host, policy)
        optimizer = torch.optim.SGD(prepared.get_optim_params(), momentum=0.3)
        _assert_master_ownership(prepared, optimizer)
        forward_order: list[str] = []
        backward_order: list[str] = []
        embedding_requires_grad: list[bool] = []
        encoder_payloads: list[tuple[str, Tensor, Tensor | None]] = []

        def vector_input(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
            encoder_payloads.append(("vector", args[0].detach().clone(), None))

        def vision_input(_module: nn.Module, args: tuple[Tensor, ...], kwargs: dict[str, Tensor]) -> None:
            encoder_payloads.append(("vision", args[0].detach().clone(), kwargs["grid_thw"].clone()))

        handles.append(model.model.vector_embedding.register_forward_pre_hook(vector_input))
        handles.append(model.model.visual.register_forward_pre_hook(vision_input, with_kwargs=True))

        # Observe maintained module __call__ boundaries; never replace or call their forward directly.
        def observe(name: str, module: nn.Module) -> None:
            def completed(_module: nn.Module, _args: tuple[object, ...], output: Tensor) -> None:
                forward_order.append(name)
                if output.requires_grad:

                    def backward(gradient: Tensor) -> None:
                        backward_order.append(name)

                    output.register_hook(backward)

            handles.append(module.register_forward_hook(completed))

        observe("tokens", model.model.language_model.embed_tokens)
        observe("vector", model.model.vector_embedding)
        for index, block in enumerate(model.model.visual.blocks):
            observe(f"vision-block-{index}", block)

        def before_decoder(_module: nn.Module, _args: tuple[object, ...], kwargs: dict[str, object]) -> None:
            inputs = kwargs["inputs_embeds"]
            assert isinstance(inputs, Tensor)
            torch.testing.assert_close(inputs, expected_embeds[:, :-1], atol=0, rtol=0)
            embedding_requires_grad.append(inputs.requires_grad)
            forward_order.append("decoder")

        handles.append(model.model.language_model.register_forward_pre_hook(before_decoder, with_kwargs=True))
        gradient_names: list[str] = []

        def before_step(_optimizer: Optimizer, _args: tuple[object, ...], _kwargs: dict[str, object]) -> None:
            _assert_master_ownership(prepared, optimizer)
            for name, parameter in prepared.named_parameters():
                if parameter.grad is not None:
                    assert isinstance(parameter, DTensor) and isinstance(parameter.grad, DTensor), name
                    assert parameter.grad.placements == parameter.placements, name
                    value = parameter.grad.to_local()
                    assert value.dtype == torch.float32 and bool(torch.isfinite(value).all()), name
                    gradient_names.append(name)

        handles.append(optimizer.register_step_pre_hook(before_step))
        prepared_optimizer = accelerator.prepare_optimizer(optimizer)
        tracker = MetricsTracker(
            1,
            2,
            1,
            {name: AverageMeter(name, ":.6f") for name in ("loss", "lr", "update_s")},
            dp_world_size=2,
        )
        deserialize_rng_state(rng)
        _, metrics = update_policy(
            tracker,
            prepared,
            batch,
            prepared_optimizer,
            0.0,
            accelerator,
            training_loss_normalization=window.normalization,
        )
        assert metrics is not None and all(math.isfinite(value) for value in metrics.values())
        for name, value in expected_metrics.items():
            torch.testing.assert_close(torch.tensor(metrics[name]), torch.tensor(value), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(torch.tensor(metrics["loss"]), expected_loss, atol=2e-5, rtol=2e-5)
        _assert_rng_equal(serialize_rng_state(), rng)
        expected_forward = [
            "tokens",
            "vector",
            *[f"vision-block-{i}" for i in range(len(model.model.visual.blocks))],
            *[f"vision-block-{i}" for i in range(len(model.model.visual.blocks))],
            "decoder",
        ]
        assert forward_order == expected_forward, (rank, forward_order)
        if case == ScheduleCase.FROZEN:
            assert embedding_requires_grad == [False]
            assert backward_order == []
            assert not any(
                ".visual." in name or ".vector_embedding." in name or ".embed_tokens." in name
                for name in gradient_names
            )
        else:
            assert backward_order == list(reversed(expected_forward[:-1])), (rank, backward_order)
        assert gradient_names
        assert len(encoder_payloads) == 3
        for (name, payload, grid), kind in zip(
            encoder_payloads, (VectorType.vector, VisionType.I, VisionType.P), strict=True
        ):
            real = [event for event in stream.flat_stream() if event.type == kind]
            if real:
                expected_payload = torch.cat([event.data for event in real])
                if kind == VectorType.vector:
                    expected_payload = torch.nn.functional.pad(
                        expected_payload, (0, model.config.vector_max_states - expected_payload.shape[-1])
                    )
                assert torch.equal(
                    payload.view(torch.uint8), expected_payload.contiguous().view(torch.uint8)
                ), name
            else:
                assert bool(payload.eq(0).all()), name
                if kind == VectorType.vector:
                    assert tuple(payload.shape) == (1, model.config.vector_max_states)
                else:
                    vcfg = model.config.vision_config
                    assert tuple(payload.shape) == (
                        vcfg.spatial_merge_size**2,
                        vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2,
                    )
            if grid is not None:
                expected_grid = [event.dims(virtual=False) for event in real] if real else [[1, 2, 2]]
                assert torch.equal(grid, torch.tensor(expected_grid, dtype=torch.long))
        assert stream.shape[1] == expected_embeds.shape[1]
        for event, original in zip(stream.flat_stream(), original_bytes, strict=True):
            assert torch.equal(event.data.contiguous().view(torch.uint8), original)
        assert normalizer.local_counts(batch) == ((0, 0) if stream.collective_only else (1, 2))
        (root / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "case": case.value,
                    "forward_order": forward_order,
                    "backward_order": backward_order,
                    "gradient_names": gradient_names,
                    "global_counts": window.normalization.totals,
                    "router_positions_expected": 0 if stream.collective_only else stream.shape[1] - 1,
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        (root / f"rank-{rank}.exception").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        for handle in handles:
            handle.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_mk1_borrows_the_shared_sharded_modality_schedule() -> None:
    assert Mk1Qwen36VLAModel._embed_sharded_modalities is Qwen35VLAModel._embed_sharded_modalities


@pytest.mark.parametrize("frozen", [False, True], ids=["active", "frozen"])
def test_modality_dependency_preserves_signed_zero_and_finite_gradients(frozen: bool) -> None:
    payload = torch.tensor([[0.0, -0.0, 0.25]], requires_grad=not frozen)
    # A whole-tensor anchor sum would overflow even though every element is finite.
    preceding = torch.full((4,), torch.finfo(torch.float32).max, requires_grad=not frozen)
    result = _ordered_modality_input(payload, preceding)
    assert torch.equal(result.detach().view(torch.uint8), payload.detach().view(torch.uint8))
    if frozen:
        assert result is payload and not result.requires_grad
    else:
        result.sum().backward()
        torch.testing.assert_close(payload.grad, torch.ones_like(payload), atol=0, rtol=0)
        torch.testing.assert_close(preceding.grad, torch.zeros_like(preceding), atol=0, rtol=0)
    with torch.no_grad():
        assert not _ordered_modality_input(payload, preceding).requires_grad


def test_modality_dependency_does_not_hide_active_invalid_math() -> None:
    payload = torch.tensor([[float("nan"), float("inf")]])
    preceding = torch.ones(1, requires_grad=True)
    result = _ordered_modality_input(payload, preceding)
    assert bool(torch.isnan(result[0, 0])) and bool(torch.isinf(result[0, 1]))


@pytest.mark.parametrize("case", list(ScheduleCase), ids=lambda case: case.value)
def test_native_modality_schedule_preserves_payload_objective_rng_and_reverse_order(
    tmp_path: Path, case: ScheduleCase
) -> None:
    context = mp.spawn(_schedule_worker, args=(str(tmp_path), case), nprocs=2, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("modality preservation gate exceeded 180s; timeout alone is not product RED")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        for rank in range(2):
            evidence = tmp_path / f"rank-{rank}.exception"
            if evidence.is_file():
                print(evidence.read_text(encoding="utf-8"))
    for rank in range(2):
        assert (tmp_path / f"rank-{rank}.json").is_file()
