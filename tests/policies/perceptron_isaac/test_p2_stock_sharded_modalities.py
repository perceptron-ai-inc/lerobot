"""Reviewer R1: empty stock inputs must not skip supported FSDP2 modality units.

All modality encoders, hybrid decoder layers, router reductions, flow/text losses,
stock input staging/counting and update_policy execute unchanged. The measured
TensorStream is the supported prepacked input API; no tokenizer assets are needed.
CPU preparation uses accelerate's real fsdp2_prepare_model with the same minimal
CPU mesh carrier as the existing FSDP2 tests, not a replacement wrapping policy.
Runtime RED/GREEN belongs to the parent's approved offline two-Gloo gate.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, TorchDynamoPlugin
from accelerate.utils.fsdp_utils import fsdp2_prepare_model
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from lerobot.common.training_window import plan_training_window, prepare_training_input
from lerobot.configs.accelerator import FSDPConfig
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    TORCH_SDPA_ATTENTION_BACKEND,
    Mk1Qwen36VLAForActionGeneration,
    _build_config,
    _configure_attention_backend,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import (
    PerceptronIsaacPolicy,
    inspect_fsdp2_trainable_masters,
)
from lerobot.policies.perceptron_isaac.modeling_qwen36_moe import (
    GenesisQwen36GatedDeltaNet,
    GenesisQwen36VisionModel,
)
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
)
from lerobot.policies.perceptron_isaac.tensor_stream import (
    ALL_TYPES,
    Event,
    TensorStream,
    TextType,
    VisionType,
    create_stream,
)
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config as _native_config
from tests.policies.perceptron_isaac.test_modeling_training import _config as _policy_config


class ModalityWrap(str, Enum):
    VISION_BLOCK = "vision_block"
    VISION_CONTAINER = "vision_container"

    @property
    def encoder_path(self) -> str:
        return "_isaac_model.model.visual"


@dataclass(frozen=True)
class CpuFsdpState:
    fsdp_plugin: FullyShardedDataParallelPlugin
    dynamo_plugin: TorchDynamoPlugin


@dataclass(frozen=True)
class CpuMeshTopology:
    fsdp_dim_names: tuple[str, ...] = ("dp_shard_cp",)


@dataclass(frozen=True)
class CpuFsdpPreparation:
    """Fields read by accelerate's production CPU-capable FSDP2 preparation helper."""

    state: CpuFsdpState
    torch_device_mesh: DeviceMesh
    parallelism_config: CpuMeshTopology
    is_main_process: bool
    device: torch.device = torch.device("cpu")
    mixed_precision: str = "no"


@dataclass
class ModalityTrace:
    """Observe calls only; never invoke an encoder, mutate inputs or replace a collective."""

    rank: int
    directory: Path
    required_units: tuple[str, ...]
    entered: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)
    decoder_entered: bool = False
    decoder_completed: bool = False

    def record(self, message: str) -> None:
        # One update and a finite set of native modules bound this trace without truncation.
        with (self.directory / f"rank-{self.rank}.trace").open("a", encoding="utf-8") as output:
            output.write(message + "\n")
            output.flush()

    def observe_unit(self, path: str, unit: nn.Module) -> list[RemovableHandle]:
        def enter(_module: nn.Module, _args: tuple[object, ...]) -> None:
            self.entered.add(path)
            self.record(f"enter {path}")

        def complete(_module: nn.Module, _args: tuple[object, ...], _output: object) -> None:
            self.completed.add(path)
            self.record(f"complete {path}")

        # Enter precedes FSDP's gather hook; complete requires the genuine native forward.
        return [
            unit.register_forward_pre_hook(enter, prepend=True),
            unit.register_forward_hook(complete),
        ]

    def before_decoder(self, _module: nn.Module, _args: tuple[object, ...]) -> None:
        self.decoder_entered = True
        missing = sorted(set(self.required_units) - self.entered)
        self.record(f"before decoder; missing modality units={missing}")
        assert not missing, (
            f"stock collective dummy skipped sharded modality before decoder on rank {self.rank}: "
            f"{missing}. Real ranks gather these units before decoder; collective order must agree."
        )

    def after_decoder(self, _module: nn.Module, _args: tuple[object, ...], _output: object) -> None:
        self.decoder_completed = True
        self.record("complete decoder")


def _tiny_native_policy(
    wrapping: ModalityWrap,
) -> tuple[PerceptronIsaacPolicy, Mk1Qwen36VLAForActionGeneration]:
    raw = _native_config(test_geometry=True, mtp_present=False)
    raw["genesis_vla"]["action_expert"].update(action_dim=2, action_horizon=2)
    contract = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
    native_config = _build_config(raw, contract)
    _configure_attention_backend(native_config, TORCH_SDPA_ATTENTION_BACKEND)
    model = Mk1Qwen36VLAForActionGeneration(native_config).float()
    assert native_config.text_config.layer_types == ["linear_attention", "full_attention"]
    assert isinstance(model.model.language_model.layers[0].linear_attn, GenesisQwen36GatedDeltaNet)
    config = _policy_config()
    config.push_to_hub = False
    policy = PerceptronIsaacPolicy(config)
    policy._isaac_model = model
    policy._configure_training_parameters()  # Real freezing/RoPE preparation before wrapping.
    assert policy.config.train_storage_fp32
    return policy, model


def _measured_modality_stream(wrapping: ModalityWrap) -> TensorStream:
    # Native patch_size=2, temporal_patch_size=2, merge=2: four tubelets -> one token.
    modality = Event(
        data=torch.linspace(-0.2, 0.3, 4 * 3 * 2 * 2 * 2).reshape(4, 24),
        time=(0.0, 0.0),
        type=VisionType.I,
        role="user",
        dims_virtual=[1, 1, 1],
        dims_real=[1, 2, 2],
        idx_range=(0, 1),
    )
    events = [
        Event.from_text_tokens(torch.tensor([3, 4]), time=(0.0, 0.0), role="user"),
        modality,
        Event.from_text_tokens(
            torch.tensor([0]),
            time=(0.0, 0.0),
            type=TextType.action_c,
            role="assistant",
            tags={"action_target": [[0.2, -0.4], [0.8, 0.1]], "action_is_pad": [False, False]},
        ),
        Event.from_text_tokens(
            torch.tensor([8, 9, 2]), time=(0.0, 0.0), type=TextType.action, role="assistant"
        ),
    ]
    return TensorStream([create_stream(events, ALL_TYPES)])


def _supported_wrap_config(wrapping: ModalityWrap, model: Mk1Qwen36VLAForActionGeneration) -> FSDPConfig:
    if wrapping == ModalityWrap.VISION_CONTAINER:
        # The old vision-size profile hit a helper/container limitation, not the modality defect.
        assert isinstance(model.model.visual, GenesisQwen36VisionModel)
        return FSDPConfig(
            wrap_modules=["GenesisQwen36VisionModel", "GenesisQwen36DecoderLayer", "ActionExpertBlock"]
        )
    return FSDPConfig(
        wrap_modules=["Qwen3_5MoeVisionBlock", "GenesisQwen36DecoderLayer", "ActionExpertBlock"]
    )


def _local_tensor(value: Tensor) -> Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def _assert_master_ownership(policy: PerceptronIsaacPolicy, optimizer: Optimizer) -> None:
    report = inspect_fsdp2_trainable_masters(policy)
    named = {name: param for name, param in policy.named_parameters() if param.requires_grad}
    assert report.units > 1
    assert not report.low_precision and not report.uncovered_trainable_names
    assert set(report.trainable_param_names) == set(named)
    owned = [param for group in optimizer.param_groups for param in group["params"]]
    assert len(owned) == len({id(param) for param in owned}) == len(named)
    assert {id(param) for param in owned} == {id(param) for param in named.values()}
    for name, param in named.items():
        assert isinstance(param, DTensor), f"unsharded optimizer master: {name}"
        assert param.to_local().dtype == torch.float32, f"non-F32 optimizer master: {name}"


def _sharded_modality_worker(rank: int, directory: str, wrapping: ModalityWrap, empty_rank: bool) -> None:
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
        assert accelerator.num_processes == 2 and accelerator.device.type == "cpu"
        torch.manual_seed(71)
        policy, model = _tiny_native_policy(wrapping)
        normalizer = policy.get_training_loss_normalizer()
        stream = (
            TensorStream.collective_dummy()
            if empty_rank and rank == 1
            else _measured_modality_stream(wrapping)
        )
        batch: dict[str, Any] = {
            PERCEPTRON_ISAAC_STREAM_KEY: stream,
            PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY: [] if stream.collective_only else [0],
        }
        # No test-supplied denominators: actual stock staging and integer SUM planning.
        state = prepare_training_input(lambda: batch, normalizer, accelerator)
        window = plan_training_window(0, [state], normalizer, accelerator)
        local_counts = normalizer.local_counts(normalizer.decode_batch(state, torch.device("cpu")))
        assert local_counts == ((0, 0) if stream.collective_only else (1, 2))
        assert window.normalization.totals == ((1, 2) if empty_rank else (2, 4))
        config = _supported_wrap_config(wrapping, model)
        host = CpuFsdpPreparation(
            state=CpuFsdpState(config.build_plugin(), TorchDynamoPlugin()),
            torch_device_mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard_cp",)),
            parallelism_config=CpuMeshTopology(),
            is_main_process=accelerator.is_main_process,
        )
        prepared = fsdp2_prepare_model(host, policy)
        assert prepared is policy and isinstance(prepared, FSDPModule)
        units = [
            (name, module)
            for name, module in prepared.named_modules()
            if name.startswith(wrapping.encoder_path + ".") or name == wrapping.encoder_path
            if isinstance(module, FSDPModule)
        ]
        assert units, f"supported {wrapping.value} configuration did not shard the native modality"
        if wrapping == ModalityWrap.VISION_BLOCK:
            assert any(name.endswith("visual.blocks.0") for name, _ in units)
        if wrapping == ModalityWrap.VISION_CONTAINER:
            assert any(name == wrapping.encoder_path for name, _ in units)
        trace = ModalityTrace(rank, root, tuple(name for name, _ in units))
        trace.record(
            f"wrapped modality units={list(trace.required_units)}; collective_only={stream.collective_only}"
        )
        for name, unit in units:
            handles.extend(trace.observe_unit(name, unit))
        # Prepend before *any* language-model FSDP hook, not just a decoder layer's hook.
        handles.append(
            model.model.language_model.register_forward_pre_hook(trace.before_decoder, prepend=True)
        )
        handles.append(model.model.language_model.register_forward_hook(trace.after_decoder))
        optimizer = torch.optim.SGD(prepared.get_optim_params(), momentum=0.3)
        _assert_master_ownership(prepared, optimizer)
        gradients: dict[str, Tensor] = {}

        def observe_gradients(
            _optimizer: Optimizer, _args: tuple[object, ...], _kwargs: dict[str, object]
        ) -> None:
            _assert_master_ownership(prepared, optimizer)
            for name, param in prepared.named_parameters():
                if param.grad is not None:
                    assert isinstance(param.grad, DTensor), f"nonsharded gradient: {name}"
                    assert isinstance(param, DTensor)
                    assert param.grad.placements == param.placements
                    local = param.grad.to_local()
                    assert local.dtype == torch.float32 and bool(torch.isfinite(local).all()), name
                    gradients[name] = local.detach().clone()

        handles.append(optimizer.register_step_pre_hook(observe_gradients))
        prepared_optimizer = accelerator.prepare_optimizer(optimizer)
        tracker = MetricsTracker(
            1,
            2,
            1,
            {name: AverageMeter(name, ":.6f") for name in ("loss", "lr", "update_s")},
            dp_world_size=2,
        )
        active_batch = normalizer.decode_batch(state, accelerator.device)
        _, metrics = update_policy(
            tracker,
            prepared,
            active_batch,
            prepared_optimizer,
            0.0,
            accelerator,
            training_loss_normalization=window.normalization,
        )
        assert metrics is not None and all(math.isfinite(value) for value in metrics.values())
        assert trace.decoder_entered and trace.decoder_completed
        assert set(trace.required_units) <= trace.completed, "native sharded modality did not finish"
        assert gradients, "real backward produced no optimizer gradients"
        modality_gradients = [
            value for name, value in gradients.items() if name.startswith(wrapping.encoder_path + ".")
        ]
        assert modality_gradients, "native modality has no optimizer-owned gradients"
        assert any(bool(value.ne(0).any()) for value in modality_gradients), (
            "native modality gradient is all zero"
        )
        for prefix in (
            "_isaac_model.lm_head.",
            "_isaac_model.model.language_model.layers.0.",
            "_isaac_model.model.language_model.layers.1.",
            "_isaac_model.model.action_expert.",
        ):
            assert any(name.startswith(prefix) for name in gradients), f"loss bypassed native {prefix}"
        _assert_master_ownership(prepared, optimizer)
        for param in prepared.parameters():
            assert bool(torch.isfinite(_local_tensor(param)).all())
        receipt = {
            "rank": rank,
            "wrapping": wrapping.value,
            "collective_only": stream.collective_only,
            "local_counts": local_counts,
            "global_counts": window.normalization.totals,
            "required_units": trace.required_units,
            "entered": sorted(trace.entered),
            "completed": sorted(trace.completed),
            "gradient_names": sorted(gradients),
            "metrics": metrics,
        }
        (root / f"rank-{rank}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    except Exception:
        original = traceback.format_exc()
        (root / f"rank-{rank}.exception").write_text(original, encoding="utf-8")
        print(f"rank {rank} sharded modality worker failed", file=sys.stderr, flush=True)
        print(original, file=sys.stderr, flush=True)
        raise
    finally:
        for handle in handles:
            handle.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_sharded_modalities(tmp_path: Path, wrapping: ModalityWrap, *, empty_rank: bool) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    context = mp.spawn(
        _sharded_modality_worker, args=(str(tmp_path), wrapping, empty_rank), nprocs=2, join=False
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("two-Gloo modality gate exceeded 180s; timeout alone is NOT product RED")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        # Spawn may pick a peer EOF first. Preserve *all* original exceptions and bounded traces.
        for rank in range(2):
            for suffix in ("trace", "exception"):
                evidence = tmp_path / f"rank-{rank}.{suffix}"
                if evidence.is_file():
                    print(f"--- {evidence.name} ---\n{evidence.read_text(encoding='utf-8')}", file=sys.stderr)
    for rank in range(2):
        assert (tmp_path / f"rank-{rank}.json").is_file(), f"rank {rank} did not finish real update_policy"


@pytest.mark.parametrize("wrapping", list(ModalityWrap), ids=lambda wrapping: wrapping.value)
def test_stock_empty_rank_enters_sharded_modality_before_decoder(
    tmp_path: Path, wrapping: ModalityWrap
) -> None:
    _run_sharded_modalities(tmp_path, wrapping, empty_rank=True)


@pytest.mark.parametrize("wrapping", list(ModalityWrap), ids=lambda wrapping: wrapping.value)
def test_stock_both_ranks_modality_finishes_real_fsdp2_forward_backward(
    tmp_path: Path, wrapping: ModalityWrap
) -> None:
    _run_sharded_modalities(tmp_path, wrapping, empty_rank=False)
