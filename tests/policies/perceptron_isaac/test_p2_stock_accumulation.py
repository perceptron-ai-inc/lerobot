"""P2.08 stock-loop contracts; runtime RED belongs to the parent CPU gate.

Only dataset/model/token serialization and synthetic model-artifact IO are leaves.
The real train loop, pack filtering/normalization, policy forward/counting,
Accelerator/DDP, update_policy and training-state save/resume remain in circuit.
No test supplies update-window denominators or runs a replacement trainer.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as functional
from accelerate import Accelerator
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

pytest.importorskip("transformers", reason="requires the parent offline ISAAC environment")
pytest.importorskip("mharmony", reason="requires the parent offline ISAAC environment")

from lerobot.common.train_utils import save_checkpoint
from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.distributed.factory import make_accelerator
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.optimizers import SGDConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig
from lerobot.policies.perceptron_isaac.isaac_stats import IsaacNormalizationStats, IsaacStatsBlock
from lerobot.policies.perceptron_isaac.loss_perceptron_isaac import (
    ISAAC_FLOW_LOSS_MASK_KEY,
    ISAAC_FLOW_NOISE_KEY,
    ISAAC_FLOW_TIMESTEPS_KEY,
)
from lerobot.policies.perceptron_isaac.mharmony_native import (
    IsaacActionOutlierError,
    IsaacMharmonyContentPlan,
    IsaacMharmonyRenderMetadata,
    IsaacNativeMharmonyRenderer,
)
from lerobot.policies.perceptron_isaac.mk1_checkpoint_contract import Mk1CheckpointContract
from lerobot.policies.perceptron_isaac.modeling_mk1_vla import (
    Mk1Qwen36VLAForActionGeneration,
    _build_config,
)
from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.modeling_qwen35_vla import ModelOutput
from lerobot.policies.perceptron_isaac.processor_perceptron_isaac import (
    PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY,
    PERCEPTRON_ISAAC_STREAM_KEY,
    PerceptronIsaacMharmonyPackProcessorStep,
)
from lerobot.policies.perceptron_isaac.tensor_stream import Event, TensorStream, TextType, create_stream
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    ImageAugmentationProcessorStep,
    PolicyProcessorPipeline,
    batch_to_transition,
    transition_to_batch,
)
from lerobot.scripts import lerobot_train
from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig
from lerobot.utils.constants import ACTION, OBS_STATE, PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig, DummyCheckpointPolicy
from tests.policies.perceptron_isaac.test_mk1_checkpoint_contract import _config as _native_config
from tests.policies.perceptron_isaac.test_modeling_training import _config


class Case(str, Enum):
    JOINT = "joint"
    EMPTY_CHUNKS = "empty_chunks"
    NATIVE_EMPTY_CHUNKS = "native_empty_chunks"
    EMPTY_TEXT = "empty_text"
    CROSSING = "crossing"
    TAIL = "tail"
    AUXILIARY = "auxiliary"
    ORDINARY = "ordinary"
    RESUME = "resume"


@dataclass(frozen=True)
class Run:
    case: Case
    steps: int = 2
    accumulation: int = 2
    draws: int = 2
    checkpoint: Path | None = None
    interrupt_at: int | None = None


@dataclass(frozen=True)
class ExpertArgs:
    action_dim: int = 2
    action_horizon: int = 2
    clean_at_0: bool = False


class DrawRecord(TypedDict):
    x: Tensor
    tau: Tensor


class ForwardRecord(TypedDict):
    indices: Tensor
    images: Tensor
    targets: Tensor
    pads: Tensor
    labels: Tensor
    mask: Tensor
    draws: DrawRecord
    metrics: dict[str, float]
    positions: int
    padding: int


class NativeRouterRecord(TypedDict):
    tokens: list[Tensor]
    dispatch: list[Tensor]


class Receipt(TypedDict):
    initial: dict[str, Tensor]
    final: dict[str, Tensor]
    gradients: list[dict[str, Tensor]]
    pending: dict[str, Tensor]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any] | None
    forwards: list[ForwardRecord]
    preprocessed: list[Tensor]
    uniform: Tensor
    normal: Tensor
    augmentation: Tensor | None
    backward_divisor: int
    native_router: list[NativeRouterRecord]
    policy_updates: int


class CadencePolicy(DummyCheckpointPolicy):
    """Ordinary forward signature; update() counts only landed optimizer updates."""

    def __init__(self, config: DummyCheckpointConfig) -> None:
        super().__init__(config)
        self.applied_updates = 0

    def update(self) -> None:
        self.applied_updates += 1


class NativeCaptureModel(Mk1Qwen36VLAForActionGeneration):
    """Observe real recorder/materialization outputs without changing native computation."""

    def __init__(self) -> None:
        raw = _native_config(test_geometry=True, mtp_present=False)
        raw["text_config"]["layer_types"] = ["full_attention", "full_attention"]
        raw["text_config"]["full_attention_interval"] = 1
        raw["genesis_vla"]["action_expert"].update(action_dim=2, action_horizon=2)
        contract = Mk1CheckpointContract.parse_allowlisted(raw, allow_test_only_reduced_geometry=True)
        super().__init__(_build_config(raw, contract))
        self.router_records: list[NativeRouterRecord] = []

    def train_forward(self, stream: TensorStream) -> ModelOutput:
        output = super().train_forward(stream)
        self.router_records.append(
            {
                "tokens": [value.detach().clone() for value in output.additional_info["router_token_count"]],
                "dispatch": [
                    value.detach().clone() for value in output.additional_info["router_dispatch_counts"]
                ],
            }
        )
        return output


class RecordingExpert(nn.Module):
    """Tiny model leaf; randomness is sampled by the production loss, not injected on resume."""

    def __init__(self) -> None:
        super().__init__()
        self.args = ExpertArgs()
        self.scale = nn.Parameter(torch.tensor(0.3))

    def sample_timesteps(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype)

    def forward(
        self, vlm_activations: Tensor, vlm_mask: Tensor, x_tau: Tensor, tau: Tensor, **kwargs: Any
    ) -> Tensor:
        del vlm_activations, vlm_mask, kwargs
        del tau
        return x_tau * self.scale


class StockModel(nn.Module):
    """Small train_forward leaf with separate task and auxiliary parameter gradients."""

    def __init__(self, auxiliary: bool) -> None:
        super().__init__()
        self.action_expert = RecordingExpert()
        self.context_seed = nn.Parameter(torch.tensor([0.1, -0.2, 0.3, 0.4]))
        self.lm_head = nn.Linear(4, 32, bias=False)
        self.router = nn.Parameter(torch.tensor(0.7), requires_grad=auxiliary)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.linspace(-0.3, 0.5, 128).reshape(32, 4))

    def train_forward(self, stream: TensorStream) -> ModelOutput:
        batch, length = stream.shape
        activations = self.context_seed.view(1, 1, -1).expand(batch, length - 1, -1)
        return ModelOutput(
            final_activations=activations,
            final_embedding=self.lm_head,
            flow_matching_expert=self.action_expert,
            additional_info=(
                {"load_balancing_loss": [self.router.square()], "router_z_loss": [self.router.pow(4)]}
                if self.router.requires_grad
                else {}
            ),
        )


class ArtifactIOPolicy(PerceptronIsaacPolicy):
    """Synthetic model-artifact IO only; forward and normalization capability are inherited.

    This is not an authenticated ISAAC package fixture. It isolates package copying
    from the stock trainer's real optimizer/RNG/pending-window checkpoint contract.
    No production reader/hash check is patched or waived.
    """

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        assert state_dict is None
        PreTrainedPolicy._save_pretrained(self, save_directory)

    def finalize_pretrained_package(self, save_directory: str | Path) -> None:
        PreTrainedPolicy.finalize_pretrained_package(self, save_directory)


class TokenLeaf:
    def __call__(self, normalized_actions: np.ndarray) -> list[list[int]]:
        assert normalized_actions.shape == (1, 2, 2)
        return [[9, 2]]


class SerializationLeaf(IsaacNativeMharmonyRenderer):
    """Replace vocabulary/image IO, not build_training's normalization or outlier decision."""

    def _render_plan_to_tensor_stream(
        self,
        plan: IsaacMharmonyContentPlan,
        *,
        device: Any,
        dtype: Any,
        patch_size: int | None,
        max_num_patches: int | None,
        min_num_patches: int | None,
        pixel_shuffle_scale: int | None,
        temporal_patch_size: int | None,
        rewrite_final_assistant_footer: bool = True,
    ) -> TensorStream:
        del dtype, patch_size, max_num_patches, min_num_patches, pixel_shuffle_scale
        del temporal_patch_size, rewrite_final_assistant_footer
        # Task supplies only the tiny vocabulary leaf's text length/role, not counts to train().
        task = plan.preamble.splitlines()[0].split()
        label_count = int(task[1])
        text_role = "user" if task[0] == "no_text" else "assistant"
        flow = plan.assistant_content[0]
        labels = [9, 2, 10][:label_count]
        events = [
            Event.from_text_tokens(
                torch.tensor([3, 4], device=device), time=(0.0, 0.0), type=TextType.text, role="user"
            ),
            Event.from_text_tokens(
                torch.tensor([0], device=device),
                time=(0.0, 0.0),
                type=TextType.action_c,
                role="assistant",
                tags=flow["tags"],
            ),
            Event.from_text_tokens(
                torch.tensor([8, *labels], device=device),
                time=(0.0, 0.0),
                type=TextType.action,
                role=text_role,
            ),
        ]
        return TensorStream([create_stream(events, list(TextType))])


@dataclass
class Metadata:
    frames: int
    stats: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    camera_keys: list[str] = field(default_factory=lambda: ["observation.images.image"])
    depth_keys: list[str] = field(default_factory=list)
    has_language_columns: bool = False
    fps: int = 20

    @property
    def episodes(self) -> dict[str, list[int]]:
        return {"dataset_from_index": [0], "dataset_to_index": [self.frames]}


class LocalDataset(Dataset[dict[str, Any]]):
    """Finite data leaf; real EpisodeAwareSampler and Accelerator shard it."""

    def __init__(self, run: Run) -> None:
        self.num_frames = 6 if run.case == Case.CROSSING else 12
        self.num_episodes = 1
        self.meta = Metadata(self.num_frames)
        self.episodes = None
        self.absolute_to_relative_idx = None
        self.image_transforms = None
        order = list(EpisodeAwareSampler([0], [self.num_frames], shuffle=True, seed=83))
        self.positions = {index: position for position, index in enumerate(order)}
        self.run = run

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = self.positions[index]
        global_batch, within = divmod(position, 3)
        retained = (1, 2, 3, 1)[global_batch]
        if self.run.case in {Case.CROSSING, Case.RESUME}:
            retained = (1, 3, 3, 3)[global_batch]  # Only two rejected rows; every later triple is usable.
        if self.run.case in {Case.EMPTY_CHUNKS, Case.NATIVE_EMPTY_CHUNKS} and global_batch % 2 == 0:
            retained = 0
        if self.run.accumulation == 1 or self.run.case in {Case.TAIL, Case.ORDINARY}:
            retained = 3
        target = torch.tensor([[0.2 + position / 10, -0.4], [0.8, 0.1 + within / 5]])
        if within >= retained:
            target.fill_(100.0)  # Real normalize -> outlier -> pack retention, not a test-side filter.
        no_text = self.run.case == Case.EMPTY_TEXT and global_batch % 2 == 0
        return {
            "index": torch.tensor(index),
            "frame_index": torch.tensor(index),
            OBS_STATE: torch.tensor([index / 20, 0.2]),
            "observation.images.image": torch.full((3, 8, 8), 0.25 + index / 100),
            ACTION: target,
            "action_is_pad": torch.tensor([False, within == 0]),
            "task": f"{'no_text' if no_text else 'labels'} {1 + position % 3}",
        }


class PreparedProcessor:
    """Use real pack pipeline, then add deterministic per-draw oracle fixture tensors."""

    def __init__(self, run: Run) -> None:
        self.run = run
        self.indices: list[Tensor] = []
        block = IsaacStatsBlock(q01=np.full(2, -1.0), q99=np.full(2, 1.0))
        stats = IsaacNormalizationStats(block, block, 20.0, 2, "flow_matching_stats_v1")
        metadata = IsaacMharmonyRenderMetadata(
            camera_order=["image"],
            image_size=[8, 8],
            n_obs_steps=1,
            action_horizon=2,
            action_dim=2,
            proprio_dim=2,
        )
        step = PerceptronIsaacMharmonyPackProcessorStep(
            enabled=True,
            camera_order=["image"],
            image_keys=["observation.images.image"],
            image_size=(8, 8),
            n_obs_steps=1,
            chunk_size=2,
            action_dim=2,
            proprio_dim=2,
            normalize_language=False,
            dtype="float32",
            train_skip_outlier_threshold=20.0,
        )
        step._stats = stats
        step._stream_builder = SerializationLeaf(metadata=metadata, stats=stats)
        step._fast_processor = TokenLeaf()
        self.pipeline: PolicyProcessorPipeline = PolicyProcessorPipeline(
            steps=[step], to_transition=batch_to_transition, to_output=transition_to_batch
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.indices.append(batch["index"].clone())
        prepared = self.pipeline(batch)
        if self.run.case == Case.RESUME:
            return prepared  # No injected noise or timesteps on the checkpoint path.
        count = batch[ACTION].shape[0]
        masks = (~batch["action_is_pad"])[None, :, :, None].expand(self.run.draws, -1, -1, 2).float().clone()
        masks[-1, :, 0, 0] = 0.25
        prepared[ISAAC_FLOW_LOSS_MASK_KEY] = masks
        prepared[ISAAC_FLOW_NOISE_KEY] = (
            torch.arange(self.run.draws * count * 4).reshape(self.run.draws, count, 2, 2).float() / 10
        )
        prepared[ISAAC_FLOW_TIMESTEPS_KEY] = torch.linspace(0.2, 0.8, self.run.draws)[:, None].expand(
            -1, count
        )
        return prepared

    def save_pretrained(self, directory: Path) -> None:
        self.pipeline.save_pretrained(directory)


class StopAtCheckpointError(Exception):
    pass


def _named_gradients(policy: nn.Module) -> dict[str, Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in policy.named_parameters()
        if param.grad is not None
    }


def _record(batch: dict[str, Any], draw: DrawRecord, metrics: dict[str, float], draws: int) -> ForwardRecord:
    stream = batch[PERCEPTRON_ISAAC_STREAM_KEY]
    targets, pads, labels = [], [], []
    padding = 0
    for sample in stream.streams:
        for event in sample.events:
            if event.type == TextType.action_c:
                targets.append(event.tags["action_target"])
                pads.append(event.tags["action_is_pad"])
            elif event.type == TextType.action and event.role == "assistant":
                labels.extend(event.data.reshape(-1)[1:].tolist())
            elif event.type == TextType.padding:
                padding += event.num_tokens()
    kept = torch.tensor(batch[PERCEPTRON_ISAAC_KEPT_SAMPLE_INDICES_KEY], dtype=torch.long)
    if kept.numel() == 0:
        targets, pads, labels = [], [], []  # Collective-only dummy is not real supervision.
    target_tensor = torch.tensor(targets).reshape(-1, 2, 2)
    pad_tensor = torch.tensor(pads, dtype=torch.bool).reshape(-1, 2)
    mask = (~pad_tensor)[None, :, :, None].expand(draws, -1, -1, 2).float().clone()
    if ISAAC_FLOW_LOSS_MASK_KEY in batch:
        mask = batch[ISAAC_FLOW_LOSS_MASK_KEY].index_select(1, kept)
    return {
        "indices": batch["index"].clone(),
        "images": batch["observation.images.image"].clone(),
        "targets": target_tensor,
        "pads": pad_tensor,
        "labels": torch.tensor(labels, dtype=torch.long),
        "mask": mask,
        "draws": copy.deepcopy(draw),
        "metrics": metrics,
        "positions": sum(sum(event.num_tokens() for event in sample.events) - 1 for sample in stream.streams),
        "padding": padding,
    }


def _worker(rank: int, rendezvous: str, directory: str, run: Run) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29500", OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    root = Path(directory)
    dataset = LocalDataset(run)
    processor = PreparedProcessor(run)
    torch.manual_seed(71)
    model = (
        NativeCaptureModel().float()
        if run.case == Case.NATIVE_EMPTY_CHUNKS
        else StockModel(run.case == Case.AUXILIARY)
    )
    policy_config = _config()
    policy_config.push_to_hub = False
    policy_config.train_samples_per_chunk = run.draws
    policy = ArtifactIOPolicy(policy_config)
    policy._isaac_model = model
    policy._training_parameters_configured = True
    ordinary = CadencePolicy(DummyCheckpointConfig(device="cpu", hidden=2, push_to_hub=False))
    with torch.no_grad():
        ordinary.net.weight.fill_(0.5)
        ordinary.net.bias.zero_()
    active: PreTrainedPolicy = ordinary if run.case == Case.ORDINARY else policy
    assert type(policy).forward is PerceptronIsaacPolicy.forward
    initial = {name: value.detach().clone() for name, value in active.state_dict().items()}
    forwards: list[ForwardRecord] = []
    gradients: list[dict[str, Tensor]] = []
    optimizers: list[Optimizer] = []
    schedulers: list[LRScheduler] = []
    accelerators: list[Accelerator] = []
    augmentations: list[ImageAugmentationProcessorStep] = []
    draw: DrawRecord = {"x": torch.empty(0), "tau": torch.empty(0)}
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/stock-window"),
        policy=active.config,
        output_dir=root / "output",
        seed=83,
        steps=run.steps,
        batch_size=3,
        num_workers=0,
        persistent_workers=False,
        dataloader_multiprocessing_context=None,
        log_freq=0,
        env_eval_freq=0,
        save_checkpoint=run.interrupt_at is not None,
        save_freq=run.interrupt_at or 0,
        use_policy_training_preset=False,
        optimizer=SGDConfig(lr=0.01, momentum=0.3, grad_clip_norm=0.0),
        scheduler=ConstantWithWarmupSchedulerConfig(num_warmup_steps=0),
        accelerator=AcceleratorConfig(
            gradient_accumulation=GradientAccumulationConfig(steps=run.accumulation)
        ),
    )
    if run.checkpoint is not None:
        saved_config = run.checkpoint / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME
        cfg = TrainPipelineConfig.from_pretrained(
            saved_config,
            cli_args=["--resume=true", f"--steps={run.steps}", f"--output_dir={root / 'output'}"],
        )
        assert cfg.policy is not None
        cfg.policy.push_to_hub = False
        cfg.save_checkpoint = False
        active.config = cfg.policy
        active.load_state_dict(load_file(run.checkpoint / PRETRAINED_MODEL_DIR / "model.safetensors"))

    real_optimizer_factory = make_optimizer_and_scheduler
    real_accelerator_factory = make_accelerator
    real_save = save_checkpoint

    def optimizer_factory(
        config: TrainPipelineConfig, supplied_policy: PreTrainedPolicy
    ) -> tuple[Optimizer, LRScheduler]:
        optimizer, scheduler = real_optimizer_factory(config, supplied_policy)
        assert scheduler is not None
        optimizer.register_step_pre_hook(lambda *_: gradients.append(_named_gradients(supplied_policy)))
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        return optimizer, scheduler

    def accelerator_factory(config: TrainPipelineConfig) -> Accelerator:
        accelerator = real_accelerator_factory(config)
        assert accelerator.device.type == "cpu"
        assert accelerator.num_processes == 2
        accelerators.append(accelerator)
        return accelerator

    def augmentation_factory(*_args: Any) -> ImageAugmentationProcessorStep | None:
        if run.case != Case.RESUME:
            return None
        # Same real augmentation implementation and private generator, placed on CPU for this gate.
        augmentation = ImageAugmentationProcessorStep(
            config=ImageTransformsConfig(
                enable=True,
                max_num_transforms=1,
                tfs={
                    "brightness": ImageTransformConfig(type="ColorJitter", kwargs={"brightness": (0.5, 1.5)})
                },
            ),
            image_keys=["observation.images.image"],
            device="cpu",
            seed=193 + rank,
        )
        augmentations.append(augmentation)
        return augmentation

    def save_then_interrupt(*args: Any, **kwargs: Any) -> None:
        real_save(*args, **kwargs)
        if kwargs["step"] == run.interrupt_at:
            raise StopAtCheckpointError

    def observe_forward(
        _module: nn.Module, args: tuple[Any, ...], result: tuple[Tensor, dict[str, float]]
    ) -> None:
        forwards.append(_record(args[0], draw, result[1], run.draws))

    def observe_expert(_module: nn.Module, args: tuple[Any, ...], _result: Tensor) -> None:
        draw["x"], draw["tau"] = args[2].detach().clone(), args[3].detach().clone()

    handle = active.register_forward_hook(observe_forward) if run.case != Case.ORDINARY else None
    expert_handle = model.action_expert.register_forward_hook(observe_expert)
    argv = ["lerobot-train"]
    if run.checkpoint is not None:
        argv.append(f"--config_path={run.checkpoint / PRETRAINED_MODEL_DIR / TRAIN_CONFIG_NAME}")
    try:
        with (
            patch.object(sys, "argv", argv),
            patch.object(lerobot_train, "make_train_eval_datasets", return_value=(dataset, None)),
            patch.object(lerobot_train, "make_policy", return_value=active),
            patch.object(lerobot_train, "make_pre_post_processors", return_value=(processor, processor)),
            patch.object(lerobot_train, "make_optimizer_and_scheduler", side_effect=optimizer_factory),
            patch.object(lerobot_train, "make_accelerator", side_effect=accelerator_factory),
            patch.object(lerobot_train, "_make_image_augmentation", side_effect=augmentation_factory),
            patch.object(lerobot_train, "save_checkpoint", side_effect=save_then_interrupt),
        ):
            try:
                lerobot_train.train(cfg)
            except StopAtCheckpointError:
                assert run.interrupt_at is not None
            except (IsaacActionOutlierError, ValueError) as error:
                if run.case in {Case.EMPTY_CHUNKS, Case.NATIVE_EMPTY_CHUNKS, Case.EMPTY_TEXT} and (
                    "no usable samples" in str(error)
                    or "no agent-role text/FAST labels" in str(error)
                    or "no flow action markers" in str(error)
                ):
                    raise AssertionError(
                        f"stock train rejected legitimate {run.case.value} support: {error}"
                    ) from error
                raise  # Setup/malformed-input failures are NOT accepted as P2.08 RED.
        assert len(accelerators) == len(optimizers) == len(schedulers) == 1
        accelerator = accelerators[0]
        augmentation_state = None
        if augmentations:
            generator = augmentations[0]._generator
            assert generator is not None
            augmentation_state = generator.get_state()
        receipt: Receipt = {
            "initial": initial,
            "final": {name: value.detach().clone() for name, value in active.state_dict().items()},
            "gradients": gradients,
            "pending": _named_gradients(active),
            "optimizer": optimizers[0].state_dict(),
            "scheduler": schedulers[0].state_dict(),
            "scaler": None if accelerator.scaler is None else accelerator.scaler.state_dict(),
            "forwards": forwards,
            "preprocessed": processor.indices,
            "uniform": torch.rand(8),
            "normal": torch.randn(8),
            "augmentation": augmentation_state,
            "backward_divisor": accelerator.gradient_accumulation_steps,
            "native_router": model.router_records if isinstance(model, NativeCaptureModel) else [],
            "policy_updates": ordinary.applied_updates,
        }
        root.mkdir(parents=True, exist_ok=True)
        torch.save(receipt, root / f"rank-{rank}.pt")
    except Exception:
        # A peer disconnect can win spawn's race to report the original rank failure.
        print(f"rank {rank} stock worker failed", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if handle is not None:
            handle.remove()
        expert_handle.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


def _launch(tmp_path: Path, run: Run) -> list[Receipt]:
    tmp_path.mkdir(parents=True)
    context = mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), str(tmp_path), run), nprocs=2, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail(
                    "stock train two-Gloo gate exceeded 180s; timeout is NOT accepted as assertion RED"
                )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]


def _oracle_loss(model: StockModel, records: list[ForwardRecord], auxiliary: bool) -> tuple[Tensor, Tensor]:
    flow_chunks, text_labels = [], []
    for record in records:
        if record["targets"].numel() == 0:
            continue
        x, tau = record["draws"]["x"], record["draws"]["tau"]
        if x.ndim == 3:
            x, tau = x.unsqueeze(0), tau.unsqueeze(0)
        tau_grid = tau[:, :, None, None]
        target = record["targets"][None] * (~record["pads"])[None, :, :, None]
        noise = (x - (1 - tau_grid) * target) / tau_grid
        error = (model.action_expert.scale * x - (target - noise)).square()
        mask = record["mask"]
        flow_chunks.append(((error * mask).sum((-2, -1)) / mask.sum((-2, -1))).mean(0))
        text_labels.append(record["labels"])
    flow = torch.cat(flow_chunks).mean()
    labels = torch.cat(text_labels)
    logits = model.context_seed @ model.lm_head.weight.t()
    text = functional.cross_entropy(logits.expand(labels.numel(), -1), labels)
    text += 0.0001 * logits.logsumexp(-1).square()
    primary = flow + text
    aux = 0.02 * model.router.square() + 0.001 * model.router.pow(4) if auxiliary else primary.new_zeros(())
    return primary, aux


def _assert_reference(receipts: list[Receipt], run: Run) -> None:
    reference = StockModel(run.case == Case.AUXILIARY)
    reference.load_state_dict(
        {key.removeprefix("_isaac_model."): value for key, value in receipts[0]["initial"].items()}
    )
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.3)
    for start in range(0, run.steps, run.accumulation):
        stop = min(start + run.accumulation, run.steps)
        records = [record for receipt in receipts for record in receipt["forwards"][start:stop]]
        primary, aux = _oracle_loss(reference, records, run.case == Case.AUXILIARY)
        # A partial tail uses Accelerator's real divisor, not its shorter buffered length.
        fraction = (stop - start) / run.accumulation
        (primary + fraction * aux).backward()
        if stop - start == run.accumulation:
            for receipt in receipts:
                observed_gradients = receipt["gradients"][start // run.accumulation]
                for name, parameter in reference.named_parameters():
                    if parameter.grad is not None:
                        torch.testing.assert_close(
                            observed_gradients[f"_isaac_model.{name}"], parameter.grad, atol=2e-6, rtol=2e-5
                        )
            optimizer.step()
            optimizer.zero_grad()
        observed = (
            sum(record["metrics"]["loss"] for record in receipts[0]["forwards"][start:stop])
            / run.accumulation
        )
        torch.testing.assert_close(
            torch.tensor(observed), (primary + fraction * aux).detach(), atol=2e-5, rtol=2e-5
        )
    for receipt in receipts:
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(
                receipt["final"][f"_isaac_model.{name}"], parameter, atol=2e-6, rtol=2e-5
            )
        assert len(receipt["gradients"]) == run.steps // run.accumulation
        assert receipt["scheduler"]["last_epoch"] == run.steps
        assert receipt["backward_divisor"] == run.accumulation


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo required"
)


@pytest.mark.parametrize("draws", [1, 2])
def test_stock_train_global_window_matches_concatenated_joint_objective(tmp_path: Path, draws: int) -> None:
    run = Run(Case.JOINT, draws=draws)
    receipts = _launch(tmp_path / "joint", run)
    _assert_reference(receipts, run)


def test_stock_train_counts_after_real_pack_filtering(tmp_path: Path) -> None:
    run = Run(Case.JOINT)
    receipts = _launch(tmp_path / "filtering", run)
    counts = [[len(record["targets"]) for record in receipt["forwards"]] for receipt in receipts]
    assert counts == [[1, 3], [2, 1]], "fixture must exercise unequal real outlier retention"
    assert sum(record["padding"] for receipt in receipts for record in receipt["forwards"]) > 0
    labels = [sum(record["labels"].numel() for record in receipt["forwards"]) for receipt in receipts]
    assert labels[0] != labels[1], "text normalization must be independent of retained chunks"
    _assert_reference(receipts, run)


def test_stock_train_zero_local_chunks_matches_global_reference(tmp_path: Path) -> None:
    run = Run(Case.EMPTY_CHUNKS)
    receipts = _launch(tmp_path / "empty-chunks", run)
    assert all(record["targets"].numel() == 0 for record in receipts[0]["forwards"])
    _assert_reference(receipts, run)


def test_stock_train_empty_rank_native_router_excludes_dummy_but_keeps_real_padding(tmp_path: Path) -> None:
    run = Run(Case.NATIVE_EMPTY_CHUNKS)
    receipts = _launch(tmp_path / "native-empty", run)
    assert sum(record["padding"] for record in receipts[1]["forwards"]) > 0
    for receipt in receipts:
        assert len(receipt["gradients"]) == 1
        assert len(receipt["native_router"]) == run.steps
        assert all(torch.isfinite(value).all() for value in receipt["final"].values())
        for step, routers in enumerate(receipt["native_router"]):
            # Rank zero has no retained samples. Rank one's actual NTP extent includes real padding.
            real_positions = receipts[1]["forwards"][step]["positions"]
            assert len(routers["tokens"]) == len(routers["dispatch"]) == 2
            for tokens, dispatch in zip(routers["tokens"], routers["dispatch"], strict=True):
                assert tokens.item() == real_positions, "collective dummy must add no router positions"
                assert dispatch.sum().item() == 2 * real_positions, "top-k lanes sum over real positions only"
    _assert_tree_equal(receipts[0]["final"], receipts[1]["final"])


def test_stock_train_zero_local_text_preserves_flow_objective(tmp_path: Path) -> None:
    run = Run(Case.EMPTY_TEXT)
    receipts = _launch(tmp_path / "empty-text", run)
    assert all(record["labels"].numel() == 0 for record in receipts[0]["forwards"])
    _assert_reference(receipts, run)


def test_stock_train_window_crosses_dataloader_boundary(tmp_path: Path) -> None:
    run = Run(Case.CROSSING)
    receipts = _launch(tmp_path / "crossing", run)
    # Six raw frames / (three samples * two ranks): every window spans two epochs.
    assert all(len(receipt["preprocessed"]) == 2 for receipt in receipts)
    sampler = EpisodeAwareSampler([0], [6], shuffle=True, seed=83)
    for step in range(2):
        order = list(sampler)
        for rank, receipt in enumerate(receipts):
            torch.testing.assert_close(
                receipt["forwards"][step]["indices"], torch.tensor(order[rank * 3 : (rank + 1) * 3])
            )
    _assert_reference(receipts, run)


def test_stock_train_partial_tail_preserves_update_and_scheduler_counters(tmp_path: Path) -> None:
    run = Run(Case.TAIL, steps=3)
    receipts = _launch(tmp_path / "tail", run)
    _assert_reference(receipts, run)
    assert all(receipt["pending"] for receipt in receipts), "no forced partial-tail optimizer step"


@pytest.mark.parametrize(("accumulation", "steps"), [(1, 2), (2, 2), (2, 3)])
def test_stock_train_ordinary_policy_preserves_signature_and_cadence(
    tmp_path: Path, accumulation: int, steps: int
) -> None:
    run = Run(Case.ORDINARY, accumulation=accumulation, steps=steps)
    receipts = _launch(tmp_path / "ordinary", run)
    reference = DummyCheckpointPolicy(DummyCheckpointConfig(device="cpu", hidden=2, push_to_hub=False))
    reference.load_state_dict(receipts[0]["initial"])
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.3)
    for start in range(0, run.steps, accumulation):
        indices = torch.cat(
            [
                indices
                for receipt in receipts
                for indices in receipt["preprocessed"][start : start + accumulation]
            ]
        )
        states = torch.stack((indices / 20, torch.full_like(indices, 0.2, dtype=torch.float32)), dim=-1)
        reference.net(states).mean().backward()
        if start + accumulation <= steps:
            optimizer.step()
            optimizer.zero_grad()
    for receipt in receipts:
        assert len(receipt["gradients"]) == run.steps // accumulation
        assert receipt["scheduler"]["last_epoch"] == run.steps
        assert receipt["policy_updates"] == run.steps // accumulation
        assert receipt["scaler"] is None  # CPU Accelerator has no fp16 scaler; never invent one.
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(receipt["final"][name], parameter)


def test_stock_train_accumulation_one_is_preserved(tmp_path: Path) -> None:
    run = Run(Case.JOINT, accumulation=1)
    _assert_reference(_launch(tmp_path / "one", run), run)


def test_stock_train_auxiliary_scaling_is_not_multiplied_by_window_size(tmp_path: Path) -> None:
    run = Run(Case.AUXILIARY)
    receipts = _launch(tmp_path / "auxiliary", run)
    initial = torch.tensor(0.7, requires_grad=True)
    auxiliary = 0.02 * initial.square() + 0.001 * initial.pow(4)
    expected = torch.autograd.grad(auxiliary, initial)[0]
    for receipt in receipts:
        torch.testing.assert_close(receipt["gradients"][0]["_isaac_model.router"], expected)


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


def _assert_checkpoint_resume(tmp_path: Path, steps: int, checkpoint_budget: int) -> None:
    uninterrupted = _launch(tmp_path / "full", Run(Case.RESUME, steps=steps))
    interrupted = _launch(tmp_path / "interrupted", Run(Case.RESUME, steps=checkpoint_budget, interrupt_at=1))
    checkpoint = tmp_path / "interrupted" / "output" / "checkpoints" / "000001"
    metadata = checkpoint / TRAINING_STATE_DIR / "training_step.json"
    assert metadata.is_file(), "real stock checkpoint must finish before simulating interruption"
    moved = tmp_path / "moved-checkpoint"
    checkpoint.rename(moved)
    resumed = _launch(tmp_path / "resumed", Run(Case.RESUME, steps=steps, checkpoint=moved))
    for full, prefix, continuation in zip(uninterrupted, interrupted, resumed, strict=True):
        for key in (
            "final",
            "optimizer",
            "scheduler",
            "scaler",
            "pending",
            "uniform",
            "normal",
            "augmentation",
        ):
            _assert_tree_equal(continuation[key], full[key])
        assert len(prefix["gradients"]) == 0, "save after nonsync step, before any optimizer update"
        assert len(continuation["gradients"]) == steps // 2
        # Reconstructed prefix is allowed once; compare logical suffix, not replay bookkeeping.
        suffix = continuation["forwards"][-(steps - 1) :]
        for actual, expected in zip(suffix, full["forwards"][1:], strict=True):
            _assert_tree_equal(actual["indices"], expected["indices"])
            _assert_tree_equal(actual["images"], expected["images"])
            _assert_tree_equal(actual["draws"], expected["draws"])
            _assert_tree_equal(actual["targets"], expected["targets"])
            _assert_tree_equal(actual["pads"], expected["pads"])
        _assert_tree_equal(prefix["forwards"][0]["draws"], full["forwards"][0]["draws"])


@pytest.mark.parametrize("steps", [3, 4])
def test_stock_train_midwindow_resume_matches_uninterrupted_run(tmp_path: Path, steps: int) -> None:
    _assert_checkpoint_resume(tmp_path, steps, checkpoint_budget=steps)


@pytest.mark.parametrize("steps", [3, 4])
def test_stock_train_saved_partial_tail_extends_without_losing_pending_window(
    tmp_path: Path, steps: int
) -> None:
    # The saved run knew only one available batch. Extending it must replan totals and replay the prefix.
    _assert_checkpoint_resume(tmp_path, steps, checkpoint_budget=1)
