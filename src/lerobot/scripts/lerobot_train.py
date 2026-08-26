#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import gc
import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    gather_fsdp_state_dicts,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_fsdp_optimizer_state,
    load_training_metadata,
    load_training_state,
    push_checkpoint_to_hub,
    save_checkpoint,
    should_save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.policies.processor_utils import RETAINED_SAMPLE_INDICES_KEY
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.import_utils import _peft_available, register_third_party_plugins, require_package
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

if TYPE_CHECKING or _peft_available:
    from peft import PeftModel
else:
    PeftModel = None

from .lerobot_eval import eval_policy_all


@contextmanager
def _make_eval_envs(cfg: TrainPipelineConfig) -> Iterator[dict[str, dict[int, Any]]]:
    """Create evaluation environments for one run and always dispose of them."""
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
    )
    try:
        yield envs
    finally:
        close_envs(envs)


def _dataloader_worker_kwargs(cfg: TrainPipelineConfig) -> dict[str, Any]:
    """Return worker-only DataLoader options, disabling them for single-process loading."""
    workers_enabled = cfg.num_workers > 0
    return {
        "prefetch_factor": cfg.prefetch_factor if workers_enabled else None,
        "persistent_workers": cfg.persistent_workers and workers_enabled,
        "multiprocessing_context": cfg.dataloader_multiprocessing_context if workers_enabled else None,
    }


def _compute_update_loss_denominators(
    policy: PreTrainedPolicy,
    batches: list[Any],
    accelerator: "Accelerator",
    *,
    denominator_fn: Callable | None = None,
) -> dict[str, torch.Tensor] | None:
    """Reduce policy-owned valid counts across all microbatches and ranks in one update."""
    if denominator_fn is None:
        denominator_fn = _get_loss_denominator_fn(policy, accelerator)
    if denominator_fn is None:
        return None

    local_totals: dict[str, torch.Tensor] = {}
    expected_keys: set[str] | None = None
    values_to_validate: list[torch.Tensor] = []
    names_to_validate: list[str] = []
    with torch.no_grad():
        for batch in batches:
            batch_counts = denominator_fn(batch)
            if not isinstance(batch_counts, dict) or not batch_counts:
                raise ValueError("Policy loss denominators must be a non-empty dictionary.")
            keys = set(batch_counts)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError("Policy loss denominator keys changed within one accumulation window.")
            for name, value in batch_counts.items():
                scalar = torch.as_tensor(value, device=accelerator.device, dtype=torch.float32)
                if scalar.numel() != 1:
                    raise ValueError(f"Policy loss denominator {name!r} must be one positive finite scalar.")
                scalar = scalar.reshape(())
                values_to_validate.append(scalar)
                names_to_validate.append(name)
                local_totals[name] = local_totals[name] + scalar if name in local_totals else scalar

    values = torch.stack(values_to_validate)
    invalid = ~torch.isfinite(values) | (values <= 0)
    if bool(invalid.any()):
        invalid_index = int(invalid.nonzero(as_tuple=False)[0].item())
        raise ValueError(
            f"Policy loss denominator {names_to_validate[invalid_index]!r} must be one positive finite scalar."
        )

    names = sorted(local_totals)
    reduced = accelerator.reduce(torch.stack([local_totals[name] for name in names]), reduction="sum")
    if not bool((torch.isfinite(reduced) & (reduced > 0)).all()):
        raise ValueError("Global policy loss denominators must be positive and finite.")
    return {name: reduced[index].detach() for index, name in enumerate(names)}


def _get_loss_denominator_fn(policy: PreTrainedPolicy, accelerator: "Accelerator") -> Callable | None:
    """Return a policy's optional update-wide denominator hook after unwrapping it."""
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    denominator_fn = getattr(unwrapped, "get_loss_denominators", None)
    return denominator_fn if callable(denominator_fn) else None


def _align_sample_weights_with_loss(
    sample_weights: torch.Tensor,
    per_sample_loss: torch.Tensor,
    batch: Any,
) -> torch.Tensor:
    """Select weights for samples retained by preprocessing and validate one-to-one alignment."""
    if sample_weights.ndim != 1:
        raise ValueError(f"Sample weights must be one-dimensional, got shape {tuple(sample_weights.shape)}.")
    if per_sample_loss.ndim == 0:
        loss_count = 1
    elif per_sample_loss.ndim == 1:
        loss_count = int(per_sample_loss.shape[0])
    else:
        raise ValueError(
            "Policies supporting sample weighting must return one scalar loss per retained sample; "
            f"got shape {tuple(per_sample_loss.shape)}."
        )

    retained_raw = batch.get(RETAINED_SAMPLE_INDICES_KEY) if isinstance(batch, dict) else None
    if retained_raw is not None:
        retained = torch.as_tensor(retained_raw, device="cpu", dtype=torch.long).reshape(-1)
        if retained.numel() != loss_count:
            raise ValueError(
                f"Retained sample indices contain {retained.numel()} rows, but the policy returned "
                f"{loss_count} per-sample losses."
            )
        if retained.numel() == 0:
            raise ValueError("A filtered training batch must retain at least one sample.")
        if bool((retained < 0).any()) or int(retained.max()) >= sample_weights.numel():
            raise ValueError(
                f"Retained sample indices {retained.tolist()} are outside the original weight batch "
                f"of size {sample_weights.numel()}."
            )
        if retained.unique().numel() != retained.numel():
            raise ValueError("Retained sample indices must be unique.")
        sample_weights = sample_weights.index_select(0, retained.to(device=sample_weights.device))

    if sample_weights.numel() != loss_count:
        raise ValueError(
            f"Sample weighting produced {sample_weights.numel()} weights for {loss_count} per-sample losses. "
            f"Processors that filter samples must publish {RETAINED_SAMPLE_INDICES_KEY!r}."
        )
    return sample_weights.to(device=per_sample_loss.device)


def _run_optimizer_update_microbatches(
    *,
    policy: PreTrainedPolicy,
    accelerator: "Accelerator",
    gradient_accumulation_steps: int,
    sample_weighter: Any,
    load_batch: Callable[[], Any],
    update_batch: Callable[[Any, dict[str, torch.Tensor] | None], None],
) -> float:
    """Run one optimizer update, buffering only policies that need update-wide denominators."""
    dataloading_s = 0.0
    denominator_fn = _get_loss_denominator_fn(policy, accelerator) if sample_weighter is None else None
    if denominator_fn is not None:
        batches = []
        for _ in range(gradient_accumulation_steps):
            started_at = time.perf_counter()
            batches.append(load_batch())
            dataloading_s += time.perf_counter() - started_at
        denominators = _compute_update_loss_denominators(
            policy,
            batches,
            accelerator,
            denominator_fn=denominator_fn,
        )
        for batch in batches:
            update_batch(batch, denominators)
        return dataloading_s

    # Existing policies normalize each equally-sized microbatch independently. Keep their
    # historical streaming behavior instead of retaining the full accumulation window on device.
    for _ in range(gradient_accumulation_steps):
        started_at = time.perf_counter()
        batch = load_batch()
        dataloading_s += time.perf_counter() - started_at
        update_batch(batch, None)
    return dataloading_s


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
    loss_denominators: dict[str, torch.Tensor] | None = None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    with accelerator.accumulate(policy):
        # These exact errors come from PyTorch's autograd output validation. Both have proven
        # transient on the Qwen3.5 hybrid graph: replaying the same checkpoint and sample
        # succeeds. Rebuild the consumed graph and retry, but only when there are no earlier
        # accumulated microbatch gradients to preserve. OOMs and every other RuntimeError remain
        # hard failures.
        autograd_attempts = 3 if getattr(accelerator, "gradient_accumulation_steps", 1) == 1 else 1
        cleaning_failed_graph = False
        for autograd_attempt in range(autograd_attempts):
            if cleaning_failed_graph:
                # The caught exception and its traceback are cleared when the preceding except
                # block exits. Collect only now, before rebuilding the graph; doing this inside
                # the except block leaves the traceback's references alive and can OOM the retry.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cleaning_failed_graph = False

            loss = None
            output_dict = None
            per_sample_loss = None
            failure_phase = "forward"
            try:
                # Let accelerator handle mixed precision and scale each microbatch loss
                # by the configured accumulation factor during backward.
                with accelerator.autocast():
                    if sample_weights is not None:
                        # Use per-sample loss for weighted training
                        # Note: Policies supporting sample weighting must implement
                        # forward(batch, reduction="none")
                        per_sample_loss, output_dict = policy.forward(batch, reduction="none")
                        if per_sample_loss.ndim == 0:
                            per_sample_loss = per_sample_loss.reshape(1)
                        aligned_sample_weights = _align_sample_weights_with_loss(
                            sample_weights, per_sample_loss, batch
                        )

                        # Weighted loss: each sample's contribution is scaled by its weight.
                        # We divide by weight sum (not batch size) so that if some weights are zero,
                        # the remaining samples contribute proportionally more, preserving gradient scale.
                        # Weights are pre-normalized to sum to batch_size for stable training dynamics.
                        epsilon = 1e-6
                        loss = (per_sample_loss * aligned_sample_weights).sum() / (
                            aligned_sample_weights.sum() + epsilon
                        )

                        # Log weighting statistics
                        if output_dict is None:
                            output_dict = {}
                        for key, value in weight_stats.items():
                            output_dict[f"sample_weight_{key}"] = value
                    else:
                        if loss_denominators is None:
                            loss, output_dict = policy.forward(batch)
                        else:
                            # Accelerator divides every backward loss by the accumulation count.
                            # The policy multiplies each numerator contribution by the same count,
                            # leaving one update-wide numerator/global-denominator objective.
                            loss, output_dict = policy.forward(
                                batch,
                                loss_denominators=loss_denominators,
                                loss_accumulation_scale=accelerator.gradient_accumulation_steps,
                            )

                    # TODO(rcadene): policy.unnormalize_outputs(out_dict)

                failure_phase = "backward"
                accelerator.backward(loss)
                break
            except RuntimeError as exc:
                error_text = str(exc)
                retryable_autograd_error = (
                    "isDifferentiableType(grad.scalar_type())" in error_text
                    or "Autograd not support dtype: Byte" in error_text
                )
                # The replay is only sound single-process: an aborted backward leaves
                # DDP's reducer mid-reduction, and retrying the forward on one rank while
                # its peers proceed dies with "Expected to have finished reduction in the
                # prior iteration" (this rank) and a closed-peer collective (the others).
                # Fail hard instead so the job restarts cleanly.
                if (
                    not retryable_autograd_error
                    or autograd_attempt + 1 >= autograd_attempts
                    or accelerator.num_processes > 1
                ):
                    raise
                # Backward may have populated an arbitrary subset of gradients; forward may also
                # have built a partial graph. Neither may leak into the retry.
                optimizer.zero_grad(set_to_none=True)
                logging.warning(
                    "PyTorch autograd hit its transient %s-pass dtype validator error; "
                    "retrying this batch (%d/%d): %s",
                    failure_phase,
                    autograd_attempt + 2,
                    autograd_attempts,
                    error_text.splitlines()[0],
                )
                # Rebinding (not `del`) drops the graph references while keeping the names
                # bound for the post-loop reads below.
                loss = output_dict = per_sample_loss = None
                cleaning_failed_graph = True
                continue

        grad_norm = None
        if accelerator.sync_gradients:
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )

        # AcceleratedOptimizer makes step/zero_grad no-ops on non-sync
        # microbatches, retaining accumulated gradients until the boundary.
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            if lr_scheduler is not None:
                lr_scheduler.step()
            if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
                accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    if accelerator.sync_gradients:
        if grad_norm is None:
            raise RuntimeError("Gradient synchronization completed without a gradient norm.")
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    # Aggregate the policy's scalar outputs for logging and rank-reduction across the log window.
    if output_dict:
        train_metrics.update_metrics(output_dict)
    return train_metrics, output_dict


def _should_pass_dataset_meta(cfg: TrainPipelineConfig, active_cfg) -> bool:
    """Whether the processor factory receives the training dataset's metadata.

    Native Isaac rendering validates the dataset clock and derives normalization from
    the dataset during fine-tuning (and on resume the metadata also exempts the
    serving split-brain check), and molmoact2 reads feature names from it -- so
    metadata stays available by default. GR00T is the exception outside reward-model
    training: its pipeline treats ``dataset_meta is not None`` as training mode, so an
    unconditional kwarg would silently enable state dropout and per-sample random
    crops for every GR00T finetune and stop reproducing prior runs (merge-base
    behaviour passed it only for reward-model training).
    """
    return cfg.is_reward_model_training or active_cfg.type != "groot"


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    require_package("accelerate", extra="training")
    from accelerate import Accelerator
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
    )

    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # find_unused_parameters defaults to True to handle models with conditional computation;
    # cfg.ddp_find_unused_parameters turns it off for models that use every trainable
    # parameter every forward pass, avoiding a redundant autograd-graph traversal per step.
    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.ddp_find_unused_parameters)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        # Drive Accelerate's autocast from policy.dtype (bf16/fp16 activate it; float32/absent -> launcher default).
        policy_dtype = getattr(cfg.trainable_config, "dtype", None)
        mixed_precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}.get(policy_dtype)
        accumulation_plugin = GradientAccumulationPlugin(
            num_steps=cfg.gradient_accumulation_steps,
            sync_with_dataloader=False,
        )
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
            gradient_accumulation_plugin=accumulation_plugin,
        )
    elif getattr(accelerator, "gradient_accumulation_steps", 1) != cfg.gradient_accumulation_steps:
        raise ValueError(
            "The supplied Accelerator gradient accumulation does not match the train config: "
            f"{getattr(accelerator, 'gradient_accumulation_steps', 1)} != "
            f"{cfg.gradient_accumulation_steps}."
        )
    elif cfg.gradient_accumulation_steps > 1 and getattr(
        getattr(accelerator, "gradient_state", None), "sync_with_dataloader", False
    ):
        # The internal branch above sets sync_with_dataloader=False deliberately: the dataloader
        # is prepare()d and then wrapped in cycle(), so accelerate's epoch-boundary sync would
        # force sync_gradients mid-accumulation, step on a partial window, and desynchronize the
        # fixed range(gradient_accumulation_steps) loop from the accumulation boundary. Checking
        # only the step count lets a caller-supplied Accelerator keep the default True and fail
        # partway through the first epoch. At an accumulation window of 1 every step is a sync
        # step, so a vanilla Accelerator() (default sync_with_dataloader=True) stays accepted.
        raise ValueError(
            "The supplied Accelerator has gradient_state.sync_with_dataloader=True, which "
            "desynchronizes gradient accumulation from the training loop's fixed accumulation "
            "window. Construct it with GradientAccumulationPlugin(num_steps="
            f"{cfg.gradient_accumulation_steps}, sync_with_dataloader=False)."
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: the global main process downloads once to the shared
    # dataset root, then a barrier lets every other rank read the already-populated copy.
    # LeRobotDataset skips its snapshot_download when try_load() succeeds, so no rank re-downloads.
    if is_main_process:
        logging.info("Creating dataset")
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    accelerator.wait_for_everyone()

    # Other ranks read from the shared copy populated by the main process.
    if not is_main_process:
        dataset, eval_dataset = make_train_eval_datasets(cfg)

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        require_package("peft", extra="peft")

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if _should_pass_dataset_meta(cfg, active_cfg):
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        # On resume, the checkpoint's saved processor stats are authoritative: they may have
        # been adapted by the policy (e.g. EVO1 pads state/action stats to max_state_dim),
        # and force-feeding raw dataset stats over them crashes normalization (#4006).
        # This mirrors the `dataset_stats` kwarg above, which is also skipped on resume.
        if not cfg.resume:
            preprocessor_overrides["normalizer_processor"]["stats"] = dataset.meta.stats
            postprocessor_overrides["unnormalizer_processor"]["stats"] = dataset.meta.stats
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)
    resume_metadata = load_training_metadata(cfg.checkpoint_path) if cfg.resume else {}

    if cfg.resume:
        saved_accumulation_steps = resume_metadata.get("gradient_accumulation_steps")
        if saved_accumulation_steps not in (None, cfg.gradient_accumulation_steps):
            raise ValueError(
                "Cannot sample-exactly resume with gradient_accumulation_steps="
                f"{cfg.gradient_accumulation_steps}; the checkpoint was written with "
                f"gradient_accumulation_steps={saved_accumulation_steps}."
            )
        # Under FSDP the optimizer state is sharded and must be loaded after `accelerator.prepare()`
        # (see load_fsdp_optimizer_state below), so skip the optimizer here and load it then.
        is_fsdp = accelerator.distributed_type == DistributedType.FSDP
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler, load_optimizer=not is_fsdp
        )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * cfg.gradient_accumulation_steps * num_processes
        logging.info(
            "Effective batch size: "
            f"{cfg.batch_size} x {cfg.gradient_accumulation_steps} accumulation x "
            f"{num_processes} processes = {effective_bs}"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if not cfg.dataset.streaming:
        # All non-streaming (map-style) datasets use EpisodeAwareSampler.
        # The order is a pure function of (seed, epoch), so every rank independently produces the
        # same permutation. accelerate then shards it disjointly across ranks via BatchSamplerShard
        # without needing a `generator` attribute to synchronize an RNG, and resume is sample-exact.
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=getattr(active_cfg, "drop_n_last_frames", 0),
            shuffle=True,
            seed=cfg.seed if cfg.seed is not None else 0,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
        if cfg.resume and step > 0:
            # The resume offset depends on the (num_processes, batch_size) that produced `step`, so
            # use the values recorded in the checkpoint (falling back to the current ones for older
            # ckpts that did not store them).
            saved_num_processes = resume_metadata.get("num_processes")
            saved_batch_size = resume_metadata.get("batch_size")
            ckpt_num_processes = saved_num_processes or accelerator.num_processes
            local_effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
            ckpt_batch_size = saved_batch_size or local_effective_batch_size
            if is_main_process and saved_num_processes not in (None, accelerator.num_processes):
                logging.warning(
                    f"Resuming with num_processes={accelerator.num_processes} but the checkpoint was "
                    f"written with num_processes={saved_num_processes}. The data order resumes at the "
                    "right epoch/offset, but per-rank sample-exactness requires the same world size."
                )
            if is_main_process and saved_batch_size not in (None, local_effective_batch_size):
                logging.warning(
                    f"Resuming with effective local batch size={local_effective_batch_size} but the "
                    f"checkpoint was written with batch_size={saved_batch_size}. The data order resumes "
                    "at the right epoch/offset, but per-rank sample-exactness requires the same effective "
                    "local batch size."
                )
            sampler_state = compute_sampler_state(step, len(sampler), ckpt_batch_size, ckpt_num_processes)
            sampler.load_state_dict(sampler_state)
            if is_main_process:
                logging.info(
                    f"Resuming data order at epoch {sampler_state['epoch']}, "
                    f"sample {sampler_state['start_index']}"
                )
    else:
        shuffle = True
        sampler = None

    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        **_dataloader_worker_kwargs(cfg),
    )

    # Build eval dataloader if a held-out split exists
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for t in unique_tasks:
                frames = (task_arr == t).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)

        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            **_dataloader_worker_kwargs(cfg),
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    if eval_dataloader is not None:
        policy, optimizer, dataloader, lr_scheduler, eval_dataloader = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler, eval_dataloader
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )

    # FSDP optimizer state is sharded across ranks, so it can only be loaded once the optimizer and
    # model are FSDP-wrapped (i.e. after `prepare`). Collective: every rank must participate.
    if cfg.resume and accelerator.distributed_type == DistributedType.FSDP:
        load_fsdp_optimizer_state(policy, optimizer, cfg.checkpoint_path)

    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss DDP
        # is actually optimizing. grad_norm and lr are already identical on every rank (post
        # gradient sync / deterministic scheduler) so reducing them would be a no-op collective.
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        # Derived from the post-reduce max step time; set once per log window on the main rank.
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    local_effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
    effective_batch_size = local_effective_batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        local_effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    def load_microbatch():
        batch = next(dl_iter)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        return preprocessor(batch)

    def update_microbatch(batch, loss_denominators):
        nonlocal train_tracker
        train_tracker, _ = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
            loss_denominators=loss_denominators,
        )

    for _ in range(step, cfg.steps):
        optimizer_step_start = time.perf_counter()
        dataloading_s = _run_optimizer_update_microbatches(
            policy=policy,
            accelerator=accelerator,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            sample_weighter=sample_weighter,
            load_batch=load_microbatch,
            update_batch=update_microbatch,
        )
        if not accelerator.sync_gradients:
            raise RuntimeError("Accelerator did not synchronize at the configured accumulation boundary.")
        train_tracker.dataloading_s = dataloading_s
        train_tracker.update_s = time.perf_counter() - optimizer_step_start - dataloading_s

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = should_save_checkpoint(step, cfg.save_freq, cfg.steps)
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = cfg.eval_steps > 0 and eval_dataloader is not None and step % cfg.eval_steps == 0

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process:
                # Cluster-wide throughput, derived from the already-reduced (max) step time so it
                # reflects the slowest rank — which is what actually gates the next iteration.
                step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                if step_time > 0:
                    train_tracker.samples_per_s = effective_batch_size / step_time
                logging.info(train_tracker)
                if wandb_logger:
                    # Policy sub-losses (latent_loss, action_loss, ...) are aggregated into the
                    # tracker by update_policy, so to_dict() already carries their windowed,
                    # rank-reduced averages — no per-step output_dict passthrough needed.
                    wandb_log_dict = train_tracker.to_dict()
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_eval_step:
            policy.eval()
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), accelerator.autocast():
                for eval_batch in eval_dataloader:
                    for cam_key in dataset.meta.camera_keys:
                        if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                            eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
                    eval_batch = preprocessor(eval_batch)
                    loss, _ = policy.forward(eval_batch)
                    eval_loss_sum += loss.item()
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            eval_loss = torch.tensor(eval_loss, device=device)
            eval_loss = accelerator.reduce(eval_loss, reduction="mean").item()
            policy.train()

            if is_main_process:
                logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
                if wandb_logger:
                    wandb_logger.log_dict({"eval_loss": eval_loss}, step=step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            # Under FSDP, gathering the full model + optimizer state dicts is a cross-rank collective,
            # so all ranks must participate; rank 0 then writes the materialized dicts. For DDP /
            # single-GPU the state dicts are saved the normal way inside save_checkpoint.
            is_fsdp = accelerator.distributed_type == DistributedType.FSDP
            if is_fsdp:
                model_state_dict, optim_state_dict = gather_fsdp_state_dicts(policy, optimizer)
            else:
                model_state_dict, optim_state_dict = None, None
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    num_processes=accelerator.num_processes,
                    batch_size=local_effective_batch_size,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                )
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        cfg.policy.repo_id,
                        private=cfg.policy.private,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with _make_eval_envs(cfg) as eval_env, torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    should_push_to_hub = bool(getattr(active_cfg, "push_to_hub", False))
    # FSDP state gathering is a collective and can materialize the full model on
    # every rank. Only pay that cost when the final package will actually use it.
    model_state_dict = accelerator.get_state_dict(policy) if is_fsdp and should_push_to_hub else None
    if is_main_process:
        logging.info("End of training")

        if should_push_to_hub:
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(
                    cfg,
                    peft_model=unwrapped_model,
                    state_dict=model_state_dict,
                    dataset_meta=dataset.meta,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
            else:
                unwrapped_model.push_model_to_hub(
                    cfg,
                    state_dict=model_state_dict,
                    dataset_meta=dataset.meta,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _remote_target_in_argv() -> bool:
    """True when the CLI requests a remote HF Jobs run (--job.target=<non-local>)."""
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
