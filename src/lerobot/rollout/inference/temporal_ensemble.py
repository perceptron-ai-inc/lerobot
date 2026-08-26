# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Latency-aware asynchronous chunk inference with temporal ensembling.

The producer timestamps the observation used for each prediction.  The control
thread does not blindly begin at action zero after inference finishes; it picks
the action at ``floor((now - observation_time) * action_fps)`` from every
still-active chunk and combines those actions with an exponential age weight.
This is the execution model used by the physical SO-101 runtime linked from the
MolmoAct2 documentation, adapted to LeRobot's inference-engine interface.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections import deque
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from threading import Event, Lock, Thread

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine

logger = logging.getLogger(__name__)

_IDLE_SLEEP_S = 0.01
_ERROR_RETRY_DELAY_S = 0.1
_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class TimestampedActionChunk:
    """A postprocessed action plan anchored to its source observation time."""

    actions: torch.Tensor
    observation_time: float
    chunk_id: int


def clip_action_direction_preserving(
    action: torch.Tensor, current_state: torch.Tensor, max_step: float
) -> torch.Tensor:
    """Scale a target delta so its largest component is at most ``max_step``.

    Scaling the complete vector preserves the predicted joint-space direction,
    unlike independent per-joint clipping.
    """

    if max_step <= 0:
        raise ValueError(f"max_step must be positive, got {max_step}.")
    delta = action - current_state
    biggest = float(torch.max(torch.abs(delta)).item())
    if biggest == 0.0 or biggest <= max_step:
        return action
    return current_state + delta * (max_step / biggest)


def temporal_ensemble_action(
    entries: list[TimestampedActionChunk],
    *,
    now: float,
    action_fps: float,
    ensemble_m: float,
    actions_per_chunk: int | None = None,
) -> tuple[torch.Tensor | None, list[tuple[int, int, float]]]:
    """Select wall-time-aligned actions from active chunks and ensemble them.

    Returns the action and diagnostics ``(chunk_id, step_index, age_seconds)``
    for each contributing chunk.
    """

    if action_fps <= 0:
        raise ValueError(f"action_fps must be positive, got {action_fps}.")
    if ensemble_m < 0:
        raise ValueError(f"ensemble_m must be nonnegative, got {ensemble_m}.")
    if actions_per_chunk is not None and actions_per_chunk <= 0:
        raise ValueError(f"actions_per_chunk must be positive when set, got {actions_per_chunk}.")

    active_actions: list[torch.Tensor] = []
    active_ages: list[float] = []
    diagnostics: list[tuple[int, int, float]] = []
    for entry in entries:
        age = now - entry.observation_time
        step = math.floor(age * action_fps)
        horizon = int(entry.actions.shape[0])
        if actions_per_chunk is not None:
            horizon = min(horizon, actions_per_chunk)
        if 0 <= step < horizon:
            active_actions.append(entry.actions[step])
            active_ages.append(age)
            diagnostics.append((entry.chunk_id, step, age))

    if not active_actions:
        return None, []

    actions = torch.stack(active_actions, dim=0).float()
    ages = torch.tensor(active_ages, dtype=torch.float32)
    weights = torch.exp(-ensemble_m * ages)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        # exp(-m*age) underflows float32 to exactly 0 once m*age > ~103.9, so a large
        # ensemble_m (or very stale chunks) zeroes every live weight and the division
        # below would emit NaN actions straight to the robot. Fall back to the freshest
        # live chunk, which is what the ensemble converges to as m grows anyway.
        freshest = int(torch.argmin(ages))
        return actions[freshest], [diagnostics[freshest]]
    weights /= total
    target = (weights[:, None] * actions).sum(dim=0)
    if not torch.isfinite(target).all():
        raise ValueError(
            "Temporal ensemble produced a non-finite action target; refusing to send it "
            f"to the robot (ages={active_ages}, ensemble_m={ensemble_m})."
        )
    return target, diagnostics


class TemporalEnsembleInferenceEngine(InferenceEngine):
    """Continuously predict chunks and consume them using elapsed wall time."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        robot_type: str,
        hw_features: dict,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        device: str | None,
        ensemble_m: float = 0.5,
        max_step: float | None = 15.0,
        smooth_alpha: float = 1.0,
        actions_per_chunk: int | None = None,
        ring_capacity: int = 6,
        warmup_predictions: int = 0,
        max_consecutive_errors: int = 3,
        shutdown_event: Event | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if ensemble_m < 0:
            raise ValueError("ensemble_m must be nonnegative.")
        if max_step is not None and max_step <= 0:
            raise ValueError("max_step must be positive when set.")
        if not 0.05 <= smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be between 0.05 and 1.0.")
        if actions_per_chunk is not None and actions_per_chunk <= 0:
            raise ValueError("actions_per_chunk must be positive when set.")
        if ring_capacity <= 0:
            raise ValueError("ring_capacity must be positive.")
        if warmup_predictions < 0:
            raise ValueError("warmup_predictions must be nonnegative.")
        if max_consecutive_errors <= 0:
            raise ValueError("max_consecutive_errors must be positive.")

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot_type = robot_type
        self._hw_features = hw_features
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._fps = float(fps)
        self._device = torch.device(device or "cpu")
        self._ensemble_m = float(ensemble_m)
        self._max_step = None if max_step is None else float(max_step)
        self._smooth_alpha = float(smooth_alpha)
        self._actions_per_chunk = actions_per_chunk
        self._ring_capacity = int(ring_capacity)
        self._warmup_predictions = int(warmup_predictions)
        self._max_consecutive_errors = int(max_consecutive_errors)
        self._global_shutdown_event = shutdown_event

        self._obs_lock = Lock()
        # (observation, capture wall time, episode-relative control timestep).
        # The timestep advances on notify_observation, never on model calls, so a
        # slow or fast producer cannot distort a policy-owned episode clock.
        self._latest_observation: tuple[dict, float, int] | None = None
        self._next_observation_timestep = 0
        self._last_predicted_timestep: int | None = None
        # Serializes reset() against one full producer iteration
        # (snapshot -> predict -> publish), so an episode reset can neither
        # interrupt the policy/processors mid-predict nor race a stale chunk
        # into the just-cleared ring. Ordering: _predict_lock > _ring_lock/_obs_lock.
        self._predict_lock = Lock()
        self._ring_lock = Lock()
        self._ring: deque[TimestampedActionChunk] = deque(maxlen=self._ring_capacity)
        self._next_chunk_id = 1
        self._seen_chunk_ids: set[int] = set()
        self._last_action: torch.Tensor | None = None
        self._holding = False

        self._active = Event()
        self._shutdown = Event()
        self._ready = Event()
        self._failed = Event()
        self._thread: Thread | None = None
        self._error: BaseException | None = None
        self._prediction_count = 0
        if self._warmup_predictions == 0:
            self._ready.set()

        logger.info(
            "TemporalEnsembleInferenceEngine initialized "
            "(fps=%.1f, ensemble_m=%.2f, max_step=%s, warmup_predictions=%d)",
            self._fps,
            self._ensemble_m,
            self._max_step,
            self._warmup_predictions,
        )

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = Thread(target=self._producer_loop, daemon=True, name="TemporalChunkProducer")
        self._thread.start()
        logger.info("Temporal ensemble inference producer started")

    def stop(self) -> None:
        self._shutdown.set()
        self._active.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("Temporal chunk producer did not stop within %.1fs", _JOIN_TIMEOUT_S)
        self._thread = None
        logger.info("Temporal ensemble inference producer stopped")

    def reset(self) -> None:
        """Clear episode-scoped model, processor, chunk, and smoothing state."""

        with self._predict_lock:
            self._reset_episode_state(clear_observation=True)
            self._error = None
            self._failed.clear()
            # Warmup is an engine-lifetime cost (validation/compile flush), not a
            # per-episode one: _prediction_count stays cumulative so an episode
            # reset does not stall the control loop on a fresh warmup round.
            if self._prediction_count >= self._warmup_predictions:
                self._ready.set()
            else:
                self._ready.clear()

    def _reset_episode_state(self, *, clear_observation: bool) -> None:
        """Reset stateful inference components while optionally retaining the latest frame."""

        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        with self._ring_lock:
            self._ring.clear()
            self._next_chunk_id = 1
        with self._obs_lock:
            if clear_observation:
                self._latest_observation = None
                self._next_observation_timestep = 0
        self._last_predicted_timestep = None
        self._seen_chunk_ids.clear()
        self._last_action = None
        self._holding = False

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        self._active.set()

    def notify_observation(self, obs: dict) -> None:
        # The timestamp belongs to the observation, not to the later start or end
        # of inference. A shallow copy is enough because rollout builds a fresh
        # observation dictionary for every control tick.
        with self._obs_lock:
            timestep = self._next_observation_timestep
            self._next_observation_timestep += 1
            self._latest_observation = (copy(obs), time.monotonic(), timestep)

    def _raise_if_failed(self) -> None:
        if self._failed.is_set():
            detail = f": {self._error}" if self._error is not None else ""
            raise RuntimeError(f"Temporal ensemble inference producer failed{detail}") from self._error

    def _current_state(self) -> torch.Tensor:
        with self._obs_lock:
            snapshot = self._latest_observation
        if snapshot is None:
            raise RuntimeError("No observation is available for action limiting.")
        obs, _, _ = snapshot
        missing = [key for key in self._ordered_action_keys if key not in obs]
        if missing:
            raise KeyError(
                "Cannot apply the direction-preserving action cap because the latest observation "
                f"does not contain action-state keys: {missing}."
            )
        return torch.tensor(
            [float(torch.as_tensor(obs[key]).reshape(-1)[0].item()) for key in self._ordered_action_keys],
            dtype=torch.float32,
        )

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        del obs_frame
        self._raise_if_failed()
        with self._ring_lock:
            entries = list(self._ring)
        target, diagnostics = temporal_ensemble_action(
            entries,
            now=time.monotonic(),
            action_fps=self._fps,
            ensemble_m=self._ensemble_m,
            actions_per_chunk=self._actions_per_chunk,
        )
        if target is None:
            if not self._holding:
                logger.info("Temporal ensemble has no active chunks; holding the last commanded target")
                self._holding = True
            return None
        self._holding = False

        for chunk_id, step, age in diagnostics:
            if chunk_id not in self._seen_chunk_ids:
                self._seen_chunk_ids.add(chunk_id)
                logger.info(
                    "Temporal chunk %d active at step=%d age=%.0fms",
                    chunk_id,
                    step,
                    age * 1000.0,
                )

        if self._max_step is not None:
            target = clip_action_direction_preserving(target, self._current_state(), self._max_step)
        if self._last_action is not None and self._smooth_alpha < 1.0:
            target = self._smooth_alpha * target + (1.0 - self._smooth_alpha) * self._last_action
        self._last_action = target
        return target.cpu()

    def _snapshot_observation(self) -> tuple[dict, float, int] | None:
        with self._obs_lock:
            return self._latest_observation

    def _ordered_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions.squeeze(0)
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk [T, A], got shape={tuple(actions.shape)}.")
        ordered_rows = []
        for action in actions:
            action_dict = make_robot_action(action.detach().cpu(), self._dataset_features)
            ordered_rows.append([action_dict[key] for key in self._ordered_action_keys])
        chunk = torch.tensor(ordered_rows, dtype=torch.float32)
        if chunk.shape[1] != len(self._ordered_action_keys):
            raise ValueError(
                f"Action chunk width {chunk.shape[1]} != expected {len(self._ordered_action_keys)}."
            )
        if not torch.isfinite(chunk).all():
            raise ValueError("Action chunk contains non-finite values.")
        return chunk

    def _predict(self, obs: dict, *, timestep: int) -> torch.Tensor:
        obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
        obs_batch = prepare_observation_for_inference(obs_batch, self._device, self._task, self._robot_type)
        obs_batch["task"] = [self._task]
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            preprocessed = self._preprocessor(obs_batch)
            adapt_observation = getattr(self._policy, "adapt_async_observation", None)
            if adapt_observation is not None:
                preprocessed = adapt_observation(preprocessed, timestep=timestep, fps=self._fps)
            actions = self._policy.predict_action_chunk(preprocessed)
            actions = self._postprocessor(actions)
        return self._ordered_chunk(actions)

    def _publish(self, actions: torch.Tensor, observation_time: float) -> int:
        with self._ring_lock:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            self._ring.append(TimestampedActionChunk(actions, observation_time, chunk_id))
        return chunk_id

    def _producer_iteration(self, consecutive_errors: int) -> tuple[int, float | None]:
        """Run one snapshot -> predict -> publish cycle under ``_predict_lock``.

        Returns the updated consecutive-error count and an optional sleep the
        caller should perform after releasing the lock.
        """

        snapshot = self._snapshot_observation()
        if snapshot is None:
            return consecutive_errors, _IDLE_SLEEP_S
        obs, observation_time, timestep = snapshot
        warming_up = self._prediction_count < self._warmup_predictions
        if not warming_up and timestep == self._last_predicted_timestep:
            return consecutive_errors, _IDLE_SLEEP_S
        try:
            started = time.perf_counter()
            actions = self._predict(obs, timestep=timestep)
            latency = time.perf_counter() - started
            self._prediction_count += 1
            self._last_predicted_timestep = timestep

            if self._prediction_count <= self._warmup_predictions:
                logger.info(
                    "Temporal chunk warmup %d/%d completed in %.0fms; full chunk validated and discarded",
                    self._prediction_count,
                    self._warmup_predictions,
                    latency * 1000.0,
                )
                if self._prediction_count == self._warmup_predictions:
                    # Prediction-only warmup may mutate policy queues, clocks,
                    # caches, and processors. Preserve the captured frame so it can
                    # be predicted once as the first live plan, but clear all inference
                    # state before advertising readiness.
                    self._reset_episode_state(clear_observation=False)
                    self._ready.set()
                return 0, None

            chunk_id = self._publish(actions, observation_time)
            age_ms = (time.monotonic() - observation_time) * 1000.0
            logger.info(
                "Temporal chunk predicted id=%d latency=%.0fms age_at_publish=%.0fms horizon=%d",
                chunk_id,
                latency * 1000.0,
                age_ms,
                actions.shape[0],
            )
            return 0, None
        except Exception as exc:
            consecutive_errors += 1
            logger.error(
                "Temporal chunk inference error (%d/%d): %s",
                consecutive_errors,
                self._max_consecutive_errors,
                exc,
            )
            logger.debug(traceback.format_exc())
            if consecutive_errors >= self._max_consecutive_errors:
                raise
            return consecutive_errors, _ERROR_RETRY_DELAY_S

    def _producer_loop(self) -> None:
        consecutive_errors = 0
        try:
            while not self._shutdown.is_set():
                if not self._active.wait(timeout=_IDLE_SLEEP_S):
                    continue
                if self._shutdown.is_set():
                    return
                with self._predict_lock:
                    consecutive_errors, sleep_s = self._producer_iteration(consecutive_errors)
                if sleep_s is not None:
                    time.sleep(sleep_s)
        except BaseException as exc:
            self._error = exc
            self._failed.set()
            self._ready.set()
            logger.error("Fatal temporal chunk producer error: %s", exc)
            logger.debug(traceback.format_exc())
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
