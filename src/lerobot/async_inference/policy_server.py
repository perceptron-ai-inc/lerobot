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

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import json
import logging
import math
import pickle  # nosec
import tempfile
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.lerobot_types import PolicyAction
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.peft import load_peft_policy
from lerobot.policies.utils import validate_perceptron_direct_chunk_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.rename_processor import RenameObservationsProcessorStep
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


def _validate_async_fps_contract(policy_config: Any | None, *, client_fps: float, server_fps: float) -> None:
    """Bind the client execution clock, server timing, and checkpoint clock."""

    client_fps = float(client_fps)
    server_fps = float(server_fps)
    if (
        not math.isfinite(client_fps)
        or not math.isfinite(server_fps)
        or client_fps <= 0.0
        or server_fps <= 0.0
    ):
        raise ValueError(
            f"Async fps must be finite and positive; client fps={client_fps}, server fps={server_fps}."
        )
    if abs(client_fps - server_fps) > 1e-9:
        raise ValueError(
            f"Async client fps={client_fps:g} does not match server fps={server_fps:g}. "
            "Use one control rate on both processes."
        )

    target_fps = getattr(policy_config, "target_fps", None) if policy_config is not None else None
    if target_fps is not None and (
        not math.isfinite(float(target_fps))
        or float(target_fps) <= 0.0
        or abs(server_fps - float(target_fps)) > 1e-9
    ):
        raise ValueError(
            f"Async server fps={server_fps:g} does not match checkpoint target_fps={float(target_fps):g}."
        )


def _validate_strict_async_feature_contract(
    policy_config: Any,
    lerobot_features: dict[str, Any],
    effective_rename_map: dict[str, str],
) -> None:
    """Validate names and ordering that tensor shapes alone cannot protect."""

    if not getattr(policy_config, "strict_hardware_feature_contract", False):
        return

    problems: list[str] = []
    expected_state = list(getattr(policy_config, "state_feature_names", None) or [])
    actual_state = list((lerobot_features.get(OBS_STATE) or {}).get("names", []))
    if actual_state != expected_state:
        problems.append(f"state order {actual_state} != {expected_state}")

    expected_action = list(getattr(policy_config, "action_feature_names", None) or [])
    actual_action = list((lerobot_features.get(ACTION) or {}).get("names", []))
    if actual_action != expected_action:
        problems.append(f"action order {actual_action} != {expected_action}")

    raw_images = [key for key in lerobot_features if key.startswith(OBS_IMAGES)]
    mapped_images = [effective_rename_map.get(key, key) for key in raw_images]
    expected_images = [f"{OBS_IMAGES}.{camera}" for camera in getattr(policy_config, "camera_order", ())]
    if len(mapped_images) != len(set(mapped_images)) or set(mapped_images) != set(expected_images):
        problems.append(f"camera mapping {mapped_images} != {expected_images}")

    if problems:
        raise ValueError("perceptron_isaac checkpoint contract validation failed: " + "; ".join(problems))


def _validate_async_chunk_contract(policy_config: Any, *, actions_per_chunk: int) -> None:
    """Reject direct-chunk modes that bypass Perceptron's select_action semantics."""
    validate_perceptron_direct_chunk_inference(policy_config, actions_per_chunk, mode="async")


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.rename_map: dict[str, str] = {}
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.client_fps: float | None = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

        # Cache the last loaded policy signature so identical SendPolicyInstructions
        # calls can skip the expensive weight load. `rename_map` is excluded since
        # it only affects the preprocessor and is patched in via `_inject_rename_map`.
        self._loaded_policy_key: tuple | None = None
        # The rename map the loaded pipeline shipped with (before any client-supplied
        # map was injected). Cache-hit sessions restore it when a client sends the
        # default empty map, mirroring what a full reload from the checkpoint would do.
        self._packaged_rename_map: dict[str, str] = {}

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def _validate_session_contracts(self) -> None:
        """Validate one loaded-policy/client pairing before accepting observations."""

        _validate_async_fps_contract(
            self.policy.config,
            client_fps=self.client_fps,
            server_fps=self.config.fps,
        )
        _validate_async_chunk_contract(
            self.policy.config,
            actions_per_chunk=self.actions_per_chunk,
        )
        _validate_strict_async_feature_contract(
            self.policy.config,
            self.lerobot_features,
            self.rename_map,
        )
        validate_features = getattr(self.policy, "validate_async_features", None)
        if validate_features is not None:
            validate_features(self.lerobot_features)

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()

        # Per-session policy reset (policy-independent hook): clears episode
        # state such as perceptron_isaac's frame clock, online history queues,
        # and timestamp-monotonicity guard while the cached weights stay loaded.
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.client_fps = float(policy_specs.fps)

        _validate_async_fps_contract(
            None,
            client_fps=self.client_fps,
            server_fps=self.config.fps,
        )

        policy_class = get_policy_class(self.policy_type)

        # CLI-style overrides shipped by the client, applied to the policy config
        # below so deployment-only fields can be tuned without editing the checkpoint.
        cli_overrides = list(getattr(policy_specs, "policy_config_overrides", []) or [])
        if cli_overrides:
            self.logger.info(f"Applying {len(cli_overrides)} policy config override(s): {cli_overrides}")

        rename_map = dict(policy_specs.rename_map or {})

        # HF-original mode derives the policy config's input/output features from the
        # robot handshake itself, so a different robot must trigger a rebuild there.
        # Saved-checkpoint mode keys features off the checkpoint instead (the
        # `validate_async_features` hook checks the handshake on every call).
        features_key = (
            None
            if policy_specs.pretrained_name_or_path
            else json.dumps(self.lerobot_features, sort_keys=True, default=str)
        )

        # Fast path: identical signature to the last load -> reuse the in-memory
        # policy and processors, just refresh the rename map and per-request state.
        load_key = (
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            tuple(cli_overrides),
            policy_specs.device,
            self.client_fps,
            features_key,
        )
        if self._loaded_policy_key == load_key and self.policy is not None:
            self.logger.info(
                "Policy already loaded with identical signature "
                f"(policy_type={policy_specs.policy_type}, "
                f"checkpoint={policy_specs.pretrained_name_or_path or '<hf-original>'}, "
                f"overrides={len(cli_overrides)}, device={policy_specs.device}) -- skipping reload."
            )
            # `lerobot_features` is a dict and therefore not part of `load_key`, so a
            # reconnecting client can present a different robot contract while hitting
            # this cache. Run the same checkpoint-contract validation as the slow path;
            # readiness must fail loudly here, not mid-episode with a live arm.
            # Mirror the slow path's semantics exactly: a non-empty client map is
            # injected; the default empty map restores the map the checkpoint
            # pipeline shipped with (a full reload would rebuild it from disk).
            # Neither erases a packaged map with {} nor leaks a previous client's
            # injected map into this session.
            if self.preprocessor is not None:
                effective_rename_map = rename_map if rename_map else self._packaged_rename_map
                _inject_rename_map(self.preprocessor, effective_rename_map)
                self.rename_map = dict(effective_rename_map)
            self._validate_session_contracts()
            return services_pb2.Empty()

        start = time.perf_counter()

        if policy_specs.pretrained_name_or_path:
            # LeRobot-saved checkpoint: load the saved config (applying overrides),
            # then load weights via the standard `from_pretrained` flow.
            policy_config = PreTrainedConfig.from_pretrained(
                policy_specs.pretrained_name_or_path,
                cli_overrides=cli_overrides,
            )
            policy_config.device = self.device
            _validate_async_fps_contract(
                policy_config,
                client_fps=self.client_fps,
                server_fps=self.config.fps,
            )
            _validate_async_chunk_contract(
                policy_config,
                actions_per_chunk=self.actions_per_chunk,
            )
            self.policy = _load_saved_policy(
                policy_class,
                policy_config,
                policy_specs.pretrained_name_or_path,
            )
            self.policy.to(self.device)

            # Load preprocessor and postprocessor, overriding device to match requested
            # device. The pipeline is loaded WITHOUT a rename override so the map it
            # ships with can be recorded first: cache-hit sessions restore that packaged
            # map when a client sends the default empty map. A non-empty client map is
            # then injected on top (never for {}, which would wholesale-replace rename
            # maps packaged with the checkpoint pipeline, e.g. perceptron_isaac's saved
            # preprocessor).
            device_override = {"device": self.device}
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=policy_specs.pretrained_name_or_path,
                preprocessor_overrides={"device_processor": device_override},
                postprocessor_overrides={"device_processor": device_override},
            )
            self._packaged_rename_map = _get_rename_map(self.preprocessor)
            effective_rename_map = rename_map if rename_map else self._packaged_rename_map
            _inject_rename_map(self.preprocessor, effective_rename_map)
            self.rename_map = dict(effective_rename_map)
        else:
            # HF-original checkpoint: no saved config.json. Build the config from
            # overrides only; the policy's __init__ fetches its own weights using
            # the checkpoint location passed via `policy_config_overrides`.
            self.logger.info(
                f"No `pretrained_name_or_path` provided; building {self.policy_type} config "
                f"from policy_config_overrides only (HF-original checkpoint mode)."
            )
            policy_config = self._build_policy_config_from_overrides(self.policy_type, cli_overrides)
            policy_config.device = self.device
            _validate_async_fps_contract(
                policy_config,
                client_fps=self.client_fps,
                server_fps=self.config.fps,
            )
            _validate_async_chunk_contract(
                policy_config,
                actions_per_chunk=self.actions_per_chunk,
            )

            # Populate the config's features from the robot's observation/action
            # features so the policy sees the right input/output shapes.
            policy_features = dataset_to_policy_features(self.lerobot_features)
            output_features = {
                key: ft for key, ft in policy_features.items() if ft.type is FeatureType.ACTION
            }
            input_features = {key: ft for key, ft in policy_features.items() if key not in output_features}
            policy_config.output_features = output_features
            if not policy_config.input_features:
                policy_config.input_features = input_features

            self.policy = policy_class(policy_config)
            self.policy.to(self.device)

            # No saved pipeline -> build a fresh one from the config.
            self.preprocessor, self.postprocessor = make_pre_post_processors(self.policy.config)
            self._packaged_rename_map = _get_rename_map(self.preprocessor)
            effective_rename_map = rename_map if rename_map else self._packaged_rename_map
            _inject_rename_map(self.preprocessor, effective_rename_map)
            self.rename_map = dict(effective_rename_map)

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        # Policy-specific checkpoint-contract validation (policy-independent
        # hook): fail the load loudly now, not at the first inference.
        self._validate_session_contracts()

        # Record the load signature so identical calls can reuse this policy.
        self._loaded_policy_key = load_key

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            image_key_map=self.rename_map,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t

        # Policy-independent adaptation hook: lets a policy derive per-request
        # inputs from the scheduling metadata (e.g. perceptron_isaac anchors its
        # prompt clock to timestep/fps under chunk-rate inference). Applied to
        # the preprocessed batch — pipeline steps may rebuild the dict, so keys
        # injected earlier would not survive to the policy.
        adapt_async_observation = getattr(self.policy, "adapt_async_observation", None)
        if adapt_async_observation is not None:
            observation = adapt_async_observation(
                observation, timestep=observation_t.get_timestep(), fps=self.config.fps
            )
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        start_postprocess = time.perf_counter()
        action_tensor = self.postprocessor(action_tensor).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    @staticmethod
    def _build_policy_config_from_overrides(policy_type: str, cli_overrides: list[str]) -> PreTrainedConfig:
        """Build a fresh ``PreTrainedConfig`` from draccus CLI overrides.

        Used in HF-original mode, where there is no saved ``config.json``: draccus
        parses an empty JSON file and the overrides populate the dataclass defaults.
        """
        config_class = PreTrainedConfig.get_choice_class(policy_type)
        # draccus only needs the file while parse runs; a delete=False tempfile here
        # leaked one /tmp/tmp*.json per HF-original cache miss for the process lifetime.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_file = Path(temp_dir) / "config.json"
            temp_config_file.write_text("{}")
            with draccus.config_type("json"):
                return draccus.parse(config_class, str(temp_config_file), args=cli_overrides)

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


def _get_rename_map(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> dict[str, str]:
    """Read the current map from the first ``RenameObservationsProcessorStep``."""
    for step in preprocessor.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            return dict(step.rename_map)
    return {}


def _inject_rename_map(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    rename_map: dict[str, str],
) -> None:
    """Patch the rename map onto the first ``RenameObservationsProcessorStep``.

    Used for fresh pipelines (HF-original mode), which can't take the rename map
    through ``PolicyProcessorPipeline.from_pretrained`` overrides.
    """
    for step in preprocessor.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            step.rename_map = dict(rename_map)
            return


def _load_saved_policy(policy_class, policy_config: PreTrainedConfig, checkpoint: str):
    """Load a full checkpoint or a saved PEFT adapter for async serving.

    The async server used to call the policy class directly for every saved package.
    That works for full checkpoints but skips ``adapter_model.safetensors`` entirely.
    Mirror the policy factory PEFT branch and resolve a relative local base path
    against the adapter package so sibling checkpoint bundles remain relocatable.
    """
    if not policy_config.use_peft:
        return policy_class.from_pretrained(checkpoint, config=policy_config)
    return load_peft_policy(policy_class, policy_config, checkpoint)


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
