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
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging

# --- optional live observation view (ported from suveene-lerobot-client, observability only) -----
# Enable with LEROBOT_VIEW=1: streams the camera frames captured for the policy
# (exactly what is sent to the server) to a local rerun viewer. Off by default,
# so default behavior matches the upstream branch verbatim. Logging failures are
# swallowed and never disturb the control loop.
import os as _os  # noqa: E402
import pickle  # nosec
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    bi_yam_follower,
    bi_yam_follower_plain,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.rollout.inference.temporal_ensemble import (
    TimestampedActionChunk,
    clip_action_direction_preserving,
    temporal_ensemble_action,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)

_rr = None
if _os.environ.get("LEROBOT_VIEW", "0") not in ("0", "", "false", "False"):
    try:
        import rerun as _rr_mod

        _rr_mod.init("molmoact2_robot_client", spawn=True)
        _rr = _rr_mod
    except Exception:
        _rr = None


def _log_view(raw_observation: dict) -> None:
    """Log the outgoing camera frames (the exact arrays sent to the policy server)."""
    if _rr is None:
        return
    try:
        for key, value in raw_observation.items():
            if hasattr(value, "ndim") and getattr(value, "ndim", 0) == 3:
                _rr.log(f"camera/{key}", _rr.Image(value).compress(jpeg_quality=80))
    except Exception:  # nosec B110 - viewer logging must never disturb the control loop
        pass


# -----------------------------------------------------------------------------


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
            fps=float(config.fps),
            rename_map=dict(config.rename_map),
            policy_config_overrides=list(config.policy_config_overrides),
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # Track if robot was already disconnected (to prevent double disconnect)
        self._robot_disconnected = False

        # Timestamped SO control state. The ring is populated by the receiver
        # thread and sampled by the 30 Hz control thread.
        self._temporal_ring_lock = threading.Lock()
        self._temporal_ring: deque[TimestampedActionChunk] = deque(maxlen=config.temporal_ring_capacity)
        self._temporal_next_chunk_id = 1
        self._temporal_prediction_count = 0
        self._latest_raw_observation_lock = threading.Lock()
        self._latest_raw_observation: RawObservation | None = None
        self._control_timestep = 0

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client.

        Clear pending motion, optionally execute an exact-key robot-specific home
        target, then disconnect the robot and policy-server channel.
        """
        self.shutdown_event.set()

        # Clear action queue first
        self._clear_action_queue()
        self.logger.info("Action queue cleared")

        # Disconnect robot and policy server (if not already done in control_loop)
        if not self._robot_disconnected:
            home = self.config.shutdown_home_position
            if self.config.dispatch_actions and home is not None:
                expected = set(self.robot.action_features)
                supplied = set(home)
                if supplied != expected:
                    self.logger.error(
                        "Skipping shutdown home move: target keys do not exactly match robot action keys "
                        "(missing=%s, unexpected=%s)",
                        sorted(expected - supplied),
                        sorted(supplied - expected),
                    )
                else:
                    self.logger.info("Moving robot to configured shutdown home position...")
                    try:
                        self._slow_move_to_position(home, num_steps=100, step_sleep=0.1)
                        self.logger.info("Robot at configured shutdown home position")
                    except Exception as e:
                        self.logger.warning(f"Error moving to shutdown home position: {e}")
            elif not self.config.dispatch_actions:
                self.logger.info("dispatch_actions=false: skipping shutdown home move")
            else:
                self.logger.info("No shutdown home position configured; disconnecting immediately")

            self.robot.disconnect()
            self.logger.info("Robot disconnected")

            try:
                self.channel.close()
                self.logger.info("Disconnected from policy server")
            except Exception as e:
                self.logger.warning(f"Error closing policy server connection: {e}")

            self.logger.info("=== Robot and policy server disconnected ===")
            self.logger.info("You can now start a new inference session in another terminal")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _add_temporal_chunk(self, timed_actions: list[TimedAction]) -> None:
        if not timed_actions:
            return
        actions = torch.stack([item.get_action().detach().cpu() for item in timed_actions])
        if actions.ndim != 2 or not torch.isfinite(actions).all():
            raise ValueError(f"Invalid temporal action chunk shape/content: {tuple(actions.shape)}")
        self._temporal_prediction_count += 1
        if self._temporal_prediction_count <= self.config.temporal_warmup_chunks:
            self.logger.info(
                "Temporal warmup chunk %d/%d validated and discarded",
                self._temporal_prediction_count,
                self.config.temporal_warmup_chunks,
            )
            return
        with self._temporal_ring_lock:
            chunk_id = self._temporal_next_chunk_id
            self._temporal_next_chunk_id += 1
            self._temporal_ring.append(
                TimestampedActionChunk(
                    actions=actions,
                    observation_time=timed_actions[0].get_timestamp(),
                    chunk_id=chunk_id,
                )
            )
        self.logger.info(
            "Timestamped temporal chunk %d retained (age_at_receive=%.0fms, horizon=%d)",
            chunk_id,
            (time.time() - timed_actions[0].get_timestamp()) * 1000.0,
            actions.shape[0],
        )

    def _temporal_current_state(self) -> torch.Tensor:
        with self._latest_raw_observation_lock:
            observation = self._latest_raw_observation
        if observation is None:
            raise RuntimeError("No current observation is available for temporal action limiting")
        keys = list(self.robot.action_features)
        missing = [key for key in keys if key not in observation]
        if missing:
            raise KeyError(f"Current observation is missing action-state keys: {missing}")
        return torch.tensor([float(observation[key]) for key in keys], dtype=torch.float32)

    def _temporal_action(self) -> torch.Tensor | None:
        now = time.time()
        with self._temporal_ring_lock:
            entries = list(self._temporal_ring)
            self._temporal_ring = deque(
                (
                    entry
                    for entry in entries
                    if (now - entry.observation_time) * self.config.fps
                    < min(entry.actions.shape[0], self.config.actions_per_chunk)
                ),
                maxlen=self.config.temporal_ring_capacity,
            )
            entries = list(self._temporal_ring)
        target, diagnostics = temporal_ensemble_action(
            entries,
            now=now,
            action_fps=self.config.fps,
            ensemble_m=self.config.temporal_ensemble_m,
            actions_per_chunk=self.config.actions_per_chunk,
        )
        if target is None:
            return None
        if self.config.temporal_max_step is not None:
            target = clip_action_direction_preserving(
                target, self._temporal_current_state(), self.config.temporal_max_step
            )
        self.logger.debug("Temporal action contributors: %s", diagnostics)
        return target

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                if self.config.control_mode == "temporal_ensemble":
                    self._add_temporal_chunk(timed_actions)
                    continue

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are queued actions or retained timestamped plans."""
        if self.config.control_mode == "temporal_ensemble":
            with self._temporal_ring_lock:
                return bool(self._temporal_ring)
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        """Select and execute an action from the configured control mode."""
        if self.config.control_mode == "temporal_ensemble":
            action_tensor = self._temporal_action()
            if action_tensor is None:
                return None
            action_dict = self._action_tensor_to_action_dict(action_tensor)
            performed = self.robot.send_action(action_dict) if self.config.dispatch_actions else action_dict
            with self.latest_action_lock:
                self.latest_action += 1
            return performed

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        action_dict = self._action_tensor_to_action_dict(timed_action.get_action())
        if self.config.dispatch_actions:
            _performed_action = self.robot.send_action(action_dict)
        else:
            # Prediction-only preflight: consume the queue and keep the timestep
            # bookkeeping identical to a live run, without touching the hardware.
            _performed_action = action_dict
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _clear_action_queue(self) -> None:
        """Clear all pending actions from the queue."""
        with self.action_queue_lock:
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except Exception:
                    break
        with self._temporal_ring_lock:
            self._temporal_ring.clear()
        self.logger.debug("Action queue and temporal plan ring cleared")

    def _slow_move_to_position(
        self, target_position: dict[str, float], num_steps: int = 100, step_sleep: float = 0.03
    ) -> None:
        """Slowly move robot from current position to target position.

        Matches i2rt's slow_move logic:
        - Discrete steps (default 100)
        - Fixed sleep between steps (default 0.03s)
        - Linear interpolation: target * (i/num_steps) + start * (1 - i/num_steps)

        Args:
            target_position: Target joint positions
            num_steps: Number of interpolation steps (default 100, like i2rt)
            step_sleep: Sleep duration between steps in seconds (default 0.03s, like i2rt)
        """
        # Get current position as starting point
        current_obs = self.robot.get_observation()
        start_positions = {}
        for key in self.robot.action_features:
            if key in current_obs:
                start_positions[key] = current_obs[key]

        # Interpolate over num_steps (like i2rt's slow_move)
        for i in range(num_steps):
            blend_factor = i / num_steps  # 0 to ~1

            blended_action = {}
            for key in self.robot.action_features:
                if key in start_positions and key in target_position:
                    start_val = start_positions[key]
                    target_val = target_position[key]
                    # Same formula as i2rt: target * (i/100) + start * (1 - i/100)
                    blended_action[key] = target_val * blend_factor + start_val * (1 - blend_factor)
                elif key in target_position:
                    blended_action[key] = target_position[key]
                elif key in start_positions:
                    blended_action[key] = start_positions[key]

            self.robot.send_action(blended_action)
            time.sleep(step_sleep)

        # Final step: send exact target position
        self.robot.send_action(target_position)

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation."""
        if self.config.control_mode == "temporal_ensemble":
            return True
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            observation_time = time.time()
            raw_observation["task"] = task
            _log_view(raw_observation)  # no-op unless LEROBOT_VIEW=1; dashboard bridge wraps this
            with self._latest_raw_observation_lock:
                self._latest_raw_observation = raw_observation.copy()

            with self.latest_action_lock:
                latest_action = self.latest_action
            if self.config.control_mode == "temporal_ensemble":
                timestep = self._control_timestep
                self._control_timestep += 1
            else:
                timestep = max(latest_action, 0)

            observation = TimedObservation(
                timestamp=observation_time,
                observation=raw_observation,
                timestep=timestep,
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.config.control_mode == "temporal_ensemble" or (
                    self.must_go.is_set() and self.action_queue.empty()
                )
                current_queue_size = self.action_queue.qsize()

            sent = self.send_observation(observation)
            if self.config.control_mode == "temporal_ensemble" and not sent:
                raise RuntimeError("Failed to send timestamped observation to policy server")

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")
            if self.config.control_mode == "temporal_ensemble":
                raise
            return None

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        # A positive `duration` bounds the session; the loop then exits normally and the
        # caller's shutdown path (stop, optional home move, disconnect) runs, exactly as
        # it does for Ctrl-C.
        deadline = time.perf_counter() + self.config.duration if self.config.duration > 0 else None

        while self.running and (deadline is None or time.perf_counter() < deadline):
            control_loop_start = time.perf_counter()
            if self.config.control_mode == "temporal_ensemble":
                # Capture first so the observation timestamp anchors the next plan and the
                # outer motion cap uses the freshest measured joint state.
                _captured_observation = self.control_loop_observation(task, verbose)
                if self.actions_available():
                    _performed_action = self.control_loop_action(verbose)
            else:
                """Control loop: (1) Performing actions, when available"""
                if self.actions_available():
                    _performed_action = self.control_loop_action(verbose)

                """Control loop: (2) Streaming observations to the remote policy server"""
                if self._ready_to_send_observation():
                    _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        except KeyboardInterrupt:
            client.logger.warning("KeyboardInterrupt received - stopping client")

        except Exception as e:
            client.logger.error(f"Error in control loop: {e}")

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
