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

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import draccus
import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if not math.isfinite(float(self.fps)) or self.fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {self.fps}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return draccus.decode(cls, config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return draccus.encode(self)


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Optional LeRobot-saved checkpoint. Leave empty to build the policy config
    # purely from `policy_config_overrides` (HF-original mode). Kept after the
    # required fields above to satisfy the dataclass default-ordering rule.
    pretrained_name_or_path: str = field(
        default="",
        metadata={
            "help": (
                "LeRobot-saved checkpoint (HF repo id or local dir). "
                "Leave empty when using an HF-original checkpoint, in which case "
                "the policy config is built entirely from --policy_config_overrides."
            )
        },
    )

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})

    # Bounded sessions: end the control loop after this many seconds and shut down
    # gracefully (same path as Ctrl-C: stop, optional home move, disconnect).
    # 0 keeps the historical run-until-interrupted behavior.
    duration: float = field(
        default=0.0,
        metadata={"help": "Stop the control loop after this many seconds; 0 runs until interrupted."},
    )

    # Prediction-only preflight: everything runs (robot connect, cameras, server
    # handshake, chunk prediction/aggregation) except hardware dispatch — no action is
    # sent to the robot, including the move-to-initial-position on shutdown.
    dispatch_actions: bool = field(
        default=True,
        metadata={
            "help": (
                "When false, run prediction-only: observations stream and predicted chunks "
                "are received and aggregated, but no action is sent to the robot and the "
                "shutdown move-to-initial-position is skipped."
            )
        },
    )

    # The SO100 production path opts into latency-aware scheduling while other
    # async clients retain the historical discrete queue behavior.
    control_mode: str = field(
        default="queue",
        metadata={"help": "Action scheduling mode: queue or timestamped temporal_ensemble."},
    )
    temporal_ensemble_m: float = field(
        default=0.5,
        metadata={"help": "Exponential age-decay coefficient for temporal ensembling."},
    )
    temporal_max_step: float | None = field(
        default=15.0,
        metadata={"help": "Direction-preserving outer joint-space step cap; null disables it."},
    )
    temporal_ring_capacity: int = field(
        default=6,
        metadata={"help": "Maximum timestamped plans retained for temporal ensembling."},
    )
    temporal_warmup_chunks: int = field(
        default=0,
        metadata={"help": "Complete chunks to validate and discard before temporal execution."},
    )

    # Robot-specific shutdown is opt-in. None means clear queued motion and disconnect;
    # a supplied target must exactly match the robot's action feature keys.
    shutdown_home_position: dict[str, float] | None = field(
        default=None,
        metadata={"help": "Optional exact action-key target for a slow shutdown home move."},
    )

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Optional rename map applied server-side to translate robot observation keys
    # into the names expected by the policy's processor.
    rename_map: dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Rename map applied server-side via the policy preprocessor's "
                "RenameObservationsProcessorStep, used when the robot's observation "
                "keys differ from the policy's expected feature keys."
            )
        },
    )

    # Draccus CLI-style overrides applied to the policy config server-side before
    # the policy is instantiated, e.g. ``["--inference_action_mode=continuous"]``.
    policy_config_overrides: list[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of draccus CLI overrides (e.g. '--inference_action_mode=continuous') "
                "applied server-side to the saved policy config before instantiation."
            )
        },
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        # `pretrained_name_or_path` is optional, but the server needs at least one
        # source (a saved checkpoint or config overrides) to build the policy.
        if not self.pretrained_name_or_path and not self.policy_config_overrides:
            raise ValueError(
                "Either `pretrained_name_or_path` (LeRobot-saved checkpoint) or "
                "`policy_config_overrides` (HF-original checkpoint) must be set."
            )

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if not math.isfinite(float(self.fps)) or self.fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {self.fps}")

        if self.duration < 0:
            raise ValueError(f"duration must be nonnegative (0 = unbounded), got {self.duration}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.control_mode not in {"queue", "temporal_ensemble"}:
            raise ValueError(f"control_mode must be queue or temporal_ensemble, got {self.control_mode!r}")
        if not math.isfinite(self.temporal_ensemble_m) or self.temporal_ensemble_m < 0:
            raise ValueError("temporal_ensemble_m must be finite and nonnegative")
        if self.temporal_max_step is not None and (
            not math.isfinite(self.temporal_max_step) or self.temporal_max_step <= 0
        ):
            raise ValueError("temporal_max_step must be finite and positive when set")
        if self.temporal_ring_capacity <= 0:
            raise ValueError("temporal_ring_capacity must be positive")
        if self.temporal_warmup_chunks < 0:
            raise ValueError("temporal_warmup_chunks must be nonnegative")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return draccus.decode(cls, config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return draccus.encode(self)
