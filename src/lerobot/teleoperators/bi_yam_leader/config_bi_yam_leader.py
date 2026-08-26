#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
from dataclasses import dataclass
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_yam_leader")
@dataclass
class BiYamLeaderConfig(TeleoperatorConfig):
    """Read-only guarded Portal configuration for two YAM teaching handles."""

    left_arm_port: int = 5002
    right_arm_port: int = 5001
    server_host: str = "localhost"
    portal_readiness_file: Path | None = None
    require_portal_readiness: bool = True
    readiness_max_age_seconds: float = 3.0
    rpc_timeout_seconds: float = 1.0
    max_observation_age_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not (1 <= self.left_arm_port <= 65535 and 1 <= self.right_arm_port <= 65535):
            raise ValueError("YAM leader Portal ports must be in [1, 65535].")
        if self.left_arm_port == self.right_arm_port:
            raise ValueError("YAM left and right leader Portal ports must be distinct.")
        for name in ("readiness_max_age_seconds", "rpc_timeout_seconds", "max_observation_age_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}.")
        if self.require_portal_readiness and self.portal_readiness_file is None:
            raise ValueError(
                "Guarded YAM leaders require portal_readiness_file from "
                "lerobot-yam-portals --mode=full; set require_portal_readiness=false "
                "only for isolated fake-Portal tests."
            )
