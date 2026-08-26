#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Shared safety contracts and dependency-light plumbing for guarded local YAM Portals."""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any

import numpy as np

FOLLOWER_PORTAL_SCHEMA = "yam_guarded_follower_portal_v1"
LEADER_PORTAL_SCHEMA = "yam_guarded_leader_portal_v1"
READINESS_SCHEMA = "bi_yam_portals_v3"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
MAX_READINESS_BYTES = 64 * 1024

logger = logging.getLogger(__name__)


def finite_vector(value: Any, *, size: int, label: str) -> np.ndarray:
    """Return a finite float64 vector with one exact, safety-checked shape."""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric vector of shape ({size},).") from exc
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be finite with shape ({size},), got {vector!r}.")
    return vector


def validate_guarded_observation_envelope(
    envelope: Any,
    *,
    schema: str,
    observation_label: str,
    timestamp_subject: str,
    last_sequence: int,
    max_age_seconds: float,
) -> int:
    """Validate the common schema, sequence, and freshness contract of a guarded observation."""
    if not isinstance(envelope, dict) or envelope.get("schema") != schema:
        raise RuntimeError(f"{observation_label} observation envelope has an incompatible schema.")
    sequence = envelope.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= last_sequence:
        raise RuntimeError(
            f"{observation_label} observation sequence is stale: {sequence!r} <= {last_sequence}."
        )
    server_time_ns = envelope.get("server_time_unix_ns")
    if not isinstance(server_time_ns, int) or isinstance(server_time_ns, bool):
        raise RuntimeError(f"{observation_label} {timestamp_subject} lacks an integer server timestamp.")
    age = (time.time_ns() - server_time_ns) / 1e9
    if age < -1.0 or age > max_age_seconds:
        raise RuntimeError(
            f"{observation_label} observation is stale (age={age:.3f}s, limit={max_age_seconds:.3f}s)."
        )
    return sequence


class GuardedYamPortalClient:
    """Shared timeout-bounded socket mechanics for one guarded local YAM Portal."""

    RPC_NAMES: frozenset[str] = frozenset()
    PORTAL_SCHEMA = ""
    PORTAL_LABEL = "YAM Portal"

    def __init__(self, port: int, host: str, rpc_timeout_seconds: float, max_observation_age_seconds: float):
        self.port = int(port)
        self.host = host
        self.rpc_timeout_seconds = float(rpc_timeout_seconds)
        self.max_observation_age_seconds = float(max_observation_age_seconds)
        self._client = None
        self._health: dict[str, Any] | None = None
        self._last_observation_sequence = -1

    def _rpc(self, name: str, *args):
        if self._client is None:
            raise RuntimeError(f"{self.PORTAL_LABEL} client is not connected.")
        if name not in self.RPC_NAMES:
            raise RuntimeError(f"{self.PORTAL_LABEL} does not implement required RPC {name!r}.")
        try:
            return getattr(self._client, name)(*args).result(timeout=self.rpc_timeout_seconds)
        except Exception as exc:
            raise RuntimeError(
                f"{self.PORTAL_LABEL} RPC {name!r} failed or exceeded {self.rpc_timeout_seconds:.3f}s."
            ) from exc

    def _connect_client(self, client: Any) -> None:
        self._client = client
        try:
            if not client.connect(timeout=self.rpc_timeout_seconds):
                raise RuntimeError(
                    f"{self.PORTAL_LABEL} {self.host}:{self.port} did not connect within "
                    f"{self.rpc_timeout_seconds:.3f}s."
                )
            health = self._rpc("health")
            if not isinstance(health, dict) or health.get("schema") != self.PORTAL_SCHEMA:
                raise RuntimeError(f"{self.PORTAL_LABEL} returned incompatible health: {health!r}.")
            self._validate_health(health)
            self._health = health
            self._reset_sequences()
        except Exception:
            self._close_client()
            raise

    def _validate_health(self, health: dict[str, Any]) -> None:
        raise NotImplementedError

    def _reset_sequences(self) -> None:
        self._last_observation_sequence = -1

    def _close_client(self) -> None:
        client, self._client = self._client, None
        self._health = None
        if client is None:
            return
        try:
            client.close()
        except Exception:
            logger.warning(
                "Failed to close %s client %s:%s", self.PORTAL_LABEL, self.host, self.port, exc_info=True
            )

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._health is not None

    @property
    def health(self) -> dict[str, Any]:
        if self._health is None:
            raise RuntimeError(f"{self.PORTAL_LABEL} health is unavailable before connect.")
        return self._health


def validate_yam_robot_state_live(robot: Any, *, max_age_seconds: float) -> float | None:
    """Fail closed when i2rt's control thread has stopped refreshing its cached joint state."""
    motor_chain = getattr(robot, "motor_chain", None)
    if getattr(motor_chain, "running", None) is False:
        raise RuntimeError("i2rt YAM motor chain has stopped; refusing to serve a frozen state.")
    timestamp = getattr(getattr(robot, "_joint_state", None), "timestamp", None)
    if timestamp is None:
        return None
    age = time.time() - float(timestamp)
    if not np.isfinite(age) or age > max_age_seconds:
        raise RuntimeError(f"i2rt YAM joint state is stale (age={age:.3f}s, limit={max_age_seconds:.3f}s).")
    return age


def process_start_ticks(pid: int) -> int:
    """Return Linux boot-relative process start ticks, which disambiguate PID reuse."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return int(text[text.rfind(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError(f"Could not fingerprint live YAM process {pid}: {exc}") from exc


def _read_private_token(path: str | Path) -> tuple[dict[str, Any], os.stat_result]:
    readiness_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(readiness_path, flags)
    except OSError as exc:
        raise RuntimeError(f"YAM Portal readiness token is unavailable: {readiness_path}") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"YAM Portal readiness token is not a regular file: {readiness_path}")
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise RuntimeError("YAM Portal readiness token must have mode 0600.")
        if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
            raise RuntimeError("YAM Portal readiness token must be owned by the rollout user.")
        raw = os.read(descriptor, MAX_READINESS_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_READINESS_BYTES:
        raise RuntimeError("YAM Portal readiness token exceeds the bounded schema size.")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("YAM Portal readiness token is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("YAM Portal readiness token must contain a JSON object.")
    return payload, file_stat


def _positive_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"YAM Portal readiness {label} must be a positive integer.")
    return value


def validate_yam_portal_readiness(
    path: str | Path,
    *,
    required_portals: dict[str, dict[str, Any]],
    max_age_seconds: float,
) -> dict[str, Any]:
    """Authenticate a fresh readiness token against every declared live process.

    ``required_portals`` maps canonical CAN channel names to exact ``role``,
    ``side``, and ``port`` values. A follower client may accept a full token with
    extra leader processes, but a token can never substitute one requested
    channel, role, side, or port for another.
    """
    payload, file_stat = _read_private_token(path)
    age = time.time() - file_stat.st_mtime
    if age < -1.0 or age > max_age_seconds:
        raise RuntimeError(
            f"YAM Portal readiness heartbeat is stale (age={age:.3f}s, limit={max_age_seconds:.3f}s)."
        )
    if payload.get("schema") != READINESS_SCHEMA:
        raise RuntimeError(f"Unsupported YAM Portal readiness schema: {payload.get('schema')!r}.")
    if payload.get("mode") not in {"follower_only", "full"}:
        raise RuntimeError("YAM Portal readiness has an invalid supervisor mode.")

    supervisor_pid = _positive_int(payload.get("supervisor_pid"), label="supervisor_pid")
    supervisor_ticks = _positive_int(payload.get("supervisor_start_ticks"), label="supervisor_start_ticks")
    raw_portals = payload.get("portals")
    if not isinstance(raw_portals, dict) or not raw_portals:
        raise RuntimeError("YAM Portal readiness must identify its child processes.")
    feedback_channels = payload.get("can_feedback_verified_after_launch")
    if not isinstance(feedback_channels, list) or not all(
        isinstance(channel, str) for channel in feedback_channels
    ):
        raise RuntimeError("YAM Portal readiness lacks verified post-launch CAN feedback.")
    if len(feedback_channels) != len(set(feedback_channels)):
        raise RuntimeError("YAM Portal readiness repeats a verified CAN channel.")

    identities = {supervisor_pid: supervisor_ticks}
    ports: set[int] = set()
    for channel, record in raw_portals.items():
        if not isinstance(channel, str) or not isinstance(record, dict):
            raise RuntimeError("YAM Portal readiness child records are malformed.")
        pid = _positive_int(record.get("pid"), label=f"{channel}.pid")
        ticks = _positive_int(record.get("start_ticks"), label=f"{channel}.start_ticks")
        port = _positive_int(record.get("port"), label=f"{channel}.port")
        if port > 65535:
            raise RuntimeError(f"YAM Portal readiness {channel}.port exceeds 65535.")
        if pid in identities or port in ports:
            raise RuntimeError("YAM Portal readiness contains duplicate process or port identities.")
        if record.get("role") not in {"follower", "leader"} or record.get("side") not in {
            "left",
            "right",
        }:
            raise RuntimeError(f"YAM Portal readiness record for {channel} has invalid role/side.")
        identities[pid] = ticks
        ports.add(port)

    for channel, expected in required_portals.items():
        record = raw_portals.get(channel)
        if not isinstance(record, dict):
            raise RuntimeError(f"YAM Portal readiness is missing required channel {channel}.")
        for field in ("role", "side", "port"):
            if record.get(field) != expected[field]:
                raise RuntimeError(
                    f"YAM Portal readiness {channel}.{field}={record.get(field)!r} does not "
                    f"match {expected[field]!r}."
                )
        if channel not in feedback_channels:
            raise RuntimeError(f"YAM Portal readiness lacks post-launch CAN feedback for {channel}.")

    for pid, expected_ticks in identities.items():
        if process_start_ticks(pid) != expected_ticks:
            raise RuntimeError(f"YAM Portal process identity changed for pid {pid}.")
    return payload
