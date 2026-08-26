#!/usr/bin/env python3

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

"""Supervise the two local guarded YAM follower Portals and publish readiness."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from lerobot.utils.yam_portal import READINESS_SCHEMA, process_start_ticks

FOLLOWER_CONFIGS = (
    {
        "can_channel": "can_follower_r",
        "port": 1234,
        "role": "follower",
        "side": "right",
        "gripper": "linear_4310",
    },
    {
        "can_channel": "can_follower_l",
        "port": 1235,
        "role": "follower",
        "side": "left",
        "gripper": "linear_4310",
    },
)
LEADER_CONFIGS = (
    {
        "can_channel": "can_leader_r",
        "port": 5001,
        "role": "leader",
        "side": "right",
        "gripper": "yam_teaching_handle",
    },
    {
        "can_channel": "can_leader_l",
        "port": 5002,
        "role": "leader",
        "side": "left",
        "gripper": "yam_teaching_handle",
    },
)
FULL_CONFIGS = (*FOLLOWER_CONFIGS, *LEADER_CONFIGS)


def check_can_interfaces(configs=FOLLOWER_CONFIGS) -> None:
    missing = []
    for config in configs:
        result = subprocess.run(  # nosec B607 - fixed argv, PATH-resolved ip is intended
            ["ip", "link", "show", config["can_channel"]],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode or not any(state in result.stdout for state in ("state UP", "state UNKNOWN")):
            missing.append(config["can_channel"])
    if missing:
        raise RuntimeError(f"Missing or unavailable YAM CAN interfaces: {', '.join(missing)}")


def launch_portals(
    configs=FOLLOWER_CONFIGS, *, command_watchdog_seconds: float = 4.0
) -> list[subprocess.Popen]:
    processes = []
    for config in configs:
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.yam_portal_server",
            "--can-channel",
            config["can_channel"],
            "--port",
            str(config["port"]),
            "--role",
            config["role"],
            "--gripper",
            config["gripper"],
            "--command-watchdog-seconds",
            str(command_watchdog_seconds),
        ]
        processes.append(subprocess.Popen(command))
    return processes


def assert_portals_alive(processes, configs=FOLLOWER_CONFIGS) -> None:
    if len(processes) != len(configs):
        raise RuntimeError("YAM Portal process/config count mismatch; stopping every Portal.")
    for process, config in zip(processes, configs, strict=True):
        if (returncode := process.poll()) is not None:
            raise RuntimeError(
                f"YAM Portal for {config['can_channel']} on port {config['port']} exited "
                f"with status {returncode}; stopping every Portal."
            )


def wait_for_portals(processes, configs=FOLLOWER_CONFIGS, timeout_seconds: float = 20.0) -> None:
    pending = {int(config["port"]) for config in configs}
    deadline = time.monotonic() + timeout_seconds
    while pending and time.monotonic() < deadline:
        assert_portals_alive(processes, configs)
        for port in tuple(pending):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                    pending.remove(port)
            except OSError:
                pass
        if pending:
            time.sleep(0.1)
    assert_portals_alive(processes, configs)
    if pending:
        raise RuntimeError(f"Timed out waiting for YAM Portal ports: {sorted(pending)}")


def verify_motor_feedback(processes, configs=FOLLOWER_CONFIGS, timeout_seconds: float = 5.0) -> None:
    for config in configs:
        assert_portals_alive(processes, configs)
        interface = config["can_channel"]
        filters = [",".join(f"{can_id:03X}:7FF" for can_id in range(0x11, 0x18))]
        if config.get("role") == "leader":
            # The leader contract requires both arm-motor traffic and a live
            # teaching-handle encoder at standard ID 0x50E.
            filters = [
                ",".join(f"{can_id:03X}:7FF" for can_id in range(0x11, 0x17)),
                "50E:7FF",
            ]
        for can_filter in filters:
            try:
                result = subprocess.run(  # nosec B607 - fixed argv, PATH-resolved candump is intended
                    ["candump", "-n", "1", f"{interface},{can_filter}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=timeout_seconds,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("candump is required to verify YAM CAN feedback.") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"No required YAM CAN feedback observed on {interface} after startup."
                ) from exc
            if result.returncode:
                raise RuntimeError(f"CAN-feedback candump failed on {interface}: {result.stderr.strip()}")


def write_readiness(path: str | Path, processes, configs=FOLLOWER_CONFIGS) -> Path:
    readiness_path = Path(path)
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    if len(processes) != len(configs):
        raise RuntimeError("YAM Portal process/config count mismatch; no readiness will be emitted.")
    portals = {}
    for process, config in zip(processes, configs, strict=True):
        portals[config["can_channel"]] = {
            "pid": process.pid,
            "start_ticks": process_start_ticks(process.pid),
            "port": int(config["port"]),
            "role": config["role"],
            "side": config["side"],
        }
    payload = {
        "schema": READINESS_SCHEMA,
        "mode": "full" if any(config["role"] == "leader" for config in configs) else "follower_only",
        "supervisor_pid": os.getpid(),
        "supervisor_start_ticks": process_start_ticks(os.getpid()),
        "portals": portals,
        "can_feedback_verified_after_launch": [config["can_channel"] for config in configs],
        "created_unix": time.time(),
    }
    temporary = readiness_path.with_name(f".{readiness_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.chmod(0o600)
        os.replace(temporary, readiness_path)
    finally:
        temporary.unlink(missing_ok=True)
    return readiness_path


def stop_portals(processes) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch guarded local bimanual YAM Portals.")
    parser.add_argument("--readiness-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("follower_only", "full"), default="follower_only")
    parser.add_argument("--command-watchdog-seconds", type=float, default=4.0)
    args = parser.parse_args()
    configs = FOLLOWER_CONFIGS if args.mode == "follower_only" else FULL_CONFIGS
    processes = []
    readiness_path = None

    def request_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGHUP, request_shutdown)
    try:
        check_can_interfaces(configs)
        processes = launch_portals(configs, command_watchdog_seconds=args.command_watchdog_seconds)
        wait_for_portals(processes, configs)
        verify_motor_feedback(processes, configs)
        readiness_path = write_readiness(args.readiness_file, processes, configs)
        while True:
            assert_portals_alive(processes, configs)
            readiness_path.touch()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if readiness_path is not None:
            readiness_path.unlink(missing_ok=True)
        stop_portals(processes)


if __name__ == "__main__":
    main()
