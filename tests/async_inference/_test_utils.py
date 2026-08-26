from __future__ import annotations

import pickle
from typing import Any


class FakeGrpcContext:
    def __init__(self, peer: str):
        self._peer = peer

    def peer(self) -> str:
        return self._peer


def make_policy_setup_request(**config: Any):
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.transport import services_pb2

    return services_pb2.PolicySetup(data=pickle.dumps(RemotePolicyConfig(**config)))


def stop_robot_client_fast(client) -> None:
    if client.robot.is_connected:
        client._slow_move_to_position = lambda *args, **kwargs: None
        client.stop()
