#!/usr/bin/env python3

"""Read-only guarded Nav2 readiness probe for Tony2."""

import json
import os
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

import rclpy

from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer
from tf2_ros import TransformListener


SNAPSHOT_PATH = Path(
    os.getenv(
        "TONY2_NAVIGATION_SNAPSHOT",
        "/tmp/tony2_navigation_snapshot.json",
    )
)


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


class NavigationProbe(Node):
    """Observe guarded Nav2 without sending goals."""

    def __init__(self):
        super().__init__(
            "tony2_guarded_navigation_probe"
        )

        self._states = {
            "planner_server": None,
            "controller_server": None,
            "bt_navigator": None,
        }

        self._pending = {}

        # Do not use Node._clients here. rclpy.Node owns
        # that private attribute internally.
        self._state_clients = {
            name: self.create_client(
                GetState,
                f"/{name}/get_state",
            )
            for name in self._states
        }

        self._navigate_to_pose = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )

        self._tf_buffer = Buffer()

        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
        )

        self.create_timer(
            0.25,
            self._request_states,
        )

        self.create_timer(
            0.5,
            self._write_snapshot,
        )

    def _state_done(
        self,
        name,
        future,
    ):
        self._pending.pop(
            name,
            None,
        )

        try:
            response = future.result()

            self._states[name] = {
                "id": int(
                    response.current_state.id
                ),
                "label": str(
                    response.current_state.label
                ),
            }

        except Exception:
            self._states[name] = None

    def _request_states(self):
        for name, client in self._state_clients.items():
            pending = self._pending.get(name)

            if (
                pending is not None
                and not pending.done()
            ):
                continue

            if not client.service_is_ready():
                self._states[name] = None
                continue

            future = client.call_async(
                GetState.Request()
            )

            self._pending[name] = future

            future.add_done_callback(
                lambda finished, target=name:
                    self._state_done(
                        target,
                        finished,
                    )
            )

    def _is_active(self, name):
        state = self._states.get(name)

        return bool(
            isinstance(state, dict)
            and state.get("id")
            == State.PRIMARY_STATE_ACTIVE
        )

    def _transform_ready(self):
        try:
            return bool(
                self._tf_buffer.can_transform(
                    "map",
                    "base_link",
                    Time(),
                    timeout=Duration(
                        seconds=0.05
                    ),
                )
            )

        except Exception:
            return False

    def _write_snapshot(self):
        planner_active = self._is_active(
            "planner_server"
        )

        controller_active = self._is_active(
            "controller_server"
        )

        navigator_active = self._is_active(
            "bt_navigator"
        )

        action_server_ready = bool(
            self._navigate_to_pose.server_is_ready()
        )

        transform_ready = (
            self._transform_ready()
        )

        ready = all(
            (
                planner_active,
                controller_active,
                navigator_active,
                action_server_ready,
                transform_ready,
            )
        )

        payload = {
            "ok": True,
            "service":
                "tony2_guarded_navigation_probe",
            "host": "Tony2",
            "timestamp": utc_now(),
            "read_only": True,
            "goal_sent": False,
            "planner_enabled":
                planner_active,
            "controller_enabled":
                controller_active,
            "navigator_enabled":
                navigator_active,
            "action_server_ready":
                action_server_ready,
            "transform_ready":
                transform_ready,
            "ready": ready,
            "states": self._states,
            "observed_at_monotonic":
                time.monotonic(),
        }

        temporary = SNAPSHOT_PATH.with_suffix(
            SNAPSHOT_PATH.suffix + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                payload,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary,
            SNAPSHOT_PATH,
        )


def main():
    rclpy.init()

    node = NavigationProbe()

    try:
        rclpy.spin(node)

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
