#!/usr/bin/env python3

"""Read-only guarded Nav2 readiness probe for Tony2."""

import json
import math
import os
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

import rclpy

from geometry_msgs.msg import PoseWithCovarianceStamped
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
            "map_server": None,
            "amcl": None,
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

        # AMCL is the authoritative localization source for
        # persistent fixed-map navigation. Observe it only;
        # this probe never publishes a pose or transform.
        self._latest_amcl_pose = None

        self._amcl_pose_subscription = (
            self.create_subscription(
                PoseWithCovarianceStamped,
                "/amcl_pose",
                self._amcl_pose_received,
                10,
            )
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

    def _amcl_pose_received(self, message):
        """
        Record the newest read-only AMCL map pose.

        No transform, initial pose, velocity, goal, or other
        command is published from this callback.
        """

        frame_id = str(
            message.header.frame_id
        ).lstrip("/")

        if frame_id != "map":
            return

        pose = message.pose.pose
        covariance = list(
            message.pose.covariance
        )

        quaternion = pose.orientation

        siny_cosp = 2.0 * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp,
        )

        def standard_deviation(index):
            try:
                variance = float(
                    covariance[index]
                )
            except (
                IndexError,
                TypeError,
                ValueError,
            ):
                return None

            if (
                not math.isfinite(variance)
                or variance < 0.0
            ):
                return None

            return math.sqrt(variance)

        x_std = standard_deviation(0)
        y_std = standard_deviation(7)
        yaw_std = standard_deviation(35)

        self._latest_amcl_pose = {
            "received_at": utc_now(),
            "observed_at_monotonic":
                time.monotonic(),
            "pose": {
                "frame_id": "map",
                "position": {
                    "x": float(
                        pose.position.x
                    ),
                    "y": float(
                        pose.position.y
                    ),
                    "z": float(
                        pose.position.z
                    ),
                },
                "yaw_radians": float(yaw),
                "yaw_degrees": float(
                    math.degrees(yaw)
                ),
                "uncertainty": {
                    "x_standard_deviation":
                        x_std,
                    "y_standard_deviation":
                        y_std,
                    "yaw_standard_deviation_radians":
                        yaw_std,
                },
            },
        }


    def _current_map_pose(self):
        """
        Return the latest map-to-base_link transform as Mayday's
        current fixed-map position and heading.

        AMCL covariance is retained from the newest /amcl_pose
        observation, but position freshness comes from the live
        TF tree used by Nav2 itself.
        """

        try:
            transform = (
                self._tf_buffer.lookup_transform(
                    "map",
                    "base_link",
                    Time(),
                    timeout=Duration(
                        seconds=0.05
                    ),
                )
            )

        except Exception:
            return None

        translation = (
            transform.transform.translation
        )

        quaternion = (
            transform.transform.rotation
        )

        siny_cosp = 2.0 * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
        )

        yaw = math.atan2(
            siny_cosp,
            cosy_cosp,
        )

        uncertainty = {
            "x_standard_deviation": None,
            "y_standard_deviation": None,
            "yaw_standard_deviation_radians":
                None,
        }

        if isinstance(
            self._latest_amcl_pose,
            dict,
        ):
            amcl_pose = (
                self._latest_amcl_pose.get(
                    "pose"
                )
                or {}
            )

            amcl_uncertainty = (
                amcl_pose.get(
                    "uncertainty"
                )
                or {}
            )

            uncertainty.update(
                amcl_uncertainty
            )

        return {
            "frame_id": "map",
            "position": {
                "x": float(
                    translation.x
                ),
                "y": float(
                    translation.y
                ),
                "z": float(
                    translation.z
                ),
            },
            "yaw_radians": float(yaw),
            "yaw_degrees": float(
                math.degrees(yaw)
            ),
            "uncertainty": uncertainty,
        }


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
        map_server_active = self._is_active(
            "map_server"
        )

        localization_active = self._is_active(
            "amcl"
        )

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
                map_server_active,
                localization_active,
                planner_active,
                controller_active,
                navigator_active,
                action_server_ready,
                transform_ready,
            )
        )

        current_pose = (
            self._current_map_pose()
        )

        current_pose_received_at = (
            utc_now()
            if current_pose is not None
            else None
        )

        current_pose_observed_at_monotonic = (
            time.monotonic()
            if current_pose is not None
            else None
        )

        payload = {
            "ok": True,
            "service":
                "tony2_guarded_navigation_probe",
            "host": "Tony2",
            "timestamp": utc_now(),
            "read_only": True,
            "goal_sent": False,
            "pose": current_pose,
            "pose_received_at":
                current_pose_received_at,
            "pose_observed_at_monotonic":
                current_pose_observed_at_monotonic,
            "map_server_enabled":
                map_server_active,
            "localization_enabled":
                localization_active,
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
