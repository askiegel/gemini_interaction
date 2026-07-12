import time

from robot_bridge.client import RobotBridgeClient


class BehaviorManager:
    SEARCH_TURN_SPEED = 0.40
    SEARCH_TURN_SECONDS = 0.35

    CENTER_TURN_SPEED = 0.60
    CENTER_TURN_SECONDS = 0.40

    APPROACH_SPEED = 0.08
    APPROACH_SECONDS = 0.80

    DEFAULT_IMAGE_WIDTH = 640.0
    CENTER_TOLERANCE_PIXELS = 95.0
    ARRIVAL_AREA = 75000.0

    MAX_FIND_CYCLES = 30
    FIND_CYCLE_PAUSE = 1.50

    def __init__(self, robot_client=None, vision_adapter=None):
        self.robot = robot_client or RobotBridgeClient()
        self.vision = vision_adapter

    def simulate(self, mission):
        """
        Describe the intended behavior without moving the robot.
        """
        mission_type = mission.mission_type
        status = mission.status
        target = mission.target

        if status == "REJECTED":
            return "Behavior: Mission rejected. No robot action taken."

        if status == "CANCELLED":
            return "Behavior: Active mission cancelled. Robot should stop."

        simulations = {
            "FOLLOW_PERSON": (
                f"Behavior: Tracking target '{target}'. "
                "Robot Bridge is ready for safe motion execution."
            ),
            "MOVE_FORWARD": (
                "Behavior: Moving forward briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            ),
            "TURN_LEFT": (
                "Behavior: Turning left briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            ),
            "TURN_RIGHT": (
                "Behavior: Turning right briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            ),
            "FIND_OBJECT": (
                f"Behavior: Query vision for '{target}', then execute one "
                "bounded search, centering, approach, or arrival action."
            ),
            "RETURN_HOME": (
                "Behavior: Return-home mission prepared. "
                "Navigation is not implemented yet."
            ),
            "DESCRIBE_SCENE": (
                "Behavior: Scene description requested. "
                "Using current robot context."
            ),
            "STOP": "Behavior: Robot stop requested.",
        }

        return simulations.get(
            mission_type,
            "Behavior: No simulated behavior available.",
        )

    def execute(self, mission):
        """
        Execute one bounded mission action through the Robot Bridge.
        """
        if mission.status == "REJECTED":
            return {
                "ok": False,
                "executed": False,
                "behavior": mission.mission_type,
                "reason": "Mission rejected.",
            }

        if mission.status == "CANCELLED" or mission.mission_type == "STOP":
            return self._execute_stop()

        handlers = {
            "FOLLOW_PERSON": self._execute_follow_person,
            "MOVE_FORWARD": self._execute_move_forward,
            "TURN_LEFT": self._execute_turn_left,
            "TURN_RIGHT": self._execute_turn_right,
            "FIND_OBJECT": self._execute_find_object,
            "RETURN_HOME": self._execute_return_home,
            "DESCRIBE_SCENE": self._execute_describe_scene,
        }

        handler = handlers.get(mission.mission_type)

        if handler is None:
            return {
                "ok": False,
                "executed": False,
                "behavior": mission.mission_type,
                "reason": (
                    f"No executable behavior for "
                    f"'{mission.mission_type}'."
                ),
            }

        return handler(mission)

    def _execute_stop(self):
        robot_result = self.robot.stop()

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "STOP",
            "state": "STOPPED",
            "reason": "Robot stop command sent.",
            "robot_result": robot_result,
        }

    def _execute_follow_person(self, mission):
        robot_result = self.robot.move_forward(
            speed=0.08,
            seconds=0.50,
        )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "FOLLOW_PERSON",
            "target": mission.target,
            "reason": "Safe forward test motion completed.",
            "robot_result": robot_result,
        }

    def _execute_move_forward(self, mission):
        robot_result = self.robot.move_forward(
            speed=0.08,
            seconds=0.50,
        )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "MOVE_FORWARD",
            "reason": "Executed short forward movement.",
            "robot_result": robot_result,
        }

    def _execute_turn_left(self, mission):
        robot_result = self.robot.turn_left(
            speed=0.50,
            seconds=0.40,
        )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "TURN_LEFT",
            "reason": "Executed short left turn.",
            "robot_result": robot_result,
        }

    def _execute_turn_right(self, mission):
        robot_result = self.robot.turn_right(
            speed=0.50,
            seconds=0.40,
        )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "TURN_RIGHT",
            "reason": "Executed short right turn.",
            "robot_result": robot_result,
        }

    def _execute_find_object(self, mission):
        """
        Execute a bounded autonomous find-object mission.

        The behavior repeatedly performs:

            perception -> decision -> bounded motion -> automatic stop

        until the target is considered arrived or MAX_FIND_CYCLES is reached.
        """
        target_name = str(mission.target or "").strip().lower()

        if not target_name:
            return {
                "ok": False,
                "executed": False,
                "behavior": "FIND_OBJECT",
                "reason": "FIND_OBJECT requires a target.",
            }

        if self.vision is None:
            return {
                "ok": False,
                "executed": False,
                "behavior": "FIND_OBJECT",
                "target": target_name,
                "reason": "Vision Adapter is not configured.",
            }

        cycle_history = []

        for cycle_number in range(1, self.MAX_FIND_CYCLES + 1):
            cycle_result = self._execute_find_object_cycle(
                target_name=target_name,
                cycle_number=cycle_number,
            )

            cycle_history.append(
                {
                    "cycle": cycle_number,
                    "state": cycle_result.get("state"),
                    "reason": cycle_result.get("reason"),
                    "horizontal_error": cycle_result.get(
                        "horizontal_error"
                    ),
                }
            )

            if not cycle_result.get("ok"):
                cycle_result["cycles_completed"] = cycle_number
                cycle_result["cycle_history"] = cycle_history
                return cycle_result

            if cycle_result.get("state") == "ARRIVED":
                cycle_result["cycles_completed"] = cycle_number
                cycle_result["cycle_history"] = cycle_history
                return cycle_result

            time.sleep(self.FIND_CYCLE_PAUSE)

        stop_result = self.robot.stop()

        return {
            "ok": bool(stop_result.get("ok")),
            "executed": True,
            "completed": False,
            "behavior": "FIND_OBJECT",
            "target": target_name,
            "state": "SEARCH_LIMIT_REACHED",
            "reason": (
                f"Target was not reached after "
                f"{self.MAX_FIND_CYCLES} safe cycles."
            ),
            "cycles_completed": self.MAX_FIND_CYCLES,
            "cycle_history": cycle_history,
            "robot_result": stop_result,
        }

    def _execute_find_object_cycle(
        self,
        target_name,
        cycle_number,
    ):
        try:
            target = self.vision.find_target(target_name)
        except Exception as exc:
            stop_result = self.robot.stop()

            return {
                "ok": False,
                "executed": True,
                "behavior": "FIND_OBJECT",
                "target": target_name,
                "state": "VISION_ERROR",
                "cycle": cycle_number,
                "reason": f"Vision query failed: {exc}",
                "robot_result": stop_result,
            }

        if not target.get("found"):
            robot_result = self.robot.turn_left(
                speed=self.SEARCH_TURN_SPEED,
                seconds=self.SEARCH_TURN_SECONDS,
            )

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "behavior": "FIND_OBJECT",
                "target": target_name,
                "state": "SEARCHING",
                "cycle": cycle_number,
                "reason": (
                    f"{target_name} not visible. "
                    "Executed one short search turn."
                ),
                "vision_result": target,
                "robot_result": robot_result,
            }

        cx = target.get("cx")
        area = target.get("area")
        image_width = (
            target.get("image_width")
            or self.DEFAULT_IMAGE_WIDTH
        )

        if cx is None:
            stop_result = self.robot.stop()

            return {
                "ok": False,
                "executed": True,
                "behavior": "FIND_OBJECT",
                "target": target_name,
                "state": "INVALID_DETECTION",
                "cycle": cycle_number,
                "reason": "Target detection has no horizontal center.",
                "vision_result": target,
                "robot_result": stop_result,
            }

        image_center = float(image_width) / 2.0
        horizontal_error = float(cx) - image_center

        if (
            area is not None
            and float(area) >= self.ARRIVAL_AREA
            and abs(horizontal_error)
            <= self.CENTER_TOLERANCE_PIXELS
        ):
            robot_result = self.robot.stop()

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "completed": True,
                "behavior": "FIND_OBJECT",
                "target": target_name,
                "state": "ARRIVED",
                "cycle": cycle_number,
                "reason": f"Arrived at {target_name}.",
                "horizontal_error": horizontal_error,
                "vision_result": target,
                "robot_result": robot_result,
            }

        if horizontal_error < -self.CENTER_TOLERANCE_PIXELS:
            robot_result = self.robot.turn_left(
                speed=self.CENTER_TURN_SPEED,
                seconds=self.CENTER_TURN_SECONDS,
            )

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "behavior": "FIND_OBJECT",
                "target": target_name,
                "state": "CENTERING_LEFT",
                "cycle": cycle_number,
                "reason": (
                    f"{target_name} is left in the camera image. "
                    "Turning robot left to center it."
                ),
                "horizontal_error": horizontal_error,
                "vision_result": target,
                "robot_result": robot_result,
            }

        if horizontal_error > self.CENTER_TOLERANCE_PIXELS:
            robot_result = self.robot.turn_right(
                speed=self.CENTER_TURN_SPEED,
                seconds=self.CENTER_TURN_SECONDS,
            )

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "behavior": "FIND_OBJECT",
                "target": target_name,
                "state": "CENTERING_RIGHT",
                "cycle": cycle_number,
                "reason": (
                    f"{target_name} is right in the camera image. "
                    "Turning robot right to center it."
                ),
                "horizontal_error": horizontal_error,
                "vision_result": target,
                "robot_result": robot_result,
            }

        robot_result = self.robot.move_forward(
            speed=self.APPROACH_SPEED,
            seconds=self.APPROACH_SECONDS,
        )

        return {
            "ok": bool(robot_result.get("ok")),
            "executed": True,
            "behavior": "FIND_OBJECT",
            "target": target_name,
            "state": "APPROACHING",
            "cycle": cycle_number,
            "reason": f"{target_name} is centered. Moving closer.",
            "horizontal_error": horizontal_error,
            "vision_result": target,
            "robot_result": robot_result,
        }

    def _execute_return_home(self, mission):
        return {
            "ok": True,
            "executed": False,
            "behavior": "RETURN_HOME",
            "reason": "Navigation is not implemented yet.",
        }

    def _execute_describe_scene(self, mission):
        return {
            "ok": True,
            "executed": False,
            "behavior": "DESCRIBE_SCENE",
            "reason": "Information-only mission.",
        }
