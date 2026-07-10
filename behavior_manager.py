from robot_bridge.client import RobotBridgeClient


class BehaviorManager:
    def __init__(self, robot_client=None):
        self.robot = robot_client or RobotBridgeClient()

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

        if mission_type == "FOLLOW_PERSON":
            return (
                f"Behavior: Tracking target '{target}'. "
                "Robot Bridge is ready for safe motion execution."
            )

        if mission_type == "MOVE_FORWARD":
            return (
                "Behavior: Moving forward briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            )

        if mission_type == "TURN_LEFT":
            return (
                "Behavior: Turning left briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            )

        if mission_type == "TURN_RIGHT":
            return (
                "Behavior: Turning right briefly through the Robot Bridge. "
                "Automatic stop is enabled."
            )

        if mission_type == "FIND_OBJECT":
            return (
                f"Behavior: Searching for object '{target}'. "
                "Vision-guided search execution will be added next."
            )

        if mission_type == "RETURN_HOME":
            return (
                "Behavior: Return-home mission prepared. "
                "Navigation is not implemented yet."
            )

        if mission_type == "DESCRIBE_SCENE":
            return (
                "Behavior: Scene description requested. "
                "Using current robot context."
            )

        if mission_type == "STOP":
            return "Behavior: Robot stop requested."

        return "Behavior: No simulated behavior available."

    def execute(self, mission):
        """
        Execute a mission through the Robot Bridge.

        Motion commands are intentionally short. The Robot Bridge publishes
        an automatic zero-velocity stop after every motion command.
        """
        mission_type = mission.mission_type
        status = mission.status
        target = mission.target

        if status == "REJECTED":
            return {
                "ok": False,
                "executed": False,
                "behavior": mission_type,
                "reason": "Mission rejected.",
            }

        if status == "CANCELLED" or mission_type == "STOP":
            robot_result = self.robot.stop()

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "behavior": "STOP",
                "reason": "Robot stop command sent.",
                "robot_result": robot_result,
            }

        if mission_type == "FOLLOW_PERSON":
            robot_result = self.robot.move_forward(
                speed=0.08,
                seconds=0.50,
            )

            return {
                "ok": bool(robot_result.get("ok")),
                "executed": True,
                "behavior": "FOLLOW_PERSON",
                "target": target,
                "reason": "Safe forward test motion completed.",
                "robot_result": robot_result,
            }

        if mission_type == "MOVE_FORWARD":
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

        if mission_type == "TURN_LEFT":
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

        if mission_type == "TURN_RIGHT":
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

        if mission_type == "FIND_OBJECT":
            return {
                "ok": True,
                "executed": False,
                "behavior": "FIND_OBJECT",
                "target": target,
                "reason": "Vision-guided search behavior is the next milestone.",
            }

        if mission_type == "RETURN_HOME":
            return {
                "ok": True,
                "executed": False,
                "behavior": "RETURN_HOME",
                "reason": "Navigation is not implemented yet.",
            }

        if mission_type == "DESCRIBE_SCENE":
            return {
                "ok": True,
                "executed": False,
                "behavior": "DESCRIBE_SCENE",
                "reason": "Information-only mission.",
            }

        return {
            "ok": False,
            "executed": False,
            "behavior": mission_type,
            "reason": f"No executable behavior for '{mission_type}'.",
        }
