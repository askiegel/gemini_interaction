class BehaviorManager:
    def simulate(self, mission):
        mission_type = mission.mission_type
        status = mission.status
        target = mission.target

        if status == "REJECTED":
            return "Behavior: Mission rejected. No robot action taken."

        if mission_type == "FOLLOW_PERSON":
            return (
                f"Behavior: Tracking target '{target}'. "
                "Navigation is ready. Motion controller waiting for ROS2 bridge."
            )

        if mission_type == "FIND_OBJECT":
            return (
                f"Behavior: Searching for object '{target}'. "
                "Vision context will be used when live detection is connected."
            )

        if mission_type == "RETURN_HOME":
            return (
                "Behavior: Return-home mission prepared. "
                "Navigation bridge not connected yet."
            )

        if mission_type == "DESCRIBE_SCENE":
            return (
                "Behavior: Scene description requested. "
                "Using current robot context."
            )

        if mission_type == "STOP":
            return "Behavior: Robot stopped."

        if status == "CANCELLED":
            return "Behavior: Active mission cancelled. Robot should stop."

        return "Behavior: No simulated behavior available."
