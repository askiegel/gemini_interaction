from datetime import datetime


def get_robot_context():
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "robot_name": "Mini Pupper 2",
        "current_mission": "IDLE",
        "navigation_state": "STANDBY",
        "tracked_target": None,
        "detected_objects": [
            {
                "label": "person",
                "confidence": 0.92,
                "distance_m": 1.4,
                "position": "center"
            },
            {
                "label": "backpack",
                "confidence": 0.87,
                "distance_m": 1.8,
                "position": "left"
            }
        ],
        "battery_percent": 84,
        "safety": {
            "front_clear": True,
            "nearest_obstacle_m": 1.2
        }
    }


def format_robot_context(context):
    lines = []

    lines.append("Robot Context")
    lines.append("-------------")
    lines.append(f"Robot: {context['robot_name']}")
    lines.append(f"Mission: {context['current_mission']}")
    lines.append(f"Navigation: {context['navigation_state']}")
    lines.append(f"Tracked Target: {context['tracked_target']}")
    lines.append(f"Battery: {context['battery_percent']}%")
    lines.append(f"Front Clear: {context['safety']['front_clear']}")
    lines.append(f"Nearest Obstacle: {context['safety']['nearest_obstacle_m']} m")
    lines.append("")
    lines.append("Detected Objects:")

    for obj in context["detected_objects"]:
        lines.append(
            f"- {obj['label']} "
            f"confidence={obj['confidence']} "
            f"distance={obj['distance_m']}m "
            f"position={obj['position']}"
        )

    return "\n".join(lines)
