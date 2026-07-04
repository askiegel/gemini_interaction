from datetime import datetime

from vision_context import get_vision_context


def get_robot_context():
    vision = get_vision_context()

    detected_objects = vision.get("detections", [])

    if not detected_objects:
        detected_objects = [
            {
                "label": "person",
                "confidence": 0.92,
                "distance_m": 1.4,
                "position": "center",
                "tracking": False,
            },
            {
                "label": "backpack",
                "confidence": 0.87,
                "distance_m": 1.8,
                "position": "left",
                "tracking": False,
            },
        ]

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "robot_name": "Mini Pupper 2",
        "current_mission": "IDLE",
        "navigation_state": "STANDBY",
        "tracked_target": None,
        "vision": vision,
        "detected_objects": detected_objects,
        "battery_percent": 84,
        "safety": {
            "front_clear": True,
            "nearest_obstacle_m": 1.2,
        },
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

    vision = context.get("vision", {})
    lines.append("Vision:")
    lines.append(f"- Source: {vision.get('source')}")
    lines.append(f"- Online: {vision.get('online')}")
    lines.append(f"- URL: {vision.get('url')}")

    if not vision.get("online"):
        lines.append(f"- Error: {vision.get('error')}")
    lines.append("")

    lines.append("Detected Objects:")

    for obj in context["detected_objects"]:
        lines.append(
            f"- {obj.get('label')} "
            f"confidence={obj.get('confidence')} "
            f"distance={obj.get('distance_m')}m "
            f"position={obj.get('position')} "
            f"tracking={obj.get('tracking')}"
        )

    return "\n".join(lines)
