from vision_context import get_vision_context
from world_model import WorldModel


_WORLD_MODEL = WorldModel()


def get_world_model():
    return _WORLD_MODEL


def get_robot_context():
    vision = get_vision_context()
    detections = vision.get("detections", [])

    if not detections:
        detections = [
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

    _WORLD_MODEL.update_from_detections(detections, source=vision.get("source", "mock"))

    snapshot = _WORLD_MODEL.snapshot()
    snapshot["vision"] = vision

    return snapshot


def format_robot_context(context):
    lines = []

    lines.append("Robot Context")
    lines.append("-------------")

    robot = context["robot"]
    lines.append(f"Robot: {robot.get('name')}")
    lines.append(f"Mission: {robot.get('mission')}")
    lines.append(f"Navigation: {robot.get('navigation_state')}")
    lines.append(f"Battery: {robot.get('battery_percent')}%")
    lines.append(f"Front Clear: {robot.get('front_clear')}")
    lines.append(f"Nearest Obstacle: {robot.get('nearest_obstacle_m')} m")
    lines.append("")

    vision = context.get("vision", {})
    lines.append("Vision:")
    lines.append(f"- Source: {vision.get('source')}")
    lines.append(f"- Online: {vision.get('online')}")
    lines.append(f"- URL: {vision.get('url')}")

    if not vision.get("online"):
        lines.append(f"- Error: {vision.get('error')}")
    lines.append("")

    lines.append("World Entities:")

    for entity in context["entities"]:
        lines.append(
            f"- {entity.get('label')} "
            f"type={entity.get('entity_type')} "
            f"confidence={entity.get('confidence')} "
            f"distance={entity.get('distance_m')}m "
            f"position={entity.get('position')} "
            f"tracking={entity.get('tracking')}"
        )

    return "\n".join(lines)
