import os
from dataclasses import dataclass


@dataclass
class WorldSettings:
    storage_path: str = os.getenv("WORLD_MODEL_FILE", "world_model_state.json")


@dataclass
class VisionSettings:
    url: str = os.getenv(
        "VISION_SERVER_URL",
        "http://localhost:8000/detections/latest"
    )
    image_path: str = os.getenv("VISION_IMAGE_PATH", "")
    poll_interval: float = float(os.getenv("VISION_POLL_INTERVAL", "1.0"))


@dataclass
class RobotSettings:
    name: str = os.getenv("ROBOT_NAME", "mini-pupper-2")
    robot_id: str = os.getenv("ROBOT_ID", "mini-pupper-001")
    platform: str = os.getenv("ROBOT_PLATFORM", "Mini Pupper 2")


@dataclass
class ProviderSettings:
    default_provider: str = os.getenv("AI_PROVIDER", "gemini")


@dataclass
class Settings:
    world: WorldSettings
    vision: VisionSettings
    robot: RobotSettings
    provider: ProviderSettings


settings = Settings(
    world=WorldSettings(),
    vision=VisionSettings(),
    robot=RobotSettings(),
    provider=ProviderSettings(),
)
