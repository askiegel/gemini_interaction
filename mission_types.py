from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from uuid import uuid4


MISSION_ACTIVE = "ACTIVE"
MISSION_COMPLETED = "COMPLETED"
MISSION_CANCELLED = "CANCELLED"
MISSION_REJECTED = "REJECTED"
MISSION_INFO_ONLY = "INFO_ONLY"


@dataclass
class Mission:
    mission_id: str
    mission_type: str
    status: str
    target: Optional[str]
    speech: str
    created_at: str
    priority: int = 5

    def to_dict(self):
        return asdict(self)


def create_mission(mission_type, target=None, speech="", status=MISSION_ACTIVE, priority=5):
    return Mission(
        mission_id=f"mission-{uuid4().hex[:8]}",
        mission_type=mission_type,
        status=status,
        target=target,
        speech=speech,
        created_at=datetime.now().isoformat(timespec="seconds"),
        priority=priority,
    )
