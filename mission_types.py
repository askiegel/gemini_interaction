from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from uuid import uuid4


MISSION_ACTIVE = "ACTIVE"
MISSION_QUEUED = "QUEUED"
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
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    priority: int = 5
    source: str = "cognitive"
    marvin_one_step_test: bool = False
    marvin_centering_test: bool = False

    def to_dict(self):
        return asdict(self)


def now():
    return datetime.now().isoformat(timespec="seconds")


def create_mission(
    mission_type,
    target=None,
    speech="",
    status=MISSION_ACTIVE,
    priority=5,
    source="cognitive",
    marvin_one_step_test=False,
    marvin_centering_test=False,
):
    started_at = now() if status == MISSION_ACTIVE else None

    return Mission(
        mission_id=f"mission-{uuid4().hex[:8]}",
        mission_type=mission_type,
        status=status,
        target=target,
        speech=speech,
        created_at=now(),
        started_at=started_at,
        completed_at=None,
        priority=priority,
        source=source,
        marvin_one_step_test=marvin_one_step_test,
        marvin_centering_test=marvin_centering_test,
    )
