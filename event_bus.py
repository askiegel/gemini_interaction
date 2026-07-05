from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4


@dataclass
class Event:
    event_id: str
    event_type: str
    payload: Dict[str, Any]
    created_at: str
    source: str = "cognitive"

    def to_dict(self):
        return asdict(self)


class EventBus:
    def __init__(self):
        self.events = []

    def publish(self, event_type, payload=None, source="cognitive"):
        event = Event(
            event_id=f"event-{uuid4().hex[:8]}",
            event_type=event_type,
            payload=payload or {},
            created_at=datetime.now().isoformat(timespec="seconds"),
            source=source,
        )
        self.events.append(event)
        return event

    def history(self):
        return [event.to_dict() for event in self.events]

    def recent(self, limit=10):
        return [event.to_dict() for event in self.events[-limit:]]
