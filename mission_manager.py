from mission_types import (
    create_mission,
    now,
    MISSION_ACTIVE,
    MISSION_QUEUED,
    MISSION_COMPLETED,
    MISSION_CANCELLED,
    MISSION_INFO_ONLY,
    MISSION_REJECTED,
)

from event_types import (
    EVENT_MISSION_COMPLETE,
    EVENT_TARGET_FOUND,
    EVENT_LOW_BATTERY,
    EVENT_OBSTACLE_DETECTED,
)


class MissionManager:
    def __init__(self):
        self.active_mission = None
        self.mission_queue = []
        self.mission_history = []

    def handle_intent(self, intent_json):
        intent = intent_json.get("intent", "UNKNOWN")
        speech = intent_json.get("speech", "")
        target = intent_json.get("target", None)

        if intent == "STOP":
            return self.cancel_active_mission(speech)

        if intent == "FOLLOW_PERSON":
            mission = create_mission(
                mission_type="FOLLOW_PERSON",
                target=target or "person",
                speech=speech,
                status=MISSION_ACTIVE,
                priority=7,
            )
            return self.submit_mission(mission)

        if intent == "FIND_OBJECT":
            mission = create_mission(
                mission_type="FIND_OBJECT",
                target=target,
                speech=speech,
                status=MISSION_ACTIVE,
                priority=6,
            )
            return self.submit_mission(mission)

        if intent == "RETURN_HOME":
            mission = create_mission(
                mission_type="RETURN_HOME",
                target="home",
                speech=speech,
                status=MISSION_ACTIVE,
                priority=8,
            )
            return self.submit_mission(mission)

        if intent == "DESCRIBE_SCENE":
            mission = create_mission(
                mission_type="DESCRIBE_SCENE",
                target=None,
                speech=speech,
                status=MISSION_INFO_ONLY,
                priority=3,
            )
            self.mission_history.append(mission)
            return mission

        mission = create_mission(
            mission_type="UNKNOWN",
            target=None,
            speech=speech or "I could not safely understand that command.",
            status=MISSION_REJECTED,
            priority=0,
        )
        self.mission_history.append(mission)
        return mission

    def handle_event(self, event):
        event_type = event.event_type
        payload = event.payload

        if event_type == EVENT_MISSION_COMPLETE:
            return self.complete_active_mission()

        if event_type == EVENT_TARGET_FOUND:
            return self._handle_target_found(payload)

        if event_type == EVENT_LOW_BATTERY:
            return self._handle_low_battery(payload)

        if event_type == EVENT_OBSTACLE_DETECTED:
            return self._handle_obstacle_detected(payload)

        return None

    def _handle_target_found(self, payload):
        found_target = str(payload.get("target", "")).lower()

        if not self.active_mission:
            return None

        active_target = str(self.active_mission.target).lower()

        if (
            self.active_mission.mission_type == "FIND_OBJECT"
            and found_target == active_target
        ):
            return self.complete_active_mission()

        return None

    def _handle_low_battery(self, payload):
        battery_percent = payload.get("battery_percent")

        if battery_percent is None or battery_percent > 15:
            return None

        mission = create_mission(
            mission_type="RETURN_HOME",
            target="home",
            speech=f"Battery is low at {battery_percent} percent. Returning home.",
            status=MISSION_ACTIVE,
            priority=10,
        )

        if self.active_mission:
            self.active_mission.status = MISSION_QUEUED
            self.active_mission.started_at = None
            self.mission_queue.insert(0, self.active_mission)
            self.active_mission = None

        return self.submit_mission(mission)

    def _handle_obstacle_detected(self, payload):
        if not self.active_mission:
            return None

        self.active_mission.speech = "Obstacle detected. Mission is paused."
        return self.active_mission

    def submit_mission(self, mission):
        if self.active_mission is None:
            return self._activate_mission(mission)

        if self._same_mission(self.active_mission, mission):
            return self.active_mission

        mission.status = MISSION_QUEUED
        mission.started_at = None
        self.mission_queue.append(mission)
        self.mission_history.append(mission)
        return mission

    def complete_active_mission(self):
        if not self.active_mission:
            return None

        self.active_mission.status = MISSION_COMPLETED
        self.active_mission.completed_at = now()
        completed = self.active_mission
        self.active_mission = None
        self.start_next_mission()
        return completed

    def cancel_active_mission(self, speech=""):
        if self.active_mission:
            self.active_mission.status = MISSION_CANCELLED
            self.active_mission.completed_at = now()
            self.active_mission.speech = speech or self.active_mission.speech
            cancelled = self.active_mission
            self.active_mission = None
            return cancelled

        mission = create_mission(
            mission_type="STOP",
            target=None,
            speech=speech or "Robot is already stopped.",
            status=MISSION_CANCELLED,
            priority=10,
        )
        self.mission_history.append(mission)
        return mission

    def start_next_mission(self):
        if self.active_mission is not None:
            return self.active_mission

        if not self.mission_queue:
            return None

        self.mission_queue.sort(key=lambda m: m.priority, reverse=True)
        next_mission = self.mission_queue.pop(0)
        return self._activate_mission(next_mission)

    def _activate_mission(self, mission):
        mission.status = MISSION_ACTIVE
        mission.started_at = now()
        self.active_mission = mission

        if mission not in self.mission_history:
            self.mission_history.append(mission)

        return mission

    def _same_mission(self, a, b):
        return (
            a.mission_type == b.mission_type
            and str(a.target).lower() == str(b.target).lower()
            and a.status == MISSION_ACTIVE
        )

    def get_active_mission(self):
        return self.active_mission

    def get_queue(self):
        return [m.to_dict() for m in self.mission_queue]

    def get_history(self):
        return [m.to_dict() for m in self.mission_history]

    def get_state(self):
        return {
            "active_mission": self.active_mission.to_dict() if self.active_mission else None,
            "queue": self.get_queue(),
            "history_count": len(self.mission_history),
        }
