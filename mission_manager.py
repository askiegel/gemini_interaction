from mission_types import (
    create_mission,
    MISSION_ACTIVE,
    MISSION_CANCELLED,
    MISSION_INFO_ONLY,
    MISSION_REJECTED,
)


class MissionManager:
    def __init__(self):
        self.active_mission = None
        self.mission_history = []

    def handle_intent(self, intent_json):
        intent = intent_json.get("intent", "UNKNOWN")
        speech = intent_json.get("speech", "")
        target = intent_json.get("target", None)

        if intent == "STOP":
            return self._stop_current_mission(speech)

        if intent == "FOLLOW_PERSON":
            mission = create_mission(
                mission_type="FOLLOW_PERSON",
                target=target or "person",
                speech=speech,
                status=MISSION_ACTIVE,
                priority=7,
            )
            return self._activate_mission(mission)

        if intent == "FIND_OBJECT":
            mission = create_mission(
                mission_type="FIND_OBJECT",
                target=target,
                speech=speech,
                status=MISSION_ACTIVE,
                priority=6,
            )
            return self._activate_mission(mission)

        if intent == "RETURN_HOME":
            mission = create_mission(
                mission_type="RETURN_HOME",
                target="home",
                speech=speech,
                status=MISSION_ACTIVE,
                priority=8,
            )
            return self._activate_mission(mission)

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

    def _activate_mission(self, mission):
        self.active_mission = mission
        self.mission_history.append(mission)
        return mission

    def _stop_current_mission(self, speech):
        if self.active_mission:
            self.active_mission.status = MISSION_CANCELLED
            stopped = self.active_mission
            self.active_mission = None
            return stopped

        mission = create_mission(
            mission_type="STOP",
            target=None,
            speech=speech or "Robot is already stopped.",
            status=MISSION_CANCELLED,
            priority=10,
        )
        self.mission_history.append(mission)
        return mission

    def get_active_mission(self):
        return self.active_mission

    def get_history(self):
        return [m.to_dict() for m in self.mission_history]
