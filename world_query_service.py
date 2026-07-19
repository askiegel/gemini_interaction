#!/usr/bin/env python3

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


SUPPORTED_WORLD_QUERY_TYPES = {
    "LATEST_ENTITY",
    "LIST_ENTITIES",
    "CURRENT_MISSION",
    "VISION_STATUS",
}


@dataclass(frozen=True)
class WorldQueryResult:
    """
    Deterministic result from a read-only World Model query.
    """

    ok: bool
    query_type: str
    reply: str
    target: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorldQueryError(RuntimeError):
    """Raised when a World Model query is invalid or cannot be processed."""


class WorldQueryService:
    """
    Read-only conversational query interface for the World Model.

    This service:

    - reads World Model state,
    - produces deterministic spoken answers,
    - never creates missions,
    - never controls the robot,
    - never writes to the World Model,
    - never calls Gemini.
    """

    def __init__(self, world_model):
        if world_model is None:
            raise ValueError("WorldQueryService requires a World Model.")

        self.world_model = world_model

    def execute_text(
        self,
        user_text: str,
    ) -> WorldQueryResult:
        """
        Convert a supported natural-language question into a deterministic
        read-only World Model query.

        Gemini decides that a request is a WORLD_QUERY. This method decides
        which supported World Model operation should answer it.
        """
        if not isinstance(user_text, str):
            raise WorldQueryError(
                "World query text must be a string."
            )

        normalized = " ".join(
            user_text.strip().lower().split()
        )

        if not normalized:
            raise WorldQueryError(
                "World query text cannot be empty."
            )

        if self._asks_for_entity_list(normalized):
            return self.execute("LIST_ENTITIES")

        if self._asks_about_vision(normalized):
            return self.execute("VISION_STATUS")

        if self._asks_about_mission(normalized):
            return self.execute("CURRENT_MISSION")

        target = self._extract_entity_target(normalized)

        if target:
            return self.execute(
                "LATEST_ENTITY",
                target=target,
            )

        raise WorldQueryError(
            "I could not determine which World Model "
            "information the question requested."
        )

    def execute(
        self,
        query_type: str,
        target: Optional[str] = None,
    ) -> WorldQueryResult:
        normalized_query = self._normalize_query_type(query_type)
        normalized_target = self._normalize_target(target)

        self._refresh_world_model()

        if normalized_query == "LATEST_ENTITY":
            return self._query_latest_entity(normalized_target)

        if normalized_query == "LIST_ENTITIES":
            return self._query_list_entities()

        if normalized_query == "CURRENT_MISSION":
            return self._query_current_mission()

        if normalized_query == "VISION_STATUS":
            return self._query_vision_status()

        raise WorldQueryError(
            f"Unsupported world query type: {normalized_query}"
        )

    def _query_latest_entity(
        self,
        target: Optional[str],
    ) -> WorldQueryResult:
        if not target:
            raise WorldQueryError(
                "LATEST_ENTITY requires a target."
            )

        matching_entities = [
            entity
            for entity in self._get_entities()
            if self._normalize_label(
                entity.get("label")
            ) == self._normalize_label(target)
        ]

        if not matching_entities:
            return WorldQueryResult(
                ok=True,
                query_type="LATEST_ENTITY",
                target=target,
                reply=(
                    f"I do not have a recorded observation "
                    f"of {self._spoken_target(target)}."
                ),
                data={
                    "found": False,
                },
            )

        latest = max(
            matching_entities,
            key=lambda entity: self._timestamp_sort_value(
                entity.get("last_seen")
            ),
        )

        last_seen = latest.get("last_seen")
        age_seconds = self._timestamp_age_seconds(last_seen)
        position = self._extract_position(latest)
        tracking = bool(
            latest.get("tracking")
            or (
                latest.get("attributes") or {}
            ).get("tracking")
        )

        reply = self._build_latest_entity_reply(
            target=target,
            age_seconds=age_seconds,
            position=position,
            tracking=tracking,
        )

        return WorldQueryResult(
            ok=True,
            query_type="LATEST_ENTITY",
            target=target,
            reply=reply,
            data={
                "found": True,
                "entity": latest,
                "age_seconds": age_seconds,
                "position": position,
                "tracking": tracking,
            },
        )

    def _query_list_entities(self) -> WorldQueryResult:
        entities = self._get_entities()

        if not entities:
            return WorldQueryResult(
                ok=True,
                query_type="LIST_ENTITIES",
                reply=(
                    "I do not currently have any recorded "
                    "entities in the World Model."
                ),
                data={
                    "count": 0,
                    "labels": [],
                },
            )

        label_counts: Dict[str, int] = {}

        for entity in entities:
            label = self._normalize_label(
                entity.get("label")
            )

            if not label:
                label = "unknown object"

            label_counts[label] = label_counts.get(label, 0) + 1

        spoken_items = []

        for label in sorted(label_counts):
            count = label_counts[label]

            if count == 1:
                spoken_items.append(
                    self._with_article(label)
                )
            else:
                spoken_items.append(
                    f"{count} {self._pluralize(label)}"
                )

        reply = (
            "I currently have "
            f"{self._join_spoken_items(spoken_items)} "
            "recorded in the World Model."
        )

        return WorldQueryResult(
            ok=True,
            query_type="LIST_ENTITIES",
            reply=reply,
            data={
                "count": len(entities),
                "labels": label_counts,
            },
        )

    def _query_current_mission(self) -> WorldQueryResult:
        robot_state = self._get_robot_state()
        mission = robot_state.get("mission")

        if not mission:
            return WorldQueryResult(
                ok=True,
                query_type="CURRENT_MISSION",
                reply="I do not currently have an active mission.",
                data={
                    "mission": None,
                },
            )

        if isinstance(mission, dict):
            mission_type = (
                mission.get("mission_type")
                or mission.get("type")
                or mission.get("intent")
                or "unknown"
            )

            target = mission.get("target")
            status = mission.get("status")

            parts = [
                f"My current mission is {self._spoken_name(mission_type)}"
            ]

            if target:
                parts.append(
                    f"with target {self._spoken_target(str(target))}"
                )

            if status:
                parts.append(
                    f"and its status is {self._spoken_name(status)}"
                )

            reply = " ".join(parts) + "."
        else:
            reply = (
                "My current mission is "
                f"{self._spoken_name(str(mission))}."
            )

        return WorldQueryResult(
            ok=True,
            query_type="CURRENT_MISSION",
            reply=reply,
            data={
                "mission": mission,
            },
        )

    def _query_vision_status(self) -> WorldQueryResult:
        environment = self._get_environment()
        vision = environment.get("vision") or {}

        status = (
            vision.get("vision_status")
            or vision.get("status")
        )

        camera_running = vision.get("camera_running")
        last_error = vision.get("last_error")

        if last_error:
            reply = (
                "The vision system is reporting an error: "
                f"{last_error}."
            )
        elif status:
            reply = (
                "The vision system status is "
                f"{self._spoken_name(status)}."
            )
        elif camera_running is True:
            reply = "The camera is running."
        elif camera_running is False:
            reply = "The camera is not currently running."
        else:
            reply = (
                "I do not currently have enough information "
                "to determine the vision system status."
            )

        return WorldQueryResult(
            ok=True,
            query_type="VISION_STATUS",
            reply=reply,
            data={
                "vision": vision,
            },
        )

    def _refresh_world_model(self) -> None:
        load_method = getattr(
            self.world_model,
            "load",
            None,
        )

        if callable(load_method):
            load_method()

    def _get_entities(self) -> List[Dict[str, Any]]:
        getter = getattr(
            self.world_model,
            "get_entities",
            None,
        )

        if callable(getter):
            entities = getter()
        else:
            raw_entities = getattr(
                self.world_model,
                "entities",
                {},
            )

            if isinstance(raw_entities, dict):
                entities = list(raw_entities.values())
            else:
                entities = list(raw_entities)

        result = []

        for entity in entities or []:
            if isinstance(entity, dict):
                result.append(dict(entity))
                continue

            to_dict = getattr(entity, "to_dict", None)

            if callable(to_dict):
                result.append(to_dict())
                continue

            raise WorldQueryError(
                "World Model returned an unsupported entity type."
            )

        return result

    def _get_robot_state(self) -> Dict[str, Any]:
        value = getattr(
            self.world_model,
            "robot_state",
            {},
        )

        return dict(value or {})

    def _get_environment(self) -> Dict[str, Any]:
        value = getattr(
            self.world_model,
            "environment",
            {},
        )

        return dict(value or {})

    @staticmethod
    def _asks_about_vision(text: str) -> bool:
        phrases = (
            "vision status",
            "camera status",
            "is the camera running",
            "is your camera running",
            "is vision working",
            "is your vision working",
            "is the vision system working",
            "is your vision system working",
        )

        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _asks_about_mission(text: str) -> bool:
        phrases = (
            "current mission",
            "active mission",
            "what are you doing",
            "what is your mission",
            "what mission",
            "what are you working on",
            "are you following",
            "who are you following",
        )

        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _asks_for_entity_list(text: str) -> bool:
        phrases = (
            "what do you see",
            "what can you see",
            "what objects",
            "which objects",
            "list the objects",
            "list objects",
            "what is around you",
            "who do you see",
            "who can you see",
            "what have you seen",
        )

        return any(phrase in text for phrase in phrases)

    @classmethod
    def _extract_entity_target(
        cls,
        text: str,
    ) -> Optional[str]:
        prefixes = (
            "where did you last see ",
            "where did you see ",
            "where is ",
            "where are ",
            "when did you last see ",
            "when did you see ",
            "have you seen ",
            "did you see ",
            "can you find ",
            "do you see ",
            "are you tracking ",
            "what happened to ",
        )

        target = None

        for prefix in prefixes:
            if prefix in text:
                target = text.split(prefix, 1)[1]
                break

        if target is None:
            return None

        target = target.strip(" ?.!,")
        target = cls._remove_target_determiner(target)

        trailing_phrases = (
            " last",
            " recently",
            " right now",
            " now",
            " today",
        )

        for suffix in trailing_phrases:
            if target.endswith(suffix):
                target = target[:-len(suffix)].strip()

        aliases = {
            "me": "person",
            "someone": "person",
            "anyone": "person",
            "a person": "person",
            "the person": "person",
            "my backpack": "backpack",
            "your backpack": "backpack",
            "my bag": "backpack",
            "the backpack": "backpack",
            "the bag": "backpack",
        }

        target = aliases.get(target, target)

        return target or None

    @staticmethod
    def _remove_target_determiner(target: str) -> str:
        prefixes = (
            "my ",
            "your ",
            "the ",
            "a ",
            "an ",
        )

        for prefix in prefixes:
            if target.startswith(prefix):
                return target[len(prefix):].strip()

        return target

    @staticmethod
    def _normalize_query_type(value: Any) -> str:
        if not isinstance(value, str):
            raise WorldQueryError(
                "World query type must be a string."
            )

        normalized = value.strip().upper()

        if not normalized:
            raise WorldQueryError(
                "World query type cannot be empty."
            )

        if normalized not in SUPPORTED_WORLD_QUERY_TYPES:
            raise WorldQueryError(
                f"Unsupported world query type: {normalized}"
            )

        return normalized

    @staticmethod
    def _normalize_target(value: Any) -> Optional[str]:
        if value is None:
            return None

        if not isinstance(value, str):
            raise WorldQueryError(
                "World query target must be a string or null."
            )

        normalized = " ".join(value.strip().split())

        return normalized or None

    @staticmethod
    def _normalize_label(value: Any) -> str:
        label = str(value or "").strip().lower()
        label = label.replace("_", " ").replace("-", " ")
        label = " ".join(label.split())

        aliases = {
            "back pack": "backpack",
            "rucksack": "backpack",
            "human": "person",
        }

        return aliases.get(label, label)

    @staticmethod
    def _timestamp_sort_value(value: Any) -> float:
        if not value:
            return float("-inf")

        try:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )

            return parsed.timestamp()
        except (TypeError, ValueError):
            return float("-inf")

    @staticmethod
    def _timestamp_age_seconds(
        value: Any,
    ) -> Optional[float]:
        if not value:
            return None

        try:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )

            current = datetime.now(parsed.tzinfo)

            return max(
                0.0,
                (current - parsed).total_seconds(),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_position(
        entity: Dict[str, Any],
    ) -> Optional[str]:
        direct_position = entity.get("position")

        if direct_position:
            return str(direct_position).strip().lower()

        attributes = entity.get("attributes") or {}
        attribute_position = attributes.get("position")

        if attribute_position:
            return str(attribute_position).strip().lower()

        history = entity.get("history") or []

        if history:
            latest_observation = history[-1] or {}
            location = latest_observation.get("location") or {}

            if location.get("position"):
                return str(
                    location["position"]
                ).strip().lower()

        return None

    def _build_latest_entity_reply(
        self,
        target: str,
        age_seconds: Optional[float],
        position: Optional[str],
        tracking: bool,
    ) -> str:
        spoken_target = self._spoken_target(target)

        if tracking:
            reply = (
                f"I am currently tracking {spoken_target}"
            )
        elif age_seconds is None:
            reply = (
                f"I have a recorded observation of {spoken_target}"
            )
        elif age_seconds < 2:
            reply = (
                f"I can currently see {spoken_target}"
            )
        else:
            reply = (
                f"I last saw {spoken_target} "
                f"{self._format_age(age_seconds)} ago"
            )

        if position:
            reply += (
                f" on the {self._spoken_name(position)}"
            )

        return reply + "."

    @staticmethod
    def _format_age(age_seconds: float) -> str:
        rounded_seconds = int(round(age_seconds))

        if rounded_seconds < 60:
            unit = (
                "second"
                if rounded_seconds == 1
                else "seconds"
            )

            return f"{rounded_seconds} {unit}"

        rounded_minutes = int(round(age_seconds / 60.0))

        if rounded_minutes < 60:
            unit = (
                "minute"
                if rounded_minutes == 1
                else "minutes"
            )

            return f"{rounded_minutes} {unit}"

        rounded_hours = int(round(age_seconds / 3600.0))
        unit = "hour" if rounded_hours == 1 else "hours"

        return f"{rounded_hours} {unit}"

    @staticmethod
    def _spoken_name(value: Any) -> str:
        return str(value).strip().lower().replace("_", " ")

    @staticmethod
    def _spoken_target(value: str) -> str:
        target = value.strip().lower()

        possessive_prefixes = (
            "my ",
            "your ",
            "the ",
            "a ",
            "an ",
        )

        if target.startswith(possessive_prefixes):
            return target

        if target == "person":
            return "a person"

        return f"the {target}"

    @staticmethod
    def _with_article(label: str) -> str:
        article = (
            "an"
            if label[:1].lower() in "aeiou"
            else "a"
        )

        return f"{article} {label}"

    @staticmethod
    def _pluralize(label: str) -> str:
        if label.endswith("s"):
            return label

        if label.endswith("y") and len(label) > 1:
            return label[:-1] + "ies"

        return label + "s"

    @staticmethod
    def _join_spoken_items(items: List[str]) -> str:
        if not items:
            return "nothing"

        if len(items) == 1:
            return items[0]

        if len(items) == 2:
            return f"{items[0]} and {items[1]}"

        return ", ".join(items[:-1]) + f", and {items[-1]}"
