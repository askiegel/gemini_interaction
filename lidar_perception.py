"""Background GET-only LiDAR acquisition; no motion authorization or actions.

The owner starts/stops this component explicitly. Import/construction performs
no network I/O. Readers must know the current producer session independently;
copying a session ID out of an old persisted snapshot does not establish trust.
"""

import copy
import threading
import time
import uuid

from robot_bridge.client import RobotBridgeClient
from voice_relay.lidar_sectors import finite_number, lidar_sector_payload


ACQUISITION_INTERVAL_SECONDS = 0.08
MAXIMUM_EFFECTIVE_AGE_SECONDS = 0.30
REQUEST_TIMEOUT_SECONDS = 0.25


def unavailable_state(reason, session=None, sequence=0):
    payload = lidar_sector_payload({})
    return {
        "available": False, "valid": False, "reason": reason,
        "sectors": payload["sectors"],
        "classification_thresholds": payload["classification_thresholds"],
        "source": payload["source"],
        "read_only": True, "distance_reference": "sensor_origin",
        "producer_session": session, "acquisition_sequence": sequence,
        "request_latency_seconds": None,
        "acquisition_started_at": None, "completed_at": None,
        "received_monotonic_seconds": None,
        "age_at_receipt_seconds": None, "effective_age_seconds": None,
        "maximum_effective_age_seconds": MAXIMUM_EFFECTIVE_AGE_SECONDS,
        "freshness_threshold_provisional": True,
        "freshness_note": "Measurement-based; not a physical stopping guarantee.",
    }


def read_lidar_state(state, *, expected_session, now=None):
    """Return a copy with freshness evaluated, using the host monotonic clock.

    Exactly 0.30 seconds is fresh; greater ages are stale. A missing/mismatched
    owner session never trusts persisted state, including across host restarts.
    """
    result = copy.deepcopy(state) if isinstance(state, dict) else unavailable_state("missing")
    reason = None
    if not expected_session or result.get("producer_session") != expected_session:
        reason = "producer_session_mismatch"
    elif result.get("valid") is not True or result.get("available") is not True:
        reason = result.get("reason") or "unavailable"
    else:
        now = time.monotonic() if now is None else now
        received = result.get("received_monotonic_seconds")
        age = result.get("age_at_receipt_seconds")
        if (not all(finite_number(x) for x in (now, received, age))
                or age < 0 or now < received):
            reason = "invalid_freshness"
        else:
            effective = age + (now - received)
            result["effective_age_seconds"] = effective
            if effective > MAXIMUM_EFFECTIVE_AGE_SECONDS:
                reason = "stale"
    if reason:
        result.update(available=False, valid=False, reason=reason)
    return result


class LidarPerceptionWorker:
    """One explicit producer, with injectable acquisition and clocks for tests."""

    def __init__(self, world_model, *, fetch=None, base_url=None,
                 monotonic=time.monotonic, wall_clock=time.time):
        self.world_model = world_model
        self._fetch = fetch
        self._base_url = base_url
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._publication_lock = threading.RLock()
        self._acquisition_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self._stamp = None
        self._receipt = None
        self._age = None
        self.last_error = None
        self.world_model.publish_lidar_obstacles(unavailable_state("starting", self.session))

    def _get_telemetry(self):
        if self._fetch is not None:
            return self._fetch()
        # Reuse the existing transport, with a fixed GET path and bounded timeout.
        client = RobotBridgeClient(base_url=self._base_url, timeout=REQUEST_TIMEOUT_SECONDS)
        return client._request("GET", "/telemetry/lidar")

    def run_once(self):
        """Acquire outside World Model locks and publish one complete snapshot."""
        with self._acquisition_lock:
            if self._stop.is_set():
                return unavailable_state("stopped", self.session, self.sequence)
            self.sequence += 1
            state = unavailable_state("acquisition_error", self.session, self.sequence)
            started = self._monotonic()
            state["acquisition_started_at"] = self._wall_clock()
            source = {}
            stamp = None
            previous_stamp = self._stamp
            try:
                raw = self._get_telemetry()
                received = self._monotonic()
                latency = received - started
                payload = lidar_sector_payload(raw)
                source = payload["source"]
                age = source.get("age_seconds")
                stamp = source.get("stamp_seconds")
                if not payload["ok"]:
                    raise ValueError(payload.get("error", "invalid_payload"))
                if not all(finite_number(x) and x >= 0 for x in (age, stamp, latency)):
                    raise ValueError("invalid_source_timing")
                if self._stamp is not None and stamp < self._stamp:
                    raise ValueError("scan_stamp_regressed")
                conservative_age = age + latency
                if stamp == self._stamp:
                    # Never rejuvenate a frozen scan, even if the source age resets.
                    conservative_age = max(
                        conservative_age, self._age + (received - self._receipt)
                    )
                self._stamp, self._receipt, self._age = stamp, received, conservative_age
                state.update(
                    available=True, valid=True, reason="fresh",
                    sectors=payload["sectors"], source=source,
                    request_latency_seconds=latency,
                    received_monotonic_seconds=received,
                    age_at_receipt_seconds=conservative_age,
                )
                state = read_lidar_state(state, expected_session=self.session, now=received)
            except Exception as error:
                state.update(available=False, valid=False, reason=str(error))
                if str(error) == "scan_stamp_regressed":
                    received_stamp = stamp if finite_number(stamp) else None
                    previous = (
                        previous_stamp
                        if finite_number(previous_stamp)
                        else None
                    )
                    state.update(
                        scan_stamp_previous_seconds=previous,
                        scan_stamp_received_seconds=received_stamp,
                        scan_stamp_delta_seconds=(
                            received_stamp - previous
                            if received_stamp is not None and previous is not None
                            else None
                        ),
                        source_received_at=(
                            source.get("received_at")
                            if isinstance(source, dict)
                            else None
                        ),
                        source_frame_id=(
                            source.get("frame_id")
                            if isinstance(source, dict)
                            else None
                        ),
                    )
            state["completed_at"] = self._wall_clock()
            with self._publication_lock:
                if self._stop.is_set():
                    return unavailable_state("stopped", self.session, self.sequence)
                self.world_model.publish_lidar_obstacles(state)
            return state

    def _run(self):
        try:
            while not self._stop.is_set():
                started = self._monotonic()
                self.run_once()
                self._stop.wait(max(0, ACQUISITION_INTERVAL_SECONDS - (self._monotonic() - started)))
        except Exception as error:
            # Persistence failures cannot be hidden; the last snapshot expires.
            self.last_error = str(error)
        finally:
            self.stop()

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def start(self):
        with self._publication_lock:
            if self._stop.is_set():
                raise RuntimeError("Stopped workers cannot restart; create a new producer session.")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="lidar-perception", daemon=True)
                self._thread.start()

    def stop(self):
        # In-flight HTTP completion cannot overwrite this invalidation.
        with self._publication_lock:
            self._stop.set()
            try:
                self.world_model.publish_lidar_obstacles(
                    unavailable_state("stopped", self.session, self.sequence)
                )
            except Exception as error:
                self.last_error = str(error)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=REQUEST_TIMEOUT_SECONDS + ACQUISITION_INTERVAL_SECONDS)
