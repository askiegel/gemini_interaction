"""Bounded, read-only evidence retention. Nothing here admits physical motion.

No I/O on construction. Odometry is an existing ROS observation, never an
estimator or substitute for the camera/LiDAR authorization gates. Its deployed
publisher's independence from commanded gait velocity must be verified.

MARVIN_ODOM_DIAGNOSTICS_ENABLED=1 enables the existing /odom subscriber on
runtime startup. It does not start navigation/localization. Export the runtime
status's marvin_progress_diagnostics after completion to include late samples;
the immediate mission-result snapshot may precede the first post-STOP odometry
callback. Missing samples and unverified provenance remain explicit.
"""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from collections import deque


def camera_metadata(observation):
    observation = observation or {}
    tracker = observation.get("opencv_tracker") or {}
    box = tracker.get("bbox") or observation.get("bbox")
    center = None
    if isinstance(box, dict) and all(type(box.get(k)) in (int, float) for k in ("x1", "x2", "y1", "y2")):
        center = {"x": (box["x1"] + box["x2"]) / 2, "y": (box["y1"] + box["y2"]) / 2}
    return copy.deepcopy({
        "source_frame_stamp_ns": observation.get("source_frame_stamp_ns"),
        "received_monotonic_seconds": observation.get("received_monotonic_seconds"),
        "bbox": box, "center": center, "tracker_quality": tracker.get("quality"),
        "identity_source": observation.get("identity_source"),
        "image_width": tracker.get("image_width") or observation.get("image_width"),
        "image_height": tracker.get("image_height") or observation.get("image_height"),
    })


class MarvinProgressDiagnostics:
    MAX_ACTIONS = 256
    MAX_ODOM_SAMPLES = 2048
    MAX_LIDAR_SAMPLES = 32

    def __init__(self):
        self.lock = threading.RLock()
        self.odom = deque(maxlen=self.MAX_ODOM_SAMPLES)
        self.lidar = deque(maxlen=self.MAX_LIDAR_SAMPLES)
        self.report = None
        self.source_status = {"topic": "/odom", "available": False,
                              "independent_translation_verified": False,
                              "reason": "no_odometry_received",
                              "provenance_note": "Local Stanford odom/raw uses commanded gait velocities; deployed publisher must be checked."}
        self._process = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        """Start only a read-only subscriber, off the control thread."""
        if os.getenv("MARVIN_ODOM_DIAGNOSTICS_ENABLED") != "1":
            with self.lock:
                self.source_status["reason"] = "odometry_subscriber_not_enabled"
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._read_ros, name="marvin-odom-diagnostics", daemon=True)
            self._thread.start()

    def _read_ros(self):
        helper = Path(__file__).parent / "voice_relay" / "marvin_odom_diagnostics.py"
        env = os.environ.copy()
        for key in ("ZENOH_CONFIG_OVERRIDE", "CYCLONEDDS_URI", "ROS_DISCOVERY_SERVER", "ROS_SUPER_CLIENT",
                    "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE", "FASTDDS_BUILTIN_TRANSPORTS"):
            env.pop(key, None)
        env.update(ROS_DOMAIN_ID="42", ROS_LOCALHOST_ONLY="0", RMW_IMPLEMENTATION="rmw_fastrtps_cpp")
        try:
            # Fixed ROS setup; the path is passed as an argument, never shell code.
            self._process = subprocess.Popen(
                ["/bin/bash", "-c", 'source /opt/ros/humble/setup.bash; exec /usr/bin/python3 -u "$1"',
                 "marvin-odom", str(helper)], env=env, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
            )
            if self._stop.is_set():
                self._process.terminate()
            for line in self._process.stdout:
                try:
                    self.record_odom(json.loads(line))
                except (ValueError, TypeError):
                    continue
            self._process.wait()
            with self.lock:
                self.source_status.update(available=False, reason="odometry_subscriber_exited")
        except Exception as exc:
            with self.lock:
                self.source_status.update(available=False, reason="odometry_subscriber_unavailable", error=str(exc))

    def stop(self):
        self._stop.set()
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass

    def begin(self, mission_id, *, expected_session=None, camera_model=None):
        with self.lock:
            self.owner_thread = threading.get_ident()
            self.lidar.clear()
            self.expected_session = expected_session
            self.camera_model = copy.deepcopy(camera_model)
            self.report = {"diagnostic_only": True, "mission_id": mission_id,
                           "actions": [], "actions_dropped": 0, "terminal": None,
                           "last_observation": None, "last_target_association": None,
                           "last_camera": None, "last_lidar": None,
                           "jpeg_retention": "metadata_only_no_existing_async_jpeg_writer"}

    def record_odom(self, sample):
        """Cache typed ROS samples, including their actual callback receipt time."""
        keys = ("x", "y", "yaw", "linear_x", "linear_y", "angular_z", "received_monotonic_seconds")
        if (not isinstance(sample, dict) or type(sample.get("stamp_ns")) is not int
                or sample["stamp_ns"] < 0 or not sample.get("frame_id") or not sample.get("child_frame_id")
                or not all(type(sample.get(k)) in (int, float) and math.isfinite(sample[k]) for k in keys)):
            return
        sample = copy.deepcopy(sample)
        sample["stamp_ns_decimal"] = str(sample["stamp_ns"])
        with self.lock:
            if self.odom and sample["received_monotonic_seconds"] < self.odom[-1]["received_monotonic_seconds"]:
                return
            self.odom.append(sample)
            self.source_status.update(available=True, reason="received_existing_ros_odometry",
                                      publishers=sample.get("publishers"), frame_id=sample["frame_id"])
            self._resolve_odom()
            if (self.report and self.report["terminal"] is not None
                    and sample["received_monotonic_seconds"] - self.report["terminal"]["ended_monotonic_seconds"] <= 1.0):
                self.report["terminal"]["last_odom"] = copy.deepcopy(sample)

    def prepare_action(self, kind, observation):
        with self.lock:
            rows = self.report["actions"]
            if len(rows) >= self.MAX_ACTIONS:
                rows.pop(0)
                self.report["actions_dropped"] += 1
            rows.append({"action_number": len(rows) + self.report["actions_dropped"] + 1,
                         "type": {"ADVANCING": "forward", "ALIGNING": "alignment", "SEARCHING": "search_turn"}.get(kind, kind),
                         "state": kind, "authorizing_camera": camera_metadata(observation),
                         "pre_action_target_association": copy.deepcopy(observation.get("arrival")),
                         "authorizing_semantic_frame": copy.deepcopy(self.report.get("last_semantic_frame")),
                         "command": None, "pre_odom": None, "post_odom": None,
                         "first_post_action_lidar": None, "next_target_association": None,
                         "tracker_refreshes": [], "semantic_reacquisitions": [],
                         "reacquisition_used": False, "motion_executed": False})

    def command_event(self, phase, metadata):
        with self.lock:
            if (self.report is None or self.report["terminal"] is not None or not self.report["actions"]
                    or threading.get_ident() != self.owner_thread):
                return
            row = self.report["actions"][-1]
            if phase == "start":
                now = metadata["start_monotonic_seconds"]
                if len(self.report["actions"]) > 1:
                    self.report["actions"][-2]["next_command_start_monotonic_seconds"] = now
                row["command"] = copy.deepcopy(metadata)
                row["pre_odom"] = next((copy.deepcopy(s) for s in reversed(self.odom)
                    if 0 <= now - s["received_monotonic_seconds"] <= 1.0
                    and isinstance(s.get("publishers"), list) and len(s["publishers"]) == 1
                    and type(s.get("source_age_seconds")) in (int, float)
                    and 0 <= s["source_age_seconds"] + now - s["received_monotonic_seconds"] <= 1.0), None)
            elif phase == "complete" and row["command"] is not None:
                row["command"].update(copy.deepcopy(metadata))

    def action_result(self, result, stopped_at=None):
        with self.lock:
            row = self.report["actions"][-1]
            row["motion_executed"] = result.get("motion_executed") is True
            row.update(interrupted=result.get("interrupted") is True,
                       interruption_reason=result.get("interruption_reason"),
                       source_stamp_consumed=result.get("source_stamp_consumed"),
                       full_step_completed=result.get("full_step_completed", row["motion_executed"]),
                       actual_confirmed_run_duration_seconds=result.get("actual_confirmed_run_duration_seconds"))
            approach = result.get("approach_result") or {}
            turn = result.get("turn_result") or result
            row["jit_target_association"] = copy.deepcopy(approach.get("target_standoff"))
            if (row["jit_target_association"] or {}).get("ok") is True:
                self.report["last_target_association"] = copy.deepcopy(row["jit_target_association"])
            row["jit_lidar_evidence"] = copy.deepcopy(
                approach.get("action_lidar_evidence") or turn.get("action_lidar_evidence"))
            command = row.get("command") or {}
            if command:
                command["requested_duration"] = result.get("requested_duration", result.get("duration", command.get("duration")))
            row["stop_events"] = copy.deepcopy(turn.get("stop_events") or [])
            forward = approach.get("forward_result") or {}
            if forward.get("bounded_forward_invalidated") is True:
                outcome = forward.get("interlock_dispatch_outcome") or {}
                row["interlock_dispatch_outcome"] = copy.deepcopy(outcome)
                row["stop_events"] += copy.deepcopy(outcome.get("stop_events") or [])
            # Bridge bounded completion includes automatic STOP. Retain the
            # later explicit READY/zero verification separately, never invent it.
            ack = command.get("bridge_acknowledgement") or {}
            automatic_stop = (ack.get("automatic_stop") is True and ack.get("returned_immediately") is False
                              and ack.get("ok") is True)
            stops = [event["completed_monotonic_seconds"] for event in row["stop_events"]
                     if type(event.get("completed_monotonic_seconds")) in (int, float)
                     and isinstance(event.get("result"), dict) and event["result"].get("ok") is True]
            if automatic_stop and command.get("completion_monotonic_seconds") is not None:
                stops.append(command["completion_monotonic_seconds"])
            if stopped_at is not None:
                row["bridge_zero_verified_monotonic_seconds"] = stopped_at
                row["stopped_monotonic_seconds"] = min(stops) if stops else stopped_at
            elif stops:
                row["stopped_monotonic_seconds"] = min(stops)
            self._resolve_odom()
            for scan in self.lidar:
                self._attach_lidar(scan)

    def record_lidar(self, scan):
        with self.lock:
            if self.report is None:
                return
            terminal = self.report["terminal"]
            if terminal is not None and time.monotonic() - terminal["ended_monotonic_seconds"] > .60:
                return
            self.lidar.append(copy.deepcopy(scan))
            if scan.get("available") is True and scan.get("valid") is True:
                self.report["last_lidar"] = copy.deepcopy({key: scan.get(key) for key in (
                    "producer_session", "acquisition_sequence", "source", "received_monotonic_seconds", "effective_age_seconds")})
                if terminal is not None:
                    terminal["last_lidar"] = copy.deepcopy(self.report["last_lidar"])
            self._attach_lidar(scan)

    def _attach_lidar(self, scan):
        if not self.report or not self.report["actions"]:
            return
        row = self.report["actions"][-1]
        evidence = row.get("jit_lidar_evidence") or row.get("jit_target_association") or {}
        stop = row.get("stopped_monotonic_seconds")
        receipt = scan.get("received_monotonic_seconds")
        sequence, baseline = scan.get("acquisition_sequence"), evidence.get("acquisition_sequence")
        age = scan.get("effective_age_seconds")
        geometry = scan.get("local_motion_geometry")
        if (row["first_post_action_lidar"] is None and stop is not None
                and type(receipt) in (int, float) and receipt > stop
                and scan.get("producer_session") == evidence.get("producer_session")
                and type(sequence) is int and type(baseline) is int and sequence > baseline
                and scan.get("available") is True and scan.get("valid") is True
                and isinstance(geometry, dict) and geometry.get("valid") is True
                and geometry.get("frame_id") == "lidar_link" and isinstance(geometry.get("points"), list)
                and type(age) in (int, float) and 0 <= age <= .30):
            row["first_post_action_lidar"] = copy.deepcopy(scan)
            row["first_post_action_lidar"]["target_associated"] = False

    def observe(self, observation, *, reacquisition=False):
        with self.lock:
            self.report["last_observation"] = copy.deepcopy(observation)
            camera = camera_metadata(observation)
            if camera.get("bbox") is not None:
                self.report["last_camera"] = camera
            association = observation.get("arrival") or {}
            # ALIGN has no control-side target range. Retain an explicitly
            # diagnostic association against already cached producer evidence;
            # never acquire another scan or feed this back into the controller.
            tracker = observation.get("opencv_tracker") or {}
            receipt = observation.get("received_monotonic_seconds")
            if (association.get("ok") is not True and self.lidar
                    and observation.get("identity_confirmed") is True and tracker.get("matched") is True
                    and type(tracker.get("quality")) in (int, float) and tracker["quality"] >= .80
                    and type(receipt) in (int, float) and 0 <= time.monotonic() - receipt <= 1.0):
                from lidar_perception import read_lidar_state
                from marvin_lidar_standoff import evaluate_marvin_lidar_standoff
                cached = read_lidar_state(self.lidar[-1], expected_session=self.expected_session)
                association = dict(evaluate_marvin_lidar_standoff(tracker, cached, self.camera_model,
                    expected_session=self.expected_session), diagnostic_only=True,
                    diagnostic_camera_source_frame_stamp_ns=observation.get("source_frame_stamp_ns"))
            if association.get("ok") is True:
                self.report["last_target_association"] = copy.deepcopy(association)
            if not self.report["actions"]:
                return
            row = self.report["actions"][-1]
            refresh = observation.get("post_action_tracker_diagnostics")
            if isinstance(refresh, dict):
                row["tracker_refreshes"].append(copy.deepcopy(refresh))
                fresh = next((f for f in refresh.get("frames", []) if f.get("camera_returned_cached_frame") is False), None)
                if fresh is not None and row.get("first_new_post_action_camera") is None:
                    row["first_new_post_action_camera"] = copy.deepcopy(fresh)
            if reacquisition:
                row["reacquisition_used"] = True
            elif (row.get("first_new_post_action_camera") is None
                  and type(observation.get("source_frame_stamp_ns")) is int
                  and observation["source_frame_stamp_ns"] > row["authorizing_camera"]["source_frame_stamp_ns"]):
                row["first_new_post_action_camera"] = camera_metadata(observation)
            baseline = (row.get("jit_target_association") or {}).get("acquisition_sequence")
            if (row["next_target_association"] is None and association.get("ok") is True
                    and type(baseline) is int and type(association.get("acquisition_sequence")) is int
                    and association["acquisition_sequence"] > baseline
                    and association.get("producer_session") == row["jit_target_association"].get("producer_session")):
                row["next_target_association"] = copy.deepcopy(association)
                row["next_associated_camera"] = camera_metadata(observation)

    def perception_event(self, phase, metadata):
        with self.lock:
            if (self.report is None or self.report["terminal"] is not None
                    or threading.get_ident() != self.owner_thread):
                return
            if phase == "semantic":
                self.report["last_semantic_frame"] = copy.deepcopy(metadata)
                if self.report["actions"]:
                    self.report["actions"][-1]["reacquisition_used"] = True
                    frames = self.report["actions"][-1]["semantic_reacquisitions"]
                    if frames and frames[-1].get("source_frame_stamp_ns") == metadata.get("source_frame_stamp_ns"):
                        frames[-1] = copy.deepcopy(metadata)
                    else:
                        frames.append(copy.deepcopy(metadata))
            elif phase == "post_action_frame" and self.report["actions"]:
                row = self.report["actions"][-1]
                metadata = dict(metadata)
                tracker = metadata.get("opencv_tracker") or {}
                metadata.update(camera_metadata(dict(metadata, opencv_tracker=tracker,
                    bbox=metadata.get("tracker_bbox") or metadata.get("tracker_candidate_bbox"),
                    identity_source="marvin_locked_tracker_continuity")))
                frames = row.setdefault("post_action_frames", [])
                frames.append(copy.deepcopy(metadata))
                del frames[:-128]
                if metadata.get("camera_returned_cached_frame") is False:
                    first = row.get("first_new_post_action_camera")
                    if first is None or first.get("source_frame_stamp_ns") == metadata.get("source_frame_stamp_ns"):
                        row["first_new_post_action_camera"] = copy.deepcopy(metadata)

    def _resolve_odom(self):
        if self.report is None:
            return
        for row in self.report["actions"]:
            pre, stop = row.get("pre_odom"), row.get("stopped_monotonic_seconds")
            if stop is None or row["post_odom"] is not None:
                continue
            end = row.get("next_command_start_monotonic_seconds", float("inf"))
            post = next((s for s in self.odom if stop < s["received_monotonic_seconds"] < end
                         and isinstance(s.get("publishers"), list) and len(s["publishers"]) == 1
                         and type(s.get("source_age_seconds")) in (int, float)
                         and 0 <= s["source_age_seconds"] <= 1.0
                         and s["received_monotonic_seconds"] - s["source_age_seconds"] > stop
                         and (pre is None or (s["stamp_ns"] > pre["stamp_ns"] and s["frame_id"] == pre["frame_id"]
                              and s["child_frame_id"] == pre["child_frame_id"]
                              and s.get("publishers") == pre.get("publishers")))), None)
            if post:
                row["post_odom"] = copy.deepcopy(post)
                if pre is not None:
                    dx, dy, yaw = post["x"] - pre["x"], post["y"] - pre["y"], post["yaw"] - pre["yaw"]
                    row["odom_delta"] = {"x": dx, "y": dy, "planar_translation_m": math.hypot(dx, dy),
                                         "yaw_radians": math.atan2(math.sin(yaw), math.cos(yaw))}

    def mission_stop(self, stop, zero, completed_at):
        with self.lock:
            self.report["mission_stop"] = copy.deepcopy({"stop_result": stop,
                "bridge_after_stop": zero, "completed_monotonic_seconds": completed_at})
            if (self.report["actions"] and isinstance(stop, dict) and stop.get("ok") is True
                    and zero.get("ok") is True and zero.get("status") == "READY"):
                row = self.report["actions"][-1]
                if row.get("command") is not None and row.get("stopped_monotonic_seconds") is None:
                    row["stopped_monotonic_seconds"] = completed_at
                    row["stop_time_source"] = "confirmed_terminal_stop"
                    self._resolve_odom()
                    for scan in self.lidar:
                        self._attach_lidar(scan)

    def terminal(self, state, reason):
        with self.lock:
            self.report["terminal"] = {"state": state, "reason": reason,
                "ended_monotonic_seconds": time.monotonic(),
                "last_observation": copy.deepcopy(self.report["last_observation"]),
                "last_target_association": copy.deepcopy(self.report["last_target_association"]),
                "last_camera": copy.deepcopy(self.report["last_camera"]),
                "last_lidar": copy.deepcopy(self.report["last_lidar"]),
                "last_odom": copy.deepcopy(self.odom[-1]) if self.odom else None}

    def odometry_snapshot(self):
        """Available before mission submission for a stationary source check."""
        with self.lock:
            result = copy.deepcopy(self.source_status)
            last = self.odom[-1] if self.odom else None
            result["last_sample"] = copy.deepcopy(last)
            age = max(0.0, time.monotonic() - last["received_monotonic_seconds"]) if last else None
            source_age = last.get("source_age_seconds") if last else None
            result["last_receipt_age_seconds"] = age
            result["current"] = bool(
                result["available"] is True and type(source_age) in (int, float)
                and type(age) in (int, float) and 0 <= source_age + age <= 1.0)
            result["single_publisher"] = bool(last and isinstance(last.get("publishers"), list)
                                              and len(last["publishers"]) == 1)
            return result

    def snapshot(self):
        # Capture mutable containers briefly, then serialize/copy outside the
        # cache lock so telemetry readers cannot hold up command callbacks.
        with self.lock:
            if self.report is None:
                return None
            result = dict(self.report)
            result["actions"] = []
            for row in self.report["actions"]:
                captured = dict(row)
                captured["command"] = dict(row["command"]) if row.get("command") else None
                for key in ("tracker_refreshes", "semantic_reacquisitions", "post_action_frames"):
                    if key in row:
                        captured[key] = list(row[key])
                result["actions"].append(captured)
            if result["terminal"] is not None:
                result["terminal"] = dict(result["terminal"])
            result["odometry_source"] = self.odometry_snapshot()
        result = copy.deepcopy(result)
        result["action_summary"] = []
        for row in result["actions"]:
            command = row.get("command") or {}
            row["odometry_reason"] = ("no_current_pre_action_odometry" if row["pre_odom"] is None
                else "no_new_matching_post_stop_odometry" if row["post_odom"] is None else "samples_retained_independence_unverified")
            jit, after = row.get("jit_target_association") or {}, row.get("next_target_association") or {}
            before_range, after_range = jit.get("measured_distance_m"), after.get("measured_distance_m")
            delta = after_range - before_range if type(before_range) in (int, float) and type(after_range) in (int, float) else None
            result["action_summary"].append({
                "action_number": row["action_number"], "type": row["type"],
                "motion_executed": row["motion_executed"],
                "interrupted": row.get("interrupted", False),
                "interruption_reason": row.get("interruption_reason"),
                "source_stamp_consumed": row.get("source_stamp_consumed"),
                "full_step_completed": row.get("full_step_completed", row["motion_executed"]),
                "actual_confirmed_run_duration_seconds": row.get("actual_confirmed_run_duration_seconds"),
                "source_frame_stamp_ns": row["authorizing_camera"]["source_frame_stamp_ns"],
                "tracker_quality": row["authorizing_camera"]["tracker_quality"],
                "command_linear_speed": command.get("linear_x"), "command_angular_speed": command.get("angular_z"),
                "command_duration_seconds": command.get("duration"),
                "requested_duration_seconds": command.get("requested_duration"),
                "command_start_monotonic_seconds": command.get("start_monotonic_seconds"),
                "command_completion_monotonic_seconds": command.get("completion_monotonic_seconds"),
                "stop_monotonic_seconds": row.get("stopped_monotonic_seconds"),
                "requested_nominal_displacement_m": command.get("linear_x", 0) * command.get("duration", 0) if command else None,
                "nominal_displacement_m": (command.get("linear_x", 0) * command.get("duration", 0)
                                           if command and row["motion_executed"] else None),
                "odom_translation_m": (row.get("odom_delta") or {}).get("planar_translation_m"),
                "odom_yaw_change_radians": (row.get("odom_delta") or {}).get("yaw_radians"),
                "jit_measured_target_distance_m": before_range, "jit_conservative_target_distance_m": jit.get("target_distance_m"),
                "next_measured_target_distance_m": after_range, "next_conservative_target_distance_m": after.get("target_distance_m"),
                "target_range_delta_m": delta, "jit_selected_return": jit.get("selected_return"),
                "next_selected_return": after.get("selected_return"),
                "jit_lidar_sequence": (row.get("jit_lidar_evidence") or jit).get("acquisition_sequence"),
                "first_post_action_lidar_sequence": (row.get("first_post_action_lidar") or {}).get("acquisition_sequence"),
                "reacquisition_used": row["reacquisition_used"],
            })
        return result
