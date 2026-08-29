from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ROOT / "voice_relay"),
)

from tony2_mapping_runtime import (  # noqa: E402
    Tony2MappingRuntime,
)


PROBE = (
    ROOT
    / "voice_relay"
    / "tony2_mapping_probe.py"
).read_text()


def test_mapping_probe_reads_map_to_base_link_tf():
    assert "TransformListener" in PROBE
    assert "lookup_transform(" in PROBE
    assert '"map"' in PROBE
    assert '"base_link"' in PROBE
    assert '"pose_telemetry"' in PROBE


def test_live_pose_status_returns_tony2_pose(tmp_path):
    runtime = Tony2MappingRuntime(
        runtime_dir=tmp_path,
    )

    runtime.status = lambda: {
        "state": "RUNNING",
        "running": True,
        "owned": True,
        "host": "Tony2",
    }

    pose = {
        "frame_id": "map",
        "source_frame_id": "base_link",
        "position": {
            "x": 0.1,
            "y": -0.2,
            "z": 0.0,
        },
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0,
        },
        "stamp_seconds": 123.0,
    }

    runtime._read_snapshot = lambda: {
        "pose_telemetry": {
            "available": True,
            "status": "READY",
            "received_at": "now",
            "age_seconds": 0.1,
            "error": None,
            "pose": pose,
            "source": "cartographer_tf",
        },
    }

    status_code, payload = (
        runtime.live_pose_status()
    )

    assert status_code == 200
    assert payload["ok"] is True
    assert payload["runtime_active"] is True
    assert payload["source"] == "live_cartographer_tf"

    telemetry = payload["telemetry"]

    assert telemetry["available"] is True
    assert telemetry["status"] == "READY"
    assert telemetry["pose"] == pose


def test_live_pose_status_fails_closed_without_snapshot(
    tmp_path,
):
    runtime = Tony2MappingRuntime(
        runtime_dir=tmp_path,
    )

    runtime.status = lambda: {
        "state": "RUNNING",
        "running": True,
        "owned": True,
        "host": "Tony2",
    }

    runtime._read_snapshot = lambda: None

    status_code, payload = (
        runtime.live_pose_status()
    )

    assert status_code == 503
    assert payload["ok"] is False
    assert (
        payload["telemetry"]["status"]
        == "WAITING_FOR_POSE"
    )
    assert payload["telemetry"]["pose"] is None
