#!/usr/bin/env python3

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

from behavior_manager import BehaviorManager
from config import load_config
from mission_manager import MissionManager
from provider_factory import create_provider
from robot_addressing import RobotAddressParser
from robot_bridge.client import RobotBridgeClient
from robot_fleet import load_robot_fleet
from robot_identity import get_robot_identity
from vision_adapter import VisionAdapter
from world_model import WorldModel


DEFAULT_RUNTIME_URL = os.getenv(
    "COGNITIVE_RUNTIME_URL",
    "http://127.0.0.1:8770",
)


def decode_command(args):
    if args.base64:
        try:
            return (
                base64.b64decode(args.base64)
                .decode("utf-8")
                .strip()
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid base64 command: {exc}"
            ) from exc

    if args.text:
        return args.text.strip()

    raise ValueError("No command text was provided.")


def submit_intent_to_runtime(
    user_text,
    intent,
    runtime_url=DEFAULT_RUNTIME_URL,
    timeout=10.0,
    robot_id=None,
    addressing=None,
):
    """
    Submit an already parsed intent to the persistent cognitive runtime.

    The provider is invoked only by voice_command.py. The runtime receives
    the provider-independent intent and converts it into a queued mission.
    """
    base_url = str(runtime_url).rstrip("/")
    url = f"{base_url}/missions"

    payload = {
        "source_text": user_text,
        "intent": intent,
    }

    if robot_id is not None:
        normalized_robot_id = str(robot_id).strip().lower()

        if not normalized_robot_id:
            raise ValueError(
                "robot_id must not be empty when provided."
            )

        payload["robot_id"] = normalized_robot_id

    if addressing is not None:
        if not isinstance(addressing, dict):
            raise ValueError(
                "addressing must be a dictionary when provided."
            )

        payload["addressing"] = dict(addressing)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=float(timeout),
        ) as response:
            response_body = response.read().decode("utf-8")
            result = json.loads(response_body)

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")

        try:
            error_payload = json.loads(body)
        except json.JSONDecodeError:
            error_payload = {
                "error": body or f"HTTP {exc.code}",
            }

        raise RuntimeError(
            "Runtime rejected the mission: "
            f"{error_payload.get('error', error_payload)}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to cognitive runtime at {url}: "
            f"{exc.reason}"
        ) from exc

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Cognitive runtime returned invalid JSON."
        ) from exc

    if not result.get("ok") or not result.get("accepted"):
        raise RuntimeError(
            "Cognitive runtime did not accept the mission: "
            f"{result}"
        )

    return result


def run_command(
    user_text,
    execute=False,
    submit_runtime=False,
    runtime_url=DEFAULT_RUNTIME_URL,
):
    config = load_config()
    provider = create_provider(config)

    local_identity = get_robot_identity()
    fleet = load_robot_fleet(
        local_identity=local_identity,
    )

    address_parser = RobotAddressParser(
        local_identity=local_identity,
        known_identities=fleet.remote_identities,
    )

    addressed = address_parser.parse(user_text)

    print("===== SPOKEN COMMAND =====")
    print(user_text)
    print()

    print("===== ROBOT ADDRESSING =====")
    print(json.dumps(addressed.to_dict(), indent=2))
    print()

    if not addressed.is_for(local_identity.id):
        result = {
            "ok": True,
            "accepted": False,
            "ignored": True,
            "reason": (
                "Command was addressed to another robot."
            ),
            "local_robot": local_identity.to_dict(),
            "addressing": addressed.to_dict(),
        }

        print("===== COMMAND IGNORED =====")
        print(json.dumps(result, indent=2))
        print()
        print(
            f"{local_identity.name} ignored a command addressed "
            "to another robot."
        )

        return result

    if not addressed.command_text:
        raise ValueError(
            "A robot name was recognized, but no command followed it."
        )

    intent = provider.get_intent(
        addressed.command_text
    )

    print("===== INTENT =====")
    print(json.dumps(intent, indent=2))
    print()

    # Both --runtime and the backward-compatible --execute option submit
    # through the persistent Cognitive Runtime. Execution ownership must remain
    # with the single runtime instance so commands cannot construct another
    # VisionAdapter or identity-processing pipeline.
    if submit_runtime or execute:
        result = submit_intent_to_runtime(
            user_text=user_text,
            intent=intent,
            runtime_url=runtime_url,
            robot_id=local_identity.id,
            addressing=addressed.to_dict(),
        )

        print("===== RUNTIME SUBMISSION =====")
        print(json.dumps(result, indent=2))
        print()

        mission = result.get("mission", {})

        print(
            "Mission accepted by persistent runtime: "
            f"{mission.get('mission_id', 'unknown')}"
        )

        print(
            "Mission status: "
            f"{mission.get('status', 'unknown')}"
        )

        return result

    mission_manager = MissionManager()
    mission = mission_manager.handle_intent(intent)

    print("===== MISSION =====")
    print(json.dumps(mission.to_dict(), indent=2))
    print()

    world_model = WorldModel()
    vision_adapter = VisionAdapter(
        world_model=world_model
    )

    behavior_manager = BehaviorManager(
        robot_client=RobotBridgeClient(timeout=15.0),
        vision_adapter=vision_adapter,
        world_model=world_model,
    )

    if not execute:
        simulation = behavior_manager.simulate(mission)

        result = {
            "ok": True,
            "executed": False,
            "mode": "dry-run",
            "behavior": simulation,
        }

        print("===== DRY RUN =====")
        print(json.dumps(result, indent=2))
        print()
        print("No robot command was sent.")
        return result

    result = behavior_manager.execute(mission)

    print("===== EXECUTION =====")
    print(json.dumps(result, indent=2))
    print()

    if not result.get("ok"):
        raise RuntimeError(
            "Robot behavior failed: "
            f"{result.get('robot_result', result)}"
        )

    print("Voice command completed successfully.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process one spoken command through the cognitive pipeline."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--text",
        help="Plain-text command.",
    )

    group.add_argument(
        "--base64",
        help=(
            "Base64-encoded UTF-8 command used by the "
            "browser relay."
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Submit the resulting behavior to the persistent "
            "cognitive runtime. Retained for CLI compatibility."
        ),
    )

    mode_group.add_argument(
        "--runtime",
        action="store_true",
        help=(
            "Submit the parsed intent to the persistent cognitive "
            "runtime mission queue."
        ),
    )

    parser.add_argument(
        "--runtime-url",
        default=DEFAULT_RUNTIME_URL,
        help=(
            "Persistent runtime base URL. Default: "
            f"{DEFAULT_RUNTIME_URL}"
        ),
    )

    args = parser.parse_args()

    try:
        user_text = decode_command(args)

        if not user_text:
            raise ValueError("The command was empty.")

        run_command(
            user_text=user_text,
            execute=args.execute,
            submit_runtime=args.runtime,
            runtime_url=args.runtime_url,
        )

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
