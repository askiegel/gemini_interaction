#!/usr/bin/env python3
import argparse
import base64
import json
import sys

from behavior_manager import BehaviorManager
from config import load_config
from mission_manager import MissionManager
from provider_factory import create_provider
from robot_bridge.client import RobotBridgeClient
from vision_adapter import VisionAdapter
from world_model import WorldModel


def decode_command(args):
    if args.base64:
        try:
            return base64.b64decode(args.base64).decode("utf-8").strip()
        except Exception as exc:
            raise ValueError(f"Invalid base64 command: {exc}") from exc

    if args.text:
        return args.text.strip()

    raise ValueError("No command text was provided.")


def run_command(user_text, execute=False):
    config = load_config()
    provider = create_provider(config)

    print("===== SPOKEN COMMAND =====")
    print(user_text)
    print()

    intent = provider.get_intent(user_text)

    print("===== INTENT =====")
    print(json.dumps(intent, indent=2))
    print()

    mission_manager = MissionManager()
    mission = mission_manager.handle_intent(intent)

    print("===== MISSION =====")
    print(json.dumps(mission.to_dict(), indent=2))
    print()

    world_model = WorldModel()
    vision_adapter = VisionAdapter(world_model=world_model)

    behavior_manager = BehaviorManager(
        robot_client=RobotBridgeClient(timeout=15.0),
        vision_adapter=vision_adapter,
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
            f"Robot behavior failed: {result.get('robot_result', result)}"
        )

    print("Voice command completed successfully.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Process one spoken command through the cognitive pipeline."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Plain-text command.")
    group.add_argument(
        "--base64",
        help="Base64-encoded UTF-8 command used by the browser relay.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the resulting behavior through the Robot Bridge.",
    )

    args = parser.parse_args()

    try:
        user_text = decode_command(args)

        if not user_text:
            raise ValueError("The command was empty.")

        run_command(user_text, execute=args.execute)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
