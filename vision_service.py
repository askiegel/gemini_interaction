#!/usr/bin/env python3

import argparse
import json
import signal
import time
from typing import Any, Dict, Optional

from vision_adapter import VisionAdapter
from world_model import WorldModel


class VisionService:
    """
    Persistent perception service for Mini Pupper 2.

    Responsibilities:

        Vision Server
              ↓
        Vision Adapter
              ↓
        World Model

    The service does not execute robot behavior and does not manage missions.
    It continuously imports perception observations into the World Model.
    """

    DEFAULT_POLL_INTERVAL = 0.35

    def __init__(
        self,
        world_model=None,
        vision_adapter=None,
        poll_interval=None,
    ):
        self.world_model = world_model or WorldModel()

        self.poll_interval = float(
            poll_interval
            if poll_interval is not None
            else self.DEFAULT_POLL_INTERVAL
        )

        self.vision_adapter = vision_adapter or VisionAdapter(
            world_model=self.world_model,
            poll_interval=self.poll_interval,
        )

        self.running = False
        self.started_at: Optional[float] = None

        self.cycles_completed = 0
        self.successful_cycles = 0
        self.failed_cycles = 0

        self.last_cycle_at: Optional[float] = None
        self.last_entity_ids = []
        self.last_error: Optional[str] = None

    def run_once(self):
        """
        Execute one bounded perception update.

        Returns a serializable result describing the cycle.
        """
        cycle_started_at = time.time()

        try:
            entity_ids = self.vision_adapter.process_once()

            if entity_ids is None:
                entity_ids = []

            if not isinstance(entity_ids, list):
                entity_ids = list(entity_ids)

            self.cycles_completed += 1
            self.successful_cycles += 1
            self.last_cycle_at = cycle_started_at
            self.last_entity_ids = list(entity_ids)
            self.last_error = None

            result = {
                "ok": True,
                "cycle": self.cycles_completed,
                "entity_ids": list(entity_ids),
                "entity_count": len(entity_ids),
                "vision_status": self._vision_status(),
            }

            self.world_model.update_robot_state(
                perception_service_state="RUNNING",
                perception_last_cycle=result,
                perception_cycles_completed=self.cycles_completed,
                perception_successful_cycles=self.successful_cycles,
                perception_failed_cycles=self.failed_cycles,
            )

            return result

        except Exception as exc:
            self.cycles_completed += 1
            self.failed_cycles += 1
            self.last_cycle_at = cycle_started_at
            self.last_entity_ids = []
            self.last_error = str(exc)

            result = {
                "ok": False,
                "cycle": self.cycles_completed,
                "entity_ids": [],
                "entity_count": 0,
                "vision_status": self._vision_status(),
                "error": str(exc),
            }

            self.world_model.update_robot_state(
                perception_service_state="ERROR",
                perception_last_cycle=result,
                perception_cycles_completed=self.cycles_completed,
                perception_successful_cycles=self.successful_cycles,
                perception_failed_cycles=self.failed_cycles,
            )

            return result

    def run_forever(self):
        """
        Continuously update the World Model until shutdown is requested.
        """
        self.running = True
        self.started_at = time.time()

        self.world_model.update_robot_state(
            perception_service_state="STARTING",
            perception_service_running=True,
            perception_cycles_completed=self.cycles_completed,
            perception_successful_cycles=self.successful_cycles,
            perception_failed_cycles=self.failed_cycles,
        )

        print("============================================")
        print(" Mini Pupper 2 Vision Service")
        print("============================================")
        print(f"Vision URL:    {self.vision_adapter.vision_url}")
        print(f"Poll interval: {self.poll_interval:.2f} seconds")
        print("State:         RUNNING")
        print("Stop:          Ctrl+C")
        print()

        try:
            while self.running:
                result = self.run_once()

                if result.get("ok"):
                    print(
                        "[vision-service] "
                        f"cycle={result['cycle']} "
                        f"entities={result['entity_count']} "
                        f"status={result['vision_status']}"
                    )
                else:
                    print(
                        "[vision-service] "
                        f"cycle={result['cycle']} "
                        f"error={result['error']}"
                    )

                if self.running:
                    time.sleep(self.poll_interval)

        finally:
            self.running = False

            self.world_model.update_robot_state(
                perception_service_state="STOPPED",
                perception_service_running=False,
                perception_cycles_completed=self.cycles_completed,
                perception_successful_cycles=self.successful_cycles,
                perception_failed_cycles=self.failed_cycles,
            )

            print()
            print("Vision service stopped.")

    def stop(self):
        """
        Request a clean service shutdown.
        """
        self.running = False

    def get_status(self) -> Dict[str, Any]:
        uptime_seconds = None

        if self.started_at is not None:
            uptime_seconds = max(
                0.0,
                time.time() - self.started_at,
            )

        return {
            "ok": True,
            "service": "mini_pupper_vision_service",
            "running": self.running,
            "uptime_seconds": uptime_seconds,
            "poll_interval": self.poll_interval,
            "cycles_completed": self.cycles_completed,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "last_cycle_at": self.last_cycle_at,
            "last_entity_ids": list(self.last_entity_ids),
            "last_error": self.last_error,
            "vision_url": self.vision_adapter.vision_url,
            "vision_status": self._vision_status(),
        }

    def _vision_status(self):
        environment = getattr(
            self.world_model,
            "environment",
            {},
        )

        vision = environment.get("vision", {})

        return vision.get(
            "vision_status",
            "UNKNOWN",
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Continuously import Vision Server observations "
            "into the Mini Pupper 2 World Model."
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one perception cycle and exit.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Print initial service status and exit.",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help=(
            "Seconds between perception cycles. "
            "Defaults to the configured Vision Adapter interval."
        ),
    )

    args = parser.parse_args()

    if (
        args.poll_interval is not None
        and args.poll_interval <= 0
    ):
        raise SystemExit(
            "--poll-interval must be greater than zero."
        )

    service = VisionService(
        poll_interval=args.poll_interval,
    )

    def handle_shutdown(signum, frame):
        del signum
        del frame
        service.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    if args.status:
        print(
            json.dumps(
                service.get_status(),
                indent=2,
            )
        )
        return

    if args.once:
        result = service.run_once()

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

        if not result.get("ok"):
            raise SystemExit(1)

        return

    service.run_forever()


if __name__ == "__main__":
    main()
