#!/usr/bin/env python3

import json
import multiprocessing
import os
import tempfile
import time

from world_model import WorldModel


ITERATIONS = 20


def runtime_writer(
    storage_path,
    start_event,
):
    world_model = WorldModel(storage_path)
    start_event.wait()

    for index in range(1, ITERATIONS + 1):
        world_model.update_robot_state(
            runtime_process_alive=True,
            runtime_counter=index,
            runtime_state="TESTING",
        )

        time.sleep(0.005)


def vision_writer(
    storage_path,
    start_event,
):
    world_model = WorldModel(storage_path)
    start_event.wait()

    for index in range(1, ITERATIONS + 1):
        world_model.environment["vision"] = {
            "vision_status": "DETECTIONS_AVAILABLE",
            "camera_running": True,
            "detection_count": 1,
            "vision_counter": index,
        }

        world_model.update_entity(
            entity_id="backpack-concurrency-test",
            label="backpack",
            entity_type="object",
            confidence=0.90,
            source="concurrency_test",
            location={
                "cx": 320 + index,
                "cy": 240,
            },
            attributes={
                "iteration": index,
            },
        )

        time.sleep(0.005)


def main():
    temporary_directory = tempfile.mkdtemp(
        prefix="world_model_concurrency_"
    )

    storage_path = os.path.join(
        temporary_directory,
        "world_model_state.json",
    )

    lock_path = f"{storage_path}.lock"

    try:
        start_event = multiprocessing.Event()

        runtime_process = multiprocessing.Process(
            target=runtime_writer,
            args=(
                storage_path,
                start_event,
            ),
        )

        vision_process = multiprocessing.Process(
            target=vision_writer,
            args=(
                storage_path,
                start_event,
            ),
        )

        runtime_process.start()
        vision_process.start()

        start_event.set()

        runtime_process.join(timeout=20)
        vision_process.join(timeout=20)

        assert not runtime_process.is_alive()
        assert not vision_process.is_alive()

        assert runtime_process.exitcode == 0
        assert vision_process.exitcode == 0

        print("===== RAW JSON VALIDATION =====")

        with open(
            storage_path,
            "r",
            encoding="utf-8",
        ) as world_file:
            raw_data = json.load(world_file)

        print(
            json.dumps(
                {
                    "robot_state": raw_data.get(
                        "robot_state"
                    ),
                    "environment": raw_data.get(
                        "environment"
                    ),
                    "entity_ids": list(
                        raw_data.get(
                            "entities",
                            {}
                        ).keys()
                    ),
                    "event_count": len(
                        raw_data.get(
                            "recent_events",
                            []
                        )
                    ),
                },
                indent=2,
            )
        )

        world_model = WorldModel(storage_path)

        print()
        print("===== MERGED WORLD MODEL =====")
        print(
            json.dumps(
                world_model.get_context(),
                indent=2,
            )
        )

        assert (
            world_model.robot_state[
                "runtime_process_alive"
            ]
            is True
        )

        assert (
            world_model.robot_state[
                "runtime_counter"
            ]
            == ITERATIONS
        )

        assert (
            world_model.environment[
                "vision"
            ]["vision_counter"]
            == ITERATIONS
        )

        entity = world_model.get_entity(
            "backpack-concurrency-test"
        )

        assert entity is not None
        assert entity.label == "backpack"
        assert (
            entity.attributes["iteration"]
            == ITERATIONS
        )

        assert len(entity.history) == ITERATIONS

        assert len(
            world_model.recent_events
        ) > 0

        temporary_files = [
            filename
            for filename in os.listdir(
                temporary_directory
            )
            if filename.endswith(".tmp")
        ]

        assert temporary_files == []

        print()
        print(
            "PASS: concurrent writers produced valid JSON"
        )
        print(
            "PASS: runtime robot-state updates were preserved"
        )
        print(
            "PASS: vision environment updates were preserved"
        )
        print(
            "PASS: entity history survived concurrent writes"
        )
        print(
            "PASS: recent events were merged safely"
        )
        print(
            "PASS: atomic writes left no temporary files"
        )
        print()
        print(
            "World Model concurrency test passed."
        )

    finally:
        for path in (
            storage_path,
            lock_path,
        ):
            if os.path.exists(path):
                os.remove(path)

        if os.path.isdir(temporary_directory):
            for filename in os.listdir(
                temporary_directory
            ):
                os.remove(
                    os.path.join(
                        temporary_directory,
                        filename,
                    )
                )

            os.rmdir(temporary_directory)


if __name__ == "__main__":
    multiprocessing.set_start_method(
        "fork",
        force=True,
    )

    main()
