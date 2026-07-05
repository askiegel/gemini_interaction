from world_model import WorldModel
from vision_adapter import VisionAdapter
from config import settings


def main():
    wm = WorldModel(settings.world.storage_path)

    adapter = VisionAdapter(
        wm,
        vision_url=settings.vision.url,
        image_path=settings.vision.image_path,
        poll_interval=settings.vision.poll_interval
    )

    adapter.run_forever()


if __name__ == "__main__":
    main()
