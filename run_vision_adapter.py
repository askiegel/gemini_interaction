from world_model import WorldModel
from vision_adapter import VisionAdapter


def main():
    wm = WorldModel()
    adapter = VisionAdapter(wm)
    adapter.run_forever()


if __name__ == "__main__":
    main()
