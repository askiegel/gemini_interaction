from config import settings


def main():
    print("")
    print("Mini Pupper Cognitive Configuration")
    print("===================================")

    print("")
    print("World")
    print("-----")
    print("storage_path:", settings.world.storage_path)

    print("")
    print("Vision")
    print("------")
    print("url:", settings.vision.url)
    print("image_path:", settings.vision.image_path)
    print("poll_interval:", settings.vision.poll_interval)

    print("")
    print("Robot")
    print("-----")
    print("name:", settings.robot.name)
    print("robot_id:", settings.robot.robot_id)
    print("platform:", settings.robot.platform)

    print("")
    print("Provider")
    print("--------")
    print("default_provider:", settings.provider.default_provider)


if __name__ == "__main__":
    main()
