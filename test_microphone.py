from microphone import Microphone, MicrophoneError


def main():
    print("===== MICROPHONE DEVICES =====")

    devices = Microphone.list_devices()

    if not devices:
        raise SystemExit("No microphone devices were detected.")

    for index, name in enumerate(devices):
        print(f"{index}: {name}")

    print()
    print("Speak one short test phrase.")
    print('Example: "Move forward"')
    print()

    microphone = Microphone.from_environment()

    try:
        text = microphone.listen()
    except MicrophoneError as exc:
        raise SystemExit(f"Microphone test failed: {exc}")

    print()
    print("===== RECOGNIZED TEXT =====")
    print(text)
    print()
    print("Microphone speech-to-text test passed.")
    print("No robot command was executed.")


if __name__ == "__main__":
    main()
