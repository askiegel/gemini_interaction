import os
from typing import Optional

try:
    import speech_recognition as sr
except ImportError:
    sr = None


class MicrophoneError(RuntimeError):
    """Raised when microphone capture or speech recognition fails."""


class Microphone:
    """
    Provider-independent microphone input adapter.

    This class only converts microphone audio into text. It does not interpret
    commands and never communicates directly with the robot.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        timeout: float = 5.0,
        phrase_time_limit: float = 6.0,
        language: str = "en-US",
    ):
        if sr is None:
            raise MicrophoneError(
                "SpeechRecognition is required for microphone input. "
                "Install dependencies with: pip install -r requirements.txt"
            )

        self.device_index = device_index
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.language = language
        self.recognizer = sr.Recognizer()

        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.non_speaking_duration = 0.5

    @classmethod
    def from_environment(cls):
        raw_device_index = os.getenv("MICROPHONE_DEVICE_INDEX", "").strip()

        device_index = None
        if raw_device_index:
            try:
                device_index = int(raw_device_index)
            except ValueError as exc:
                raise MicrophoneError(
                    "MICROPHONE_DEVICE_INDEX must be an integer."
                ) from exc

        return cls(
            device_index=device_index,
            timeout=float(os.getenv("MICROPHONE_TIMEOUT", "5.0")),
            phrase_time_limit=float(
                os.getenv("MICROPHONE_PHRASE_TIME_LIMIT", "6.0")
            ),
            language=os.getenv("MICROPHONE_LANGUAGE", "en-US"),
        )

    @staticmethod
    def list_devices():
        if sr is None:
            raise MicrophoneError(
                "SpeechRecognition is required for microphone input. "
                "Install dependencies with: pip install -r requirements.txt"
            )
        return sr.Microphone.list_microphone_names()

    def listen(self):
        """
        Capture one spoken phrase and return recognized text.

        Returns:
            Recognized text as a stripped string.

        Raises:
            MicrophoneError if audio capture or recognition fails.
        """
        try:
            with sr.Microphone(device_index=self.device_index) as source:
                print("Voice > Calibrating for room noise...")
                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.75,
                )

                print("Voice > Listening...")

                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

        except sr.WaitTimeoutError as exc:
            raise MicrophoneError(
                "No speech was detected before the listening timeout."
            ) from exc

        except (OSError, AttributeError) as exc:
            raise MicrophoneError(
                f"Microphone capture failed: {exc}"
            ) from exc

        try:
            text = self.recognizer.recognize_google(
                audio,
                language=self.language,
            ).strip()

        except sr.UnknownValueError as exc:
            raise MicrophoneError(
                "Speech was captured, but it could not be understood."
            ) from exc

        except sr.RequestError as exc:
            raise MicrophoneError(
                f"Speech recognition service failed: {exc}"
            ) from exc

        if not text:
            raise MicrophoneError("Speech recognition returned empty text.")

        return text
