from communication.channels import SpeechChannel


class CommunicationManager:
    """
    Central communication interface for the cognitive platform.

    All robot subsystems should communicate through this class
    instead of talking directly to output devices.
    """

    def __init__(self, speech_channel=None):
        self.speech = speech_channel or SpeechChannel()

    def say(self, text, event=None, priority=50):
        return self.speech.say(
            text=text,
            event=event,
            priority=priority,
        )

    def stop(self):
        return self.speech.stop()

    def is_busy(self):
        return self.speech.is_busy()
