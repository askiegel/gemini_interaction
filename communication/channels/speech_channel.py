class SpeechChannel:
    """
    Initial communication output channel.

    Phase 1 logs speech to the terminal.
    A later version will forward speech to the Robot Bridge.
    """

    def __init__(self):
        self._busy = False
        self.last_text = None
        self.last_event = None
        self.last_priority = None

    def say(self, text, event=None, priority=50):
        normalized_text = str(text or "").strip()

        if not normalized_text:
            raise ValueError("Speech text cannot be empty.")

        self._busy = True
        self.last_text = normalized_text
        self.last_event = event
        self.last_priority = int(priority)

        print(f"[SPEECH] {normalized_text}")

        self._busy = False

        return {
            "ok": True,
            "channel": "speech",
            "text": normalized_text,
            "event": event,
            "priority": self.last_priority,
        }

    def stop(self):
        self._busy = False
        print("[SPEECH STOP]")

        return {
            "ok": True,
            "channel": "speech",
            "stopped": True,
        }

    def is_busy(self):
        return self._busy
