from communication import CommunicationManager


class MockSpeechChannel:
    def __init__(self):
        self.messages = []
        self.stopped = False

    def say(self, text, event=None, priority=50):
        self.messages.append((text, event, priority))
        return {
            "ok": True,
            "text": text,
            "event": event,
            "priority": priority,
        }

    def stop(self):
        self.stopped = True
        return {"ok": True}

    def is_busy(self):
        return False


def test_constructor():
    comm = CommunicationManager()
    assert comm is not None


def test_dependency_injection():
    mock = MockSpeechChannel()
    comm = CommunicationManager(speech_channel=mock)

    comm.say("Hello")

    assert len(mock.messages) == 1
    assert mock.messages[0][0] == "Hello"


def test_stop():
    mock = MockSpeechChannel()
    comm = CommunicationManager(speech_channel=mock)

    comm.stop()

    assert mock.stopped


if __name__ == "__main__":
    test_constructor()
    test_dependency_injection()
    test_stop()

    print("PASS: CommunicationManager tests")
