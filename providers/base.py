from abc import ABC, abstractmethod


class CognitiveProvider(ABC):
    @abstractmethod
    def get_intent(self, user_text):
        pass
