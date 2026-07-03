import json
from datetime import datetime


class InteractionLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, user_text, intent_json):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_text": user_text,
            "intent_json": intent_json,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
