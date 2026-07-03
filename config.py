import os
from dotenv import load_dotenv


def load_config():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env file")

    return {
        "api_key": api_key,
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "log_file": os.getenv("GEMINI_LOG_FILE", "logs/conversation.log"),
    }
