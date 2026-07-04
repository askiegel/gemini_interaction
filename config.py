import os
from dotenv import load_dotenv


def load_config():
    load_dotenv()

    provider = os.getenv("COGNITIVE_PROVIDER", "gemini")
    api_key = os.getenv("GEMINI_API_KEY")

    if provider == "gemini" and not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env file")

    return {
        "provider": provider,
        "api_key": api_key,
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "log_file": os.getenv("COGNITIVE_LOG_FILE", "logs/conversation.log"),
    }
