from providers.gemini import GeminiProvider


def create_provider(config):
    provider_name = config.get("provider", "gemini").lower()

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=config["api_key"],
            model=config["model"],
        )

    raise ValueError(f"Unsupported cognitive provider: {provider_name}")
