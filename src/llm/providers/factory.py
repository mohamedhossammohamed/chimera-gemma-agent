import os

def get_provider(name: str = None):
    """
    MODEL_PROVIDER=mock | openrouter | local
    - mock: offline, 0 RAM, rule+XGBoost (default, submission-safe, no internet)
    - openrouter: via OpenRouter or via LOCAL_SHELL_URL (Docker → host → internet, looks local)
    - local: GGUF at LOCAL_MODEL_PATH (future, 2GB)

    Easy to update: just change .env — no code change.
    """
    name = (name or os.getenv("MODEL_PROVIDER", "mock")).strip().lower()
    if name == "openrouter":
        from .openrouter import OpenRouterProvider
        return OpenRouterProvider()
    elif name == "local":
        from .local import LocalProvider
        return LocalProvider()
    else:  # mock is default — fully offline, no LLM RAM
        from .mock import MockProvider
        return MockProvider()
