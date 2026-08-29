"""Provider plugins — easy to swap: mock (offline) → openrouter (host) → local (GGUF)"""
from .factory import get_provider

__all__ = ["get_provider"]
