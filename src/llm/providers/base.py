from abc import ABC, abstractmethod
from typing import Any

class LLMProvider(ABC):
    """Single interface for all providers — Docker sees same call whether it's mock, OpenRouter, or local GGUF."""

    @abstractmethod
    def call(self, parsed: Any, fused_emb: list[float]) -> dict:
        """Return dict with decision, confidence, variable_weights, free_text. Never hardcode case_id→label."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__
