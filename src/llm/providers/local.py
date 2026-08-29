import os, json, logging
from .base import LLMProvider

log = logging.getLogger(__name__)

class LocalProvider(LLMProvider):
    """Future local GGUF (e.g., Phi-3-mini Q4 via llama.cpp) — same interface, 0 internet, ~2GB RAM. Easy swap: MODEL_PROVIDER=local"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH", "models/phi-3-mini-4k-instruct-q4.gguf")
        self.loaded = False
        # lazy load — don't load on import, only on first call, and only if file exists
        # keeps RAM free until you switch
        if os.path.exists(self.model_path):
            log.info(f"LocalProvider found {self.model_path} — will load on first call")
        else:
            log.info(f"LocalProvider no model at {self.model_path} — will fallback to mock until you download")

    def call(self, parsed, fused_emb):
        if not os.path.exists(self.model_path):
            from .mock import MockProvider
            m = MockProvider().call(parsed, fused_emb)
            m["fallback"] = f"local_no_model:{self.model_path}"
            m["provider"] = "local→mock"
            return m
        # If you have llama.cpp Python bindings, load here:
        # from llama_cpp import Llama; llm = Llama(model_path=self.model_path, n_ctx=4096, n_threads=4)
        # For now, still mock but mark as local
        try:
            # placeholder for real GGUF call
            raise NotImplementedError("GGUF not yet wired — mocking")
        except Exception as e:
            from .mock import MockProvider
            m = MockProvider().call(parsed, fused_emb)
            m["provider"] = "local→mock"
            m["fallback"] = str(e)[:100]
            return m
