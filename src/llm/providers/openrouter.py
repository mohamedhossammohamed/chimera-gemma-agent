import os, time, json, logging, threading
from dataclasses import dataclass
from typing import Optional
from .base import LLMProvider

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

log = logging.getLogger(__name__)

@dataclass
class OpenRouterConfig:
    model: str = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # LOCAL SHELL: if set, Docker calls http://host.docker.internal:8000/v1 instead of OpenRouter
    # so Docker appears offline but host forwards to OpenRouter — easy to swap to local GGUF later
    local_shell_url: Optional[str] = os.getenv("LOCAL_SHELL_URL")  # e.g. http://host.docker.internal:8000/v1
    api_key: Optional[str] = None
    rpm: int = int(os.getenv("RATE_LIMIT_RPM", "18"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    timeout: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 1024

class TokenBucket:
    def __init__(self, rpm): self.rate=rpm/60; self.capacity=rpm; self.tokens=rpm; self.last=time.monotonic(); self.lock=threading.Lock()
    def consume(self, n=1):
        with self.lock:
            now=time.monotonic(); self.tokens=min(self.capacity, self.tokens+(now-self.last)*self.rate); self.last=now
            if self.tokens>=n: self.tokens-=n; return True
            return (n-self.tokens)/self.rate

_bucket=None
def _bucket_for(rpm):
    global _bucket
    if _bucket is None or _bucket.capacity!=rpm: _bucket=TokenBucket(rpm)
    return _bucket

class OpenRouterProvider(LLMProvider):
    """Calls OpenRouter (or local shell that forwards to it) — looks like local to Docker."""

    def __init__(self, cfg: OpenRouterConfig=None):
        self.cfg=cfg or OpenRouterConfig()
        self.key=(self.cfg.api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not self.key or "REPLACE" in self.key:
            log.warning("OPENROUTER_API_KEY not set — will fallback to mock")
            self.key=None
        # Use local shell URL if set — Docker thinks it's local
        base = self.cfg.local_shell_url or self.cfg.base_url
        # OpenRouter needs /v1, local shell mimics it
        if self.key and OpenAI:
            try:
                self.client=OpenAI(base_url=base, api_key=self.key, timeout=self.cfg.timeout)
                log.info(f"OpenRouterProvider {self.cfg.model} via {base} rpm{self.cfg.rpm} key...{self.key[-4:] if self.key else ''} (local_shell={bool(self.cfg.local_shell_url)})")
            except Exception as e:
                log.warning(f"OpenAI client init failed {e}")
                self.client=None
        else:
            self.client=None
        self.bucket=_bucket_for(self.cfg.rpm)

    def call(self, parsed, fused_emb):
        if not self.client or not self.key:
            from .mock import MockProvider
            return MockProvider().call(parsed,fused_emb) | {"fallback":"no_key"}
        # throttle
        w=self.bucket.consume(1)
        if isinstance(w,float):
            time.sleep(w); self.bucket.consume(1)
        # build prompt (same as before, compact)
        emb_summary=f"Fused:{len(fused_emb)} mean={sum(fused_emb)/len(fused_emb):.3f} first8={[round(x,2) for x in fused_emb[:8]]}" if fused_emb else ""
        clinical={k:v for k,v in {"age":parsed.age,"psa":parsed.psa,"psad":parsed.psad,"pirads":parsed.pirads,"bx_isup":parsed.bx_isup,"cspca":parsed.cspca,"psadt":parsed.psadt,"capra_s":parsed.capra_s,"surgical":{"margin":parsed.margin,"ece":parsed.ece}}.items() if v is not None}
        msgs=[{"role":"system","content":"Output ONLY valid JSON with decision, confidence, variable_weights, free_text."},
              {"role":"user","content":json.dumps({"clinical":clinical,"embed":emb_summary})}]
        for attempt in range(self.cfg.max_retries+1):
            try:
                resp=self.client.chat.completions.create(model=self.cfg.model, messages=msgs, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
                content=resp.choices[0].message.content
                try:
                    data=json.loads(content)
                except:
                    import re
                    m=re.search(r"\{.*\}",content,re.S)
                    data=json.loads(m.group(0)) if m else {"raw":content}
                # normalize
                vw=data.get("variable_weights",{})
                for k in ["psa","pirads","psad","bx","age"]:
                    vw.setdefault(k,"noted")
                data["variable_weights"]=vw
                data.setdefault("confidence","clear")
                data.setdefault("free_text",f"PSA {parsed.psa} PI-RADS {parsed.pirads} → {data.get('decision','')}")
                data["provider"]="openrouter"
                data["model"]=self.cfg.model
                return data
            except Exception as e:
                msg=str(e).lower()
                if "429" not in msg and "rate" not in msg:
                    log.error(f"OpenRouter fail {e}")
                    break
                if attempt==self.cfg.max_retries:
                    break
                time.sleep(min(self.cfg.max_backoff if hasattr(self.cfg,'max_backoff') else 60, 1*(2**attempt)))
        # fallback to mock on 429/credit
        from .mock import MockProvider
        m=MockProvider().call(parsed,fused_emb)
        m["fallback"]="openrouter_429"
        m["provider"]="openrouter→mock"
        return m
