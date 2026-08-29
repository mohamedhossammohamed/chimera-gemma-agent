"""
Gemma OpenRouter client — secure, rate-limited, free-tier aware
Model: google/gemma-4-26b-a4b-it:free (swap via env OPENROUTER_MODEL)
Never hardcode API key — reads OPENROUTER_API_KEY from env / .env
Rate limit: 20 req/min free tier -> default 18 RPM with token bucket + exponential backoff on 429
"""
from __future__ import annotations
import os
import time
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    retry = lambda *a, **kw: (lambda f: f)
    stop_after_attempt = wait_exponential = retry_if_exception_type = None

log = logging.getLogger(__name__)

@dataclass
class GemmaConfig:
    model: str = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key: Optional[str] = None  # if None, reads env
    rpm: int = int(os.getenv("RATE_LIMIT_RPM", "18"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    base_backoff: float = float(os.getenv("BASE_BACKOFF_SECONDS", "1.0"))
    max_backoff: float = float(os.getenv("MAX_BACKOFF_SECONDS", "60.0"))
    timeout: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 1024

class TokenBucket:
    def __init__(self, rpm: int):
        self.rate = rpm / 60.0
        self.capacity = rpm
        self.tokens = rpm
        self.last = time.monotonic()
        self.lock = threading.Lock()
    def consume(self, n=1):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            # need to wait
            needed = n - self.tokens
            wait = needed / self.rate
            return wait

_bucket: Optional[TokenBucket] = None
_bucket_lock = threading.Lock()

def _get_bucket(rpm: int) -> TokenBucket:
    global _bucket
    with _bucket_lock:
        if _bucket is None or _bucket.capacity != rpm:
            _bucket = TokenBucket(rpm)
        return _bucket

class GemmaClient:
    def __init__(self, config: GemmaConfig = None):
        self.cfg = config or GemmaConfig()
        # resolve key securely
        key = self.cfg.api_key or os.getenv("OPENROUTER_API_KEY")
        if not key or key.strip() == "" or "REPLACE" in key:
            log.warning("OPENROUTER_API_KEY not set — Gemma calls will use mock (no network). Set env var to enable.")
            self.api_key = None
        else:
            self.api_key = key.strip()
            # never log full key
            log.info(f"Gemma client ready model={self.cfg.model} rpm={self.cfg.rpm} key=...{self.api_key[-4:]}")
        self.bucket = _get_bucket(self.cfg.rpm)
        self._client = None
        if OpenAI and self.api_key:
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.api_key, timeout=self.cfg.timeout)

    def _throttle(self):
        wait = self.bucket.consume(1)
        if isinstance(wait, float):
            log.info(f"Rate limit: waiting {wait:.1f}s for token bucket")
            time.sleep(wait)
            # after wait, consume
            self.bucket.consume(1)

    def _build_prompt(self, parsed, fused_emb: list[float]) -> list[dict]:
        # compact structured prompt — 95% token cut already done in parser
        # Keep embeddings summary, not full 1024 floats
        emb_summary = ""
        if fused_emb:
            # send mean/std/min/max + first 8 dims as anchor
            import math
            mean = sum(fused_emb)/len(fused_emb) if fused_emb else 0
            # avoid heavy compute
            emb_summary = f"Fused embedding: n={len(fused_emb)} mean={mean:.3f} first8={[round(x,2) for x in fused_emb[:8]]}"
        # build clinical snippet
        clinical = {
            "age": parsed.age,
            "psa": parsed.psa,
            "psad": parsed.psad,
            "pirads": parsed.pirads,
            "bx_isup": parsed.bx_isup,
            "bx_gl": f"{parsed.bx_gl_prim}+{parsed.bx_gl_sec}" if parsed.bx_gl_prim else None,
            "ct": parsed.ct,
            "dre": parsed.dre,
            "family_history": parsed.family_history,
            "psadt": parsed.psadt,
            "psav": parsed.psav,
            "v_shaped": parsed.v_shaped,
            "capra_s": parsed.capra_s,
            "surgical": {"margin": parsed.margin, "ece": parsed.ece, "svi": parsed.svi, "lni": parsed.lni, "pt": parsed.pt},
            "pmhx_source": parsed.pmhx_source,
        }
        # clean None
        clinical = {k:v for k,v in clinical.items() if v is not None}
        system = (
            "You are a prostate cancer assistant. Output ONLY valid JSON. "
            "Fields: decision (yes/no or active_surveillance/active_treatment/continued_surveillance/watchful_waiting or risk), "
            "confidence (clear/uncertain), variable_weights {psa,pirads,psad,bx,age etc: decisive/important/noted/not_used}, "
            "free_text (1 sentence rationale), capra_s_breakdown if surgical."
        )
        user = json.dumps({"clinical": clinical, "embed_summary": emb_summary}, indent=None)
        # append raw snippet hint for safety (small)
        if hasattr(parsed, "errors") and parsed.errors:
            user += f"\nNotes: {'; '.join(parsed.errors[:2])}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def call(self, parsed, fused_emb: list[float]) -> dict:
        messages = self._build_prompt(parsed, fused_emb)
        # mock mode if no key or no openai lib
        if not self.api_key or not self._client:
            return self._mock_response(parsed)
        # throttle
        self._throttle()
        # retry with exponential backoff on 429/5xx
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    response_format={"type": "json_object"} if "gemma" in self.cfg.model else None,
                )
                content = resp.choices[0].message.content
                # try json parse
                try:
                    data = json.loads(content)
                    # validate required keys, add defaults
                    return self._normalize_response(data, parsed)
                except json.JSONDecodeError:
                    # try to extract json
                    import re
                    m = re.search(r"\{.*\}", content, re.S)
                    if m:
                        try:
                            return self._normalize_response(json.loads(m.group(0)), parsed)
                        except:
                            pass
                    return {"raw": content, "parsed": False, "free_text": content[:200]}
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                is_retryable = "429" in msg or "rate" in msg or "timeout" in msg or "5" in msg[:1]
                if not is_retryable or attempt == self.cfg.max_retries:
                    log.error(f"Gemma call failed attempt {attempt}: {e}")
                    break
                backoff = min(self.cfg.base_backoff * (2 ** attempt), self.cfg.max_backoff)
                log.warning(f"Gemma retry {attempt+1}/{self.cfg.max_retries} after {backoff:.1f}s: {e}")
                time.sleep(backoff)
        # fallback mock on failure
        log.warning(f"Gemma failed after retries, using mock: {last_err}")
        mock = self._mock_response(parsed)
        mock["_gemma_error"] = str(last_err) if last_err else "unknown"
        return mock

    def _normalize_response(self, data: dict, parsed) -> dict:
        # ensure variable_weights has expected keys
        vw = data.get("variable_weights", {})
        expected = ["psa","pirads","psad","bx","age","dre","vol","cspca","comorbidity","fh"]
        for k in expected:
            if k not in vw:
                vw[k] = "noted" if k in ("psa","pirads") else "not_used"
        data["variable_weights"] = vw
        if "confidence" not in data:
            data["confidence"] = "clear" if parsed.pirads in (4,5) or parsed.psa and parsed.psa>10 else "uncertain"
        if "free_text" not in data:
            data["free_text"] = f"PSA {parsed.psa} PI-RADS {parsed.pirads} ISUP {parsed.bx_isup}."
        return data

    def _mock_response(self, parsed) -> dict:
        # deterministic mock based on parsed — useful for offline tests / RAM constrained laptop
        # Mirrors expected JSON without network
        task = parsed.task
        if task == 1:
            decision = "yes" if (parsed.pirads in (4,5) or (parsed.psa and parsed.psa>10) or parsed.bx_isup in (4,5)) else "no"
            return {
                "decision": decision,
                "confidence": "clear" if parsed.pirads else "uncertain",
                "variable_weights": {"pirads":"decisive" if parsed.pirads in (4,5) else "important","psa":"important","psad":"important","bx":"important" if parsed.bx_isup else "not_used","age":"noted","dre":"noted","fh":"noted","comorbidity":"noted","vol":"noted","cspca":"not_used"},
                "free_text": f"PI-RADS {parsed.pirads} PSA {parsed.psa} ISUP {parsed.bx_isup} → {decision}",
                "mock": True,
            }
        elif task == 2:
            # 4-class
            if parsed.bx_isup in (4,5) or (parsed.psa and parsed.psa>20):
                dec="active_treatment"
            elif parsed.bx_isup in (1,2) and parsed.psa and parsed.psa<10:
                dec="active_surveillance"
            elif parsed.bx_isup==3:
                dec="continued_surveillance"
            else:
                dec="active_surveillance"
            return {"decision": dec, "confidence":"clear","variable_weights":{"bx_isup":"decisive","pirads":"decisive","psa":"important"},"free_text":f"ISUP {parsed.bx_isup} PSA {parsed.psa} → {dec}","mock":True}
        else:
            # Task3: risk + rationale, survival handled by PyCox separately
            risk="high" if parsed.capra_s and parsed.capra_s>=6 else "low" if parsed.capra_s and parsed.capra_s<=2 else "intermediate"
            return {"decision": risk, "confidence":"clear","variable_weights":{"psa":"important","capra_s":"decisive"},"free_text":f"CAPRA-S {parsed.capra_s} → {risk} risk","mock":True}
