#!/usr/bin/env python3
"""
Local shell — makes OpenRouter look like a local model to Docker.

Host runs:  python scripts/local_shell.py  # listens on 0.0.0.0:8000
Docker sets: LOCAL_SHELL_URL=http://host.docker.internal:8000/v1
             MODEL_PROVIDER=openrouter
             (Docker thinks it's calling a local model, host forwards to OpenRouter)

So Docker is "offline" from Grand Challenge perspective if you later swap to MODEL_PROVIDER=local,
but for now you can develop with OpenRouter without bundling 16GB.

Lean: <50KB code, no model, just FastAPI forward.
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

OPENROUTER_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")

app = FastAPI(title="Local Shell — OpenRouter forward")

@app.get("/health")
def health():
    return {"status": "ok", "mode": "local_shell", "forward": OPENROUTER_URL, "model": MODEL, "has_key": bool(OPENROUTER_KEY)}

@app.post("/v1/chat/completions")
async def forward(req: Request):
    if not OPENROUTER_KEY:
        return JSONResponse({"error": "OPENROUTER_API_KEY not set on host"}, status_code=500)
    body = await req.json()
    # force model to env MODEL if not set
    body.setdefault("model", MODEL)
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(OPENROUTER_URL, json=body, headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"})
        return JSONResponse(content=r.json(), status_code=r.status_code)

if __name__ == "__main__":
    import uvicorn
    print(f"Local shell listening on 0.0.0.0:8000 → forward to {OPENROUTER_URL} model {MODEL}")
    print("Docker: set LOCAL_SHELL_URL=http://host.docker.internal:8000/v1  MODEL_PROVIDER=openrouter")
    uvicorn.run(app, host="0.0.0.0", port=8000)
