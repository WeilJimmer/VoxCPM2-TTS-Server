"""FastAPI server exposing the VoxCPM2 engine.

Endpoints
    GET  /            → service info
    GET  /health      → readiness + device/VRAM
    POST /tts         → { "text": "...", "seed": 123? } → streamed WAV

Audio is written to a temp file, streamed to the caller, then deleted once
the response has been fully sent (BackgroundTask) — nothing is retained.
"""

from __future__ import annotations

import os
import asyncio
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .config import load_config
from .engine import VoxCPMEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voxcpm-tts.server")

cfg = load_config()
engine = VoxCPMEngine(cfg)

# Single worker → generation runs off the event loop but stays serialized
# (the engine also holds its own GPU lock as a belt-and-braces guard).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxcpm")

app = FastAPI(title="VoxCPM2 TTS", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.server.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    seed: Optional[int] = Field(None, description="Override the configured seed")


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Reject requests missing the shared secret, when one is configured."""
    if cfg.server.api_key and x_api_key != cfg.server.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.on_event("startup")
def _startup() -> None:
    logger.info("Loading model at startup (first run downloads weights)…")
    engine.load()
    logger.info("Model loaded — service ready on :%d", cfg.server.port)


@app.get("/")
def root():
    return {
        "service": "voxcpm2-tts",
        "model": cfg.model.name,
        "ready": engine.ready,
        "endpoints": ["/health", "POST /tts"],
    }


@app.get("/health")
def health():
    info = {
        "status": "ready" if engine.ready else "loading",
        "model": cfg.model.name,
        "device": engine.device,
        "sample_rate": engine.sample_rate,
    }
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["vram_free_mb"] = round(free / 1024 / 1024)
            info["vram_total_mb"] = round(total / 1024 / 1024)
    except Exception:  # noqa: BLE001
        pass
    return info


@app.post("/tts", dependencies=[Depends(require_api_key)])
async def tts(req: TTSRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="model still loading")

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    loop = asyncio.get_event_loop()
    try:
        wav_bytes, sr = await loop.run_in_executor(
            _executor, engine.synthesize, text, req.seed
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    # Write to a temp file, stream it, delete it once sent.
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voxtts_")
    with os.fdopen(fd, "wb") as f:
        f.write(wav_bytes)

    logger.info("Synthesized %d bytes @ %d Hz → %s", len(wav_bytes), sr, path)
    return FileResponse(
        path,
        media_type="audio/wav",
        filename="speech.wav",
        background=BackgroundTask(_safe_unlink, path),
        headers={"Cache-Control": "no-store"},
    )


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
