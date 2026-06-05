#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Weil <me@weils.net>
"""Local smoke test: load VoxCPM2 and synthesize one line to out.wav.

Run from the project root after installing deps:

    python scripts/smoke_test.py "你好，我是你的老婆 Ariel～"

Verifies the engine end-to-end (model load → generate → WAV) without the
HTTP layer. The output file is written to ./smoke_out.wav for you to listen.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxcpm_tts.config import load_config
from voxcpm_tts.engine import VoxCPMEngine


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else "你好，我是你的老婆 Ariel，很高興見到你～"
    cfg = load_config()
    print(f"[smoke] model={cfg.model.name} seed={cfg.generation.seed}")
    print(f"[smoke] voice prompt={cfg.voice.prompt!r}")

    engine = VoxCPMEngine(cfg)
    t0 = time.time()
    engine.load()
    print(f"[smoke] loaded in {time.time() - t0:.1f}s on {engine.device}, sr={engine.sample_rate}")

    t0 = time.time()
    wav_bytes, sr = engine.synthesize(text)
    dur = time.time() - t0
    out = Path(__file__).resolve().parent.parent / "smoke_out.wav"
    out.write_bytes(wav_bytes)
    print(f"[smoke] synthesized {len(wav_bytes)} bytes in {dur:.1f}s → {out}")


if __name__ == "__main__":
    main()
