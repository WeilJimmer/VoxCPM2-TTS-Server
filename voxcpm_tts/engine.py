# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Weil <me@weils.net>
"""VoxCPM2 inference engine.

Loads the model once into VRAM and exposes a thread-safe ``synthesize`` that
turns text into a 16-bit PCM WAV (bytes). VoxCPM exposes no seed argument, so
we seed torch/numpy/random ourselves before every call for a reproducible,
stable voice.
"""

from __future__ import annotations

import io
import random
import logging
import threading
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from .config import Config

logger = logging.getLogger("voxcpm-tts.engine")


class VoxCPMEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._sample_rate = 48000  # VoxCPM2 default; refined after load
        # VoxCPM/torch generation is not safe to run concurrently on one GPU,
        # so every synthesize() call is serialized through this lock.
        self._lock = threading.Lock()
        self._device = None

    # ── Loading ──────────────────────────────────────────────────────
    def load(self) -> None:
        """Import torch/voxcpm and pull the weights into VRAM. Blocking."""
        import torch  # imported lazily so config errors surface before the heavy import
        from voxcpm import VoxCPM

        device = self.cfg.model.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        logger.info("Loading VoxCPM model '%s' on %s …", self.cfg.model.name, device)

        # ── Backend tunables (config-driven) ─────────────────────────
        m = self.cfg.model
        if m.matmul_precision:
            # TF32 matmuls: faster on Ampere+ with negligible quality impact.
            torch.set_float32_matmul_precision(m.matmul_precision)
            logger.info("float32_matmul_precision = %s", m.matmul_precision)
        # Bypass cuDNN when a mismatched system cuDNN would otherwise crash
        # convolutions (e.g. GB10 Blackwell sublibrary version mismatch).
        torch.backends.cudnn.enabled = m.cudnn_enabled
        if not m.cudnn_enabled:
            logger.warning("cuDNN disabled — convolutions use the native CUDA path")

        load_kwargs = dict(
            load_denoiser=m.load_denoiser,
            device=device,
            optimize=m.optimize,
        )
        if self.cfg.model.hf_home:
            # Keep the (large) weight cache on the configured drive.
            load_kwargs["cache_dir"] = self.cfg.model.hf_home
        self._model = VoxCPM.from_pretrained(self.cfg.model.name, **load_kwargs)

        # Best-effort: read the real output sample rate from the model.
        sr = getattr(getattr(self._model, "tts_model", None), "sample_rate", None)
        if isinstance(sr, (int, float)) and sr > 0:
            self._sample_rate = int(sr)
        logger.info("VoxCPM ready — sample_rate=%d Hz", self._sample_rate)

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def device(self) -> Optional[str]:
        return self._device

    # ── Helpers ──────────────────────────────────────────────────────
    def _seed_everything(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_text(self, text: str) -> str:
        """Prepend the voice-design prompt unless the caller already supplied one."""
        prompt = (self.cfg.voice.prompt or "").strip()
        stripped = text.lstrip()
        if prompt and not stripped.startswith("("):
            return f"{prompt}{self.cfg.voice.prompt_separator}{text}"
        return text

    # ── Synthesis ────────────────────────────────────────────────────
    def synthesize(self, text: str, seed: Optional[int] = None) -> Tuple[bytes, int]:
        """Generate speech for ``text`` and return (wav_bytes, sample_rate).

        Thread-safe: serialized on the GPU lock. ``seed`` overrides the config
        seed for this one call (None → use the configured seed).
        """
        if not self.ready:
            raise RuntimeError("engine not loaded")

        text = (text or "").strip()
        if not text:
            raise ValueError("empty text")

        max_chars = self.cfg.generation.max_chars
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
            logger.debug("Truncated text to %d chars", max_chars)

        full_text = self._build_text(text)
        use_seed = seed if seed is not None else self.cfg.generation.seed
        gen = self.cfg.generation
        voice = self.cfg.voice

        with self._lock:
            self._seed_everything(use_seed)
            kwargs = dict(
                text=full_text,
                cfg_value=gen.cfg_value,
                inference_timesteps=gen.inference_timesteps,
                normalize=gen.normalize,
                denoise=gen.denoise,
                retry_badcase=gen.retry_badcase,
                retry_badcase_max_times=gen.retry_badcase_max_times,
                retry_badcase_ratio_threshold=gen.retry_badcase_ratio_threshold,
            )
            if voice.reference_wav:
                kwargs["reference_wav_path"] = voice.reference_wav
                if voice.reference_text:
                    kwargs["prompt_wav_path"] = voice.reference_wav
                    kwargs["prompt_text"] = voice.reference_text
            # Log the EXACT call so a result can be diffed against another tool
            # (e.g. ComfyUI) when reproducing a voice.
            logger.info(
                "generate seed=%s cfg=%s steps=%s normalize=%s denoise=%s "
                "retry_badcase=%s text=%r",
                use_seed, gen.cfg_value, gen.inference_timesteps, gen.normalize,
                gen.denoise, gen.retry_badcase, full_text,
            )
            wav = self._model.generate(**kwargs)

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        # Clip to the valid range before 16-bit quantization (guards against
        # the model occasionally overshooting [-1, 1]).
        np.clip(wav, -1.0, 1.0, out=wav)

        if gen.embed_meta:
            meta = {
                "comment": (
                    f"seed={use_seed}; cfg={gen.cfg_value}; steps={gen.inference_timesteps}; "
                    f"normalize={gen.normalize}; denoise={gen.denoise}; "
                    f"retry_badcase={gen.retry_badcase}; text={full_text}"
                ),
                "title": f"VoxCPM2 seed={use_seed}",
                "artist": "VoxCPM2",
                "software": "voxcpm-tts",
            }
        else:
            meta = None
        return self._encode_wav(wav, meta), self._sample_rate

    def _encode_wav(self, wav: np.ndarray, meta: Optional[dict]) -> bytes:
        """Encode a float32 waveform to 16-bit WAV bytes, optionally embedding
        the given RIFF INFO string tags (title/artist/software/comment)."""
        buf = io.BytesIO()
        if not meta:
            sf.write(buf, wav, self._sample_rate, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        with sf.SoundFile(buf, "w", samplerate=self._sample_rate, channels=1,
                          format="WAV", subtype="PCM_16") as f:
            for attr, val in meta.items():
                try:
                    setattr(f, attr, val)
                except Exception:  # noqa: BLE001 - metadata is best-effort
                    pass
            f.write(wav)
        return buf.getvalue()
