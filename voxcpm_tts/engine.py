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
import re
import random
import logging
import threading
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from .config import Config

logger = logging.getLogger("voxcpm-tts.engine")

# Parentheses in the request text (half- and full-width). VoxCPM reads a leading
# "(...)" as a voice-design prompt, so the spoken text's own brackets matter.
_PAREN_PAIR_RE = re.compile(r"[\(（][^\(（\)）]*[\)）]")   # a (...) pair with its content
_BRACKET_CHARS_RE = re.compile(r"[\(\)（）]")             # just the bracket characters
_WS_RE = re.compile(r"\s+")

_FORMATS = {
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
    "wav": ("WAV", "PCM_16", "audio/wav"),
}

# numpy's RNG seed must be in [0, 2**32 - 1]; any other int is folded in.
_SEED_SPACE = 2**32


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
    @staticmethod
    def _normalize_seed(seed: Optional[int]) -> Optional[int]:
        """Fold any int seed into numpy's valid range [0, 2**32-1] (handles
        too-large and negative seeds). None stays None."""
        if seed is None:
            return None
        return int(seed) % _SEED_SPACE

    def _seed_everything(self, seed: Optional[int]) -> None:
        seed = self._normalize_seed(seed)
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

    @staticmethod
    def _process_parens(text: str, mode: Optional[str]) -> str:
        """Handle parentheses in the spoken text per ``mode``:
        "keep" → unchanged, "remove" → drop (...) and its content,
        "strip" (default) → drop only the bracket chars, keep the words.
        """
        mode = (mode or "strip").strip().lower()
        if mode == "keep":
            return text
        if mode == "remove":
            out = _PAREN_PAIR_RE.sub(" ", text)
        else:  # strip
            out = _BRACKET_CHARS_RE.sub(" ", text)
        return _WS_RE.sub(" ", out).strip()

    # ── Synthesis ────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        seed: Optional[int] = None,
        fmt: Optional[str] = None,
        paren_mode: Optional[str] = None,
    ) -> Tuple[bytes, int, str]:
        """Generate speech and return (audio_bytes, sample_rate, media_type).

        Thread-safe: serialized on the GPU lock. All args except ``text`` are
        optional and fall back to config: ``seed`` (None → configured seed),
        ``fmt`` ("mp3"/"wav"), ``paren_mode`` ("strip"/"remove"/"keep").
        """
        if not self.ready:
            raise RuntimeError("engine not loaded")

        text = (text or "").strip()
        if not text:
            raise ValueError("empty text")

        # Handle the spoken text's own parentheses before adding the voice prompt.
        text = self._process_parens(text, paren_mode or self.cfg.output.paren_mode)
        if not text:
            raise ValueError("text empty after parenthesis processing")

        max_chars = self.cfg.generation.max_chars
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
            logger.debug("Truncated text to %d chars", max_chars)

        full_text = self._build_text(text)
        # Normalize up-front so the value we log/embed is the one actually used.
        use_seed = self._normalize_seed(seed if seed is not None else self.cfg.generation.seed)
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

        fmt = (fmt or self.cfg.output.format or "mp3").strip().lower()
        if fmt not in _FORMATS:
            fmt = "mp3"
        audio_bytes = self._encode(wav, fmt, meta)
        return audio_bytes, self._sample_rate, _FORMATS[fmt][2]

    def _encode(self, wav: np.ndarray, fmt: str, meta: Optional[dict]) -> bytes:
        """Encode a float32 waveform to ``fmt`` ("mp3"/"wav") bytes, optionally
        embedding the given string tags (RIFF INFO / ID3, best-effort)."""
        sf_format, subtype, _ = _FORMATS.get(fmt, _FORMATS["mp3"])
        buf = io.BytesIO()
        if not meta:
            sf.write(buf, wav, self._sample_rate, format=sf_format, subtype=subtype)
            return buf.getvalue()
        with sf.SoundFile(buf, "w", samplerate=self._sample_rate, channels=1,
                          format=sf_format, subtype=subtype) as f:
            for attr, val in meta.items():
                try:
                    setattr(f, attr, val)
                except Exception:  # noqa: BLE001 - metadata is best-effort
                    pass
            f.write(wav)
        return buf.getvalue()
