# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Weil <me@weils.net>
"""Configuration loader for the VoxCPM2 TTS service.

Reads config.yaml (next to the project root by default) into typed dataclasses,
then applies environment-variable overrides so the systemd unit can tweak a few
values without editing the file. Env names are prefixed with ``VOXTTS_``.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency — env vars still work without it
    load_dotenv = None

logger = logging.getLogger("voxcpm-tts.config")

# config.yaml lives at the project root (parent of this package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 9824
    api_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class ModelConfig:
    name: str = "openbmb/VoxCPM2"
    hf_home: Optional[str] = None
    device: Optional[str] = None
    load_denoiser: bool = False
    # torch.compile via VoxCPM's optimizer. Big win on datacenter GPUs; little
    # benefit on few-SM parts (GB10/consumer) and adds startup compile time.
    optimize: bool = True
    # cuDNN convolution backend. On brand-new GPUs (e.g. GB10 Blackwell) a
    # *system* cuDNN can shadow the pip one and crash with a sublibrary version
    # mismatch — set false to bypass cuDNN (slightly slower convs, no crash).
    # Prefer fixing LD_LIBRARY_PATH (see README); this is the fallback.
    cudnn_enabled: bool = True
    # torch.set_float32_matmul_precision: "high"/"medium" enable TF32 (faster on
    # Ampere+; negligible quality impact), "highest" = full fp32. null = leave
    # the torch default untouched.
    matmul_precision: Optional[str] = "high"


@dataclass
class VoiceConfig:
    prompt: str = ""
    # String inserted between the "(prompt)" and the text. VoxCPM examples use
    # either "" (no space) or " " — match whatever produced your good result
    # elsewhere, since it changes tokenization.
    prompt_separator: str = " "
    reference_wav: Optional[str] = None
    reference_text: Optional[str] = None


@dataclass
class GenerationConfig:
    seed: Optional[int] = 20240601
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize: bool = True
    # Run the optional denoiser on the OUTPUT. Requires model.load_denoiser=true.
    denoise: bool = False
    # VoxCPM auto-retries "bad cases" with changed conditions — great for
    # robustness, but it silently perturbs a fixed-seed result. Set false for
    # strict, reproducible output (e.g. to match a ComfyUI generation).
    retry_badcase: bool = True
    retry_badcase_max_times: int = 3
    retry_badcase_ratio_threshold: float = 6.0
    max_chars: Optional[int] = 400
    # Embed the generation params (seed/cfg/steps/prompt/text…) into each wav's
    # metadata (RIFF INFO comment) — handy for debugging the live service.
    embed_meta: bool = False


@dataclass
class OutputConfig:
    # Default response audio format: "mp3" or "wav". Overridable per request.
    format: str = "mp3"
    # How to treat parentheses in the request text (VoxCPM reads a leading
    # "(...)" as a voice-design prompt, so the spoken text's own brackets can be
    # misinterpreted): "strip" = drop the brackets but keep the words (replace
    # with a space), "remove" = drop brackets AND their inner text, "keep" =
    # leave untouched. Overridable per request.
    paren_mode: str = "strip"


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ── Env-var coercers ────────────────────────────────────────────────
# Each takes (raw_env_string, current_value) and returns the parsed value,
# falling back to current on parse errors.

def _coerce_bool(value, fallback):
    if value is None:
        return fallback
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return fallback


def _as_str(raw, cur):          # exact value, whitespace preserved (e.g. " ")
    return raw


def _as_str_or_none(raw, cur):  # empty/blank → None (disable)
    s = raw.strip()
    return s if s else None


def _as_int(raw, cur):
    try:
        return int(raw)
    except (TypeError, ValueError):
        return cur


def _as_int_or_none(raw, cur):
    if raw.strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return cur


def _as_float(raw, cur):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return cur


def _as_csv_list(raw, cur):
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or cur


# (env var, config section, field, coercer). Covers EVERY config field.
_ENV_MAP = [
    # server
    ("VOXTTS_HOST",                          "server",     "host",                          _as_str),
    ("VOXTTS_PORT",                          "server",     "port",                          _as_int),
    ("VOXTTS_API_KEY",                       "server",     "api_key",                       _as_str_or_none),
    ("VOXTTS_CORS_ORIGINS",                  "server",     "cors_origins",                  _as_csv_list),
    # model
    ("VOXTTS_MODEL",                         "model",      "name",                          _as_str),
    ("VOXTTS_HF_HOME",                       "model",      "hf_home",                       _as_str_or_none),
    ("VOXTTS_DEVICE",                        "model",      "device",                        _as_str_or_none),
    ("VOXTTS_LOAD_DENOISER",                 "model",      "load_denoiser",                 _coerce_bool),
    ("VOXTTS_OPTIMIZE",                      "model",      "optimize",                      _coerce_bool),
    ("VOXTTS_CUDNN",                         "model",      "cudnn_enabled",                 _coerce_bool),
    ("VOXTTS_MATMUL_PRECISION",              "model",      "matmul_precision",              _as_str_or_none),
    # voice
    ("VOXTTS_VOICE_PROMPT",                  "voice",      "prompt",                        _as_str),
    ("VOXTTS_VOICE_PROMPT_SEPARATOR",        "voice",      "prompt_separator",              _as_str),
    ("VOXTTS_REFERENCE_WAV",                 "voice",      "reference_wav",                 _as_str_or_none),
    ("VOXTTS_REFERENCE_TEXT",                "voice",      "reference_text",                _as_str_or_none),
    # generation
    ("VOXTTS_SEED",                          "generation", "seed",                          _as_int_or_none),
    ("VOXTTS_CFG_VALUE",                     "generation", "cfg_value",                     _as_float),
    ("VOXTTS_INFERENCE_TIMESTEPS",           "generation", "inference_timesteps",           _as_int),
    ("VOXTTS_NORMALIZE",                     "generation", "normalize",                     _coerce_bool),
    ("VOXTTS_DENOISE",                       "generation", "denoise",                       _coerce_bool),
    ("VOXTTS_RETRY_BADCASE",                 "generation", "retry_badcase",                 _coerce_bool),
    ("VOXTTS_RETRY_BADCASE_MAX_TIMES",       "generation", "retry_badcase_max_times",       _as_int),
    ("VOXTTS_RETRY_BADCASE_RATIO_THRESHOLD", "generation", "retry_badcase_ratio_threshold", _as_float),
    ("VOXTTS_MAX_CHARS",                     "generation", "max_chars",                     _as_int_or_none),
    ("VOXTTS_EMBED_META",                    "generation", "embed_meta",                    _coerce_bool),
    # output
    ("VOXTTS_FORMAT",                        "output",     "format",                        _as_str),
    ("VOXTTS_PAREN_MODE",                    "output",     "paren_mode",                    _as_str),
]


def _apply_env(cfg: Config) -> None:
    """Override any config field from its VOXTTS_* env var (see _ENV_MAP)."""
    env = os.environ
    for env_name, section, field_name, coercer in _ENV_MAP:
        if env_name in env:
            sec = getattr(cfg, section)
            setattr(sec, field_name, coercer(env[env_name], getattr(sec, field_name)))


def load_config(path: Optional[os.PathLike] = None) -> Config:
    """Load the config from YAML, falling back to defaults for any missing keys."""
    # Load a local .env first so its VOXTTS_* vars feed the env overrides below.
    # Real shell env vars take precedence (override=False).
    if load_dotenv is not None and DEFAULT_ENV_PATH.exists():
        load_dotenv(DEFAULT_ENV_PATH, override=False)
        logger.info("Loaded env from %s", DEFAULT_ENV_PATH)

    path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            logger.info("Loaded config from %s", path)
        except Exception as e:  # noqa: BLE001 - config errors should not be fatal
            logger.warning("Failed to read %s (%s) — using defaults", path, e)
    else:
        logger.warning("Config %s not found — using defaults", path)

    cfg = Config(
        server=ServerConfig(**{**ServerConfig().__dict__, **(data.get("server") or {})}),
        model=ModelConfig(**{**ModelConfig().__dict__, **(data.get("model") or {})}),
        voice=VoiceConfig(**{**VoiceConfig().__dict__, **(data.get("voice") or {})}),
        generation=GenerationConfig(
            **{**GenerationConfig().__dict__, **(data.get("generation") or {})}
        ),
        output=OutputConfig(**{**OutputConfig().__dict__, **(data.get("output") or {})}),
    )
    _apply_env(cfg)

    # Point HuggingFace at the configured cache dir before any model import.
    if cfg.model.hf_home:
        os.environ.setdefault("HF_HOME", cfg.model.hf_home)

    return cfg
