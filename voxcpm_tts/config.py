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


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def _coerce_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_bool(value, fallback):
    if value is None:
        return fallback
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off", ""):
        return False
    return fallback


def _apply_env(cfg: Config) -> None:
    """Override a handful of values from the environment (deploy-time tweaks)."""
    env = os.environ
    if "VOXTTS_HOST" in env:
        cfg.server.host = env["VOXTTS_HOST"]
    if "VOXTTS_PORT" in env:
        cfg.server.port = _coerce_int(env["VOXTTS_PORT"], cfg.server.port)
    if "VOXTTS_API_KEY" in env:
        cfg.server.api_key = env["VOXTTS_API_KEY"] or None
    if "VOXTTS_MODEL" in env:
        cfg.model.name = env["VOXTTS_MODEL"]
    if "VOXTTS_HF_HOME" in env:
        cfg.model.hf_home = env["VOXTTS_HF_HOME"] or None
    if "VOXTTS_DEVICE" in env:
        cfg.model.device = env["VOXTTS_DEVICE"] or None
    if "VOXTTS_OPTIMIZE" in env:
        cfg.model.optimize = _coerce_bool(env["VOXTTS_OPTIMIZE"], cfg.model.optimize)
    if "VOXTTS_CUDNN" in env:
        cfg.model.cudnn_enabled = _coerce_bool(env["VOXTTS_CUDNN"], cfg.model.cudnn_enabled)
    if "VOXTTS_MATMUL_PRECISION" in env:
        raw = env["VOXTTS_MATMUL_PRECISION"].strip()
        cfg.model.matmul_precision = raw or None
    if "VOXTTS_SEED" in env:
        raw = env["VOXTTS_SEED"]
        cfg.generation.seed = None if raw == "" else _coerce_int(raw, cfg.generation.seed)


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
    )
    _apply_env(cfg)

    # Point HuggingFace at the configured cache dir before any model import.
    if cfg.model.hf_home:
        os.environ.setdefault("HF_HOME", cfg.model.hf_home)

    return cfg
