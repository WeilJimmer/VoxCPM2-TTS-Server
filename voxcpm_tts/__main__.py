# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Weil <me@weils.net>
"""Entry point: ``python -m voxcpm_tts`` starts the HTTP service."""

import logging

import uvicorn

from .config import load_config
from .server import app

logger = logging.getLogger("voxcpm-tts")


def main() -> None:
    cfg = load_config()
    logger.info("Starting VoxCPM2 TTS on %s:%d", cfg.server.host, cfg.server.port)
    # Pass the app object (not an import string) so we load config exactly once.
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")


if __name__ == "__main__":
    main()
