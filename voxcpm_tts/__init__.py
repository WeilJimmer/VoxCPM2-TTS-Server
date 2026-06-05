"""VoxCPM2 TTS service — a small FastAPI wrapper around the VoxCPM2 model.

Generates speech on the GPU, streams the resulting WAV to the caller, and
deletes the temp file once it has been sent. Designed to be driven by the
waifu-daemon browser client (TTS checkbox), and deployed on a GPU server
via the bundled systemd unit.
"""

__version__ = "0.1.0"
