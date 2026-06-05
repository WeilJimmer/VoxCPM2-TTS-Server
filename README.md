# VoxCPM2 TTS Service

A small GPU TTS microservice for the **waifu-daemon** Live2D companion. It keeps
[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) resident in VRAM and exposes a
single `POST /tts` endpoint that returns a `.wav`. The browser client plays the
audio and syncs it with the on-screen speech bubble.

- **Voice:** soft, sweet young-girl timbre via VoxCPM2 *voice design* (a
  parenthetical prompt — no reference clip needed), with a **fixed seed** for a
  stable voice across calls. All tunable in [`config.yaml`](config.yaml).
- **No retained audio:** each request synthesizes to a temp file, streams it,
  and deletes it once sent.
- **Footprint:** VoxCPM2 is ~2B params (bf16), ≈4–5 GB on disk, ~8 GB VRAM —
  comfortable on a 12 GB RTX 3060.

## Requirements

- Python ≥ 3.10, < 3.13
- NVIDIA GPU, CUDA ≥ 12.0 (CPU works but is slow)
- A CUDA build of PyTorch ≥ 2.5

## Install

```bash
cd tts
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

# 1) Install a CUDA build of torch FIRST (match your CUDA — cu124 shown):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) Then the rest:
pip install -r requirements.txt
```

The VoxCPM2 weights download automatically on first run (cached under
`HF_HOME` — set `model.hf_home` in `config.yaml`, or `VOXTTS_HF_HOME`, to keep
them on a roomy disk).

> **Windows notes:** install into a fresh venv (the global Scripts dir can
> hit `*.exe` file-locks). HuggingFace caching falls back to copies instead of
> symlinks unless you enable Developer Mode — it still works, just uses more
> disk.

## Run

```bash
# Quick engine check (writes ./smoke_out.wav, no HTTP):
python scripts/smoke_test.py "你好，我是你的老婆 Ariel～"

# Start the HTTP service:
python -m voxcpm_tts
```

Smoke-test the API:

```bash
curl -X POST http://localhost:9824/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，我是你的老婆 Ariel～"}' --output speech.wav

curl http://localhost:9824/health
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/` | — | service info |
| `GET` | `/health` | — | readiness, device, VRAM, sample rate |
| `POST` | `/tts` | `{"text": "...", "seed": 123?}` | `audio/wav` (streamed, then deleted) |

If `server.api_key` is set, send it as the `X-API-Key` header.

## Configuration

Everything lives in [`config.yaml`](config.yaml). The most relevant knobs:

| Key | Meaning |
|---|---|
| `voice.prompt` | Voice-design description (the soft girl voice). English is most reliable; a Chinese variant is provided commented out. |
| `voice.reference_wav` | Optional path to a reference clip → voice **cloning** (most consistent identity). |
| `generation.seed` | Fixed RNG seed for a reproducible voice (`null` = random each time). |
| `generation.cfg_value` | Guidance strength (higher = sticks closer to the prompt). |
| `generation.inference_timesteps` | Diffusion steps (quality vs. speed). |
| `generation.max_chars` | Hard cap on input length. |
| `model.hf_home` | Where to cache weights. |
| `model.optimize` | `torch.compile` the model (skip on few-SM GPUs). |
| `model.cudnn_enabled` | Set `false` to bypass cuDNN (see Troubleshooting). |
| `model.matmul_precision` | `high` = TF32 (faster), `highest` = full fp32. |

A few deploy-time values can also be overridden by env vars (handy for keeping
**one shared `config.yaml`** across machines and differing only via env):
`VOXTTS_HOST`, `VOXTTS_PORT`, `VOXTTS_API_KEY`, `VOXTTS_MODEL`, `VOXTTS_HF_HOME`,
`VOXTTS_DEVICE`, `VOXTTS_SEED`, `VOXTTS_OPTIMIZE` (0/1), `VOXTTS_CUDNN` (0/1),
`VOXTTS_MATMUL_PRECISION`.

These can live in a **`.env`** in the project root (loaded automatically at
startup via python-dotenv; real shell env vars still win). Copy the template
and uncomment the block for your machine:

```bash
cp .env.example .env   # then edit — GB10 and Windows presets are inside
```

## Deploy on a GPU server (systemd)

A unit is provided at [`systemd/voxcpm-tts.service`](systemd/voxcpm-tts.service)
— **not installed automatically**:

```bash
sudo cp systemd/voxcpm-tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voxcpm-tts
journalctl -u voxcpm-tts -f
```

Adjust `User`, `WorkingDirectory`, `ExecStart`, and the `Environment=` lines for
your box. Then point the waifu client's **TTS server URL** at
`http://<server>:9824/tts` and tick the **啟用語音 (TTS)** checkbox.

## Troubleshooting

### `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` on load (e.g. GB10 Blackwell)

The model loads and the diffusion warmup runs, then it crashes on the first
conv in the AudioVAE. Cause: the host has a **system cuDNN** in `/lib` whose
sublibraries get `dlopen`-ed ahead of the pip wheel's, mixing versions. Check:

```bash
ldconfig -p | grep -i cudnn        # system cuDNN present in /lib/... → the culprit
```

**Preferred fix** — force torch to use the venv's bundled cuDNN:

```bash
# nvidia.cudnn is a namespace package (__file__ is None) — use __path__:
CUDNN_LIB=$(.venv/bin/python -c "import nvidia.cudnn,os;print(os.path.join(list(nvidia.cudnn.__path__)[0],'lib'))")
LD_LIBRARY_PATH="$CUDNN_LIB" python -m voxcpm_tts
```

> Note: `LD_LIBRARY_PATH` must be set **before** launch (the dynamic linker
> reads it at process start), so it can't come from `.env`. If you'd rather not
> deal with it, the `.env` route is simpler: set `VOXTTS_CUDNN=0` to bypass
> cuDNN entirely.

Make it permanent via the `Environment=LD_LIBRARY_PATH=...` line in the systemd
unit. **Fallback** (no cuDNN at all, slightly slower convs): set
`model.cudnn_enabled: false` in `config.yaml`.

## Notes

- The browser client must be served over **http** (not https) to call an http
  TTS endpoint, or you'll hit mixed-content blocking. Use a TLS reverse proxy
  if you need https end-to-end.
- Voice-design output can vary run to run; the fixed seed pins it. For a
  rock-solid identity, drop a clean reference clip in `voice.reference_wav`.
