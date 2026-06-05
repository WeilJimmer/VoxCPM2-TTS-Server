# VoxCPM2 TTS Service

A small GPU text-to-speech microservice that gives the **waifu-daemon** Live2D
companion a voice. It keeps [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
resident in VRAM and exposes one endpoint, `POST /tts`, that returns a `.wav`.
The browser client plays the audio and syncs it with the on-screen speech bubble.

**Highlights**

- 🎀 **Soft young-girl voice** via VoxCPM2 *voice design* — a parenthetical
  prompt steers the timbre, no reference clip required.
- 🎯 **Fixed seed** for a stable voice across calls (VoxCPM has no seed arg, so
  torch/numpy/random are seeded manually).
- 🧹 **No retained audio** — each request synthesizes to a temp file, streams it,
  then deletes it.
- 🎛 **Everything is tunable** from `config.yaml`, and **every** field has a
  `VOXTTS_*` env override so one shared config works across machines.
- 🖥 **Runs on a 12 GB RTX 3060** (≈6.5 GB VRAM, ≈5 GB disk) and deploys to a
  Linux GPU server via the bundled systemd unit.

### How it fits together

```
 Live2D browser client  ──POST /tts {text}──►  this service (GPU)
 (waifu-daemon/client)   ◄──── audio/wav ─────  VoxCPM2 in VRAM
        │  plays audio, syncs the speech bubble
        ▼
   speaker 🔊
```

---

## Requirements

| | |
|---|---|
| Python | ≥ 3.10 and < 3.13 (3.12 recommended) |
| GPU | NVIDIA, CUDA ≥ 12.0 (CPU works but is very slow) |
| PyTorch | a **CUDA build** ≥ 2.5 |
| Disk | ≈5 GB for the model weights + a few GB for deps |
| VRAM | ≈6.5 GB with the defaults |

---

## Installation

### 🪟 Windows (RTX 3060) — step by step

Open **PowerShell** in the project folder.

**1. Create and activate a virtual environment.** Always use a venv on Windows —
installing into the global `C:\PythonXX` can hit `Scripts\*.exe` file-locks.

```powershell
cd D:\Weil\Desktop\AI\tts
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

> **Tip:** if you already have a CUDA build of torch installed globally and want
> to avoid re-downloading ~2.5 GB, create the venv with
> `python -m venv --system-site-packages .venv` instead — it reuses the global
> torch and you can skip step 2.

**2. Install a CUDA build of PyTorch first.** Pick the index URL matching your
CUDA (check with `nvidia-smi`; `cu124` and `cu121` are both fine on recent
drivers):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**3. Install the service dependencies:**

```powershell
pip install -r requirements.txt
```

**4. Verify the GPU is visible to torch:**

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# e.g. 2.5.1+cu121 True NVIDIA GeForce RTX 3060
```

**5. Put the model cache on a roomy drive** (the weights are ~5 GB; don't fill
`C:`). Create a `.env` (see [Configuration](#configuration)):

```powershell
Copy-Item .env.example .env
# then edit .env and set:
#   VOXTTS_HF_HOME=D:/Weil/Desktop/AI/tts/hf_cache
#   VOXTTS_OPTIMIZE=0          # triton isn't on Windows; skip torch.compile
```

> **HuggingFace symlink warning on Windows** is harmless — without Developer
> Mode it caches by copying instead of symlinking, just using a bit more disk.

### 🟩 Linux GPU server (DGX / GB10) — step by step

```bash
cd ~/repo/tts-server
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match the box's CUDA
pip install -r requirements.txt
cp .env.example .env            # then edit (see the GB10 preset inside)
```

On brand-new GPUs (e.g. GB10 Blackwell) you may hit a cuDNN error on first load —
see [Troubleshooting](#troubleshooting). The quick fix is `VOXTTS_CUDNN=0` in
`.env`.

---

## First run & verify

```bash
# 1) Engine-only smoke test — loads the model and writes ./smoke_out.wav (no HTTP).
python scripts/smoke_test.py "你好，我是你的老婆 Ariel～"

# 2) Start the HTTP service (first run downloads the weights, ~5 GB):
python -m voxcpm_tts
```

When you see `Model loaded — service ready on :9824`, check health:

```bash
curl http://localhost:9824/health
# {"status":"ready","device":"cuda","sample_rate":48000,"vram_free_mb":...}
```

---

## Usage

### Call the API

The service expects **UTF-8 JSON**. On Windows, the console codepage can mangle
non-ASCII text passed inline to `curl`, so for Chinese send the body **from a
UTF-8 file**:

**PowerShell (Windows) — robust for Chinese:**

```powershell
'{"text":"你好，我是你的老婆 Ariel～"}' | Out-File -Encoding utf8 req.json
curl.exe -X POST http://localhost:9824/tts `
  -H "Content-Type: application/json" `
  --data-binary "@req.json" --output speech.wav
```

**bash (Linux/macOS):**

```bash
curl -X POST http://localhost:9824/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，我是你的老婆 Ariel～","seed":20240601}' \
  --output speech.wav
```

`seed` is optional and overrides the configured seed for that one call.

### Connect the waifu client

In the waifu-daemon client sidebar:

1. Set **TTS 伺服器** to `http://<host>:9824/tts`.
2. Tick **啟用語音 (TTS)**.

Each assistant reply is then POSTed here, the returned audio auto-plays, and the
speech bubble stays visible for the audio's duration.

> The browser page must be served over **http** to call an **http** TTS endpoint
> (or both over https), otherwise the browser blocks it as mixed content.

### API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET`  | `/`       | — | service info JSON |
| `GET`  | `/health` | — | `status`, `device`, `sample_rate`, VRAM free/total |
| `POST` | `/tts`    | `{"text": "...", "seed": 123?}` | `audio/wav` (streamed, then the temp file is deleted) |

If `server.api_key` is set, send it as the `X-API-Key` request header.

---

## Configuration

There are two layers, highest priority first:

1. **Shell / `.env` env vars** (`VOXTTS_*`) — per-machine, **win** over the file.
2. **`config.yaml`** — the shared, committed defaults.

Keep **one shared `config.yaml`** and differ per-machine via `.env`. The `.env`
lives in the project root, is gitignored, and is loaded automatically at startup
(python-dotenv). Copy the template:

```bash
cp .env.example .env     # GB10 and Windows presets are inside, commented
```

### Full reference

Every field below is settable in `config.yaml` (under its section) **or** via the
env var. Bools accept `1/0`, `true/false`, `yes/no`, `on/off`.

#### `server:`

| `config.yaml` key | Env var | Default | Meaning |
|---|---|---|---|
| `host` | `VOXTTS_HOST` | `0.0.0.0` | Bind address |
| `port` | `VOXTTS_PORT` | `9824` | Listen port |
| `api_key` | `VOXTTS_API_KEY` | `null` | Require this as `X-API-Key`; blank = no auth |
| `cors_origins` | `VOXTTS_CORS_ORIGINS` | `["*"]` | Allowed browser origins (env = comma-separated) |

#### `model:`

| `config.yaml` key | Env var | Default | Meaning |
|---|---|---|---|
| `name` | `VOXTTS_MODEL` | `openbmb/VoxCPM2` | HF repo id or local path |
| `hf_home` | `VOXTTS_HF_HOME` | `null` | Weight cache dir (blank = `~/.cache/huggingface`) |
| `device` | `VOXTTS_DEVICE` | `null` | `cuda` / `cuda:0` / `cpu`; blank = auto |
| `load_denoiser` | `VOXTTS_LOAD_DENOISER` | `false` | Load the ZipEnhancer denoiser (extra VRAM/disk) |
| `optimize` | `VOXTTS_OPTIMIZE` | `true` | `torch.compile` the model — set `false` on few-SM GPUs (GB10) or Windows (no triton) |
| `cudnn_enabled` | `VOXTTS_CUDNN` | `true` | `false` bypasses cuDNN (see Troubleshooting) |
| `matmul_precision` | `VOXTTS_MATMUL_PRECISION` | `high` | `high`/`medium` = TF32 (faster), `highest` = full fp32 |

#### `voice:`

| `config.yaml` key | Env var | Default | Meaning |
|---|---|---|---|
| `prompt` | `VOXTTS_VOICE_PROMPT` | `(A gentle, soft … young girl's voice…)` | Voice-design description prepended to the text |
| `prompt_separator` | `VOXTTS_VOICE_PROMPT_SEPARATOR` | `" "` (a space) | String between `(prompt)` and the text — `""` = none (changes tokenization) |
| `reference_wav` | `VOXTTS_REFERENCE_WAV` | `null` | Path to a clip → voice **cloning** (most consistent identity) |
| `reference_text` | `VOXTTS_REFERENCE_TEXT` | `null` | Transcript of the clip (enables "ultimate cloning") |

#### `generation:`

| `config.yaml` key | Env var | Default | Meaning |
|---|---|---|---|
| `seed` | `VOXTTS_SEED` | `20240601` | Fixed RNG seed; blank = random each call |
| `cfg_value` | `VOXTTS_CFG_VALUE` | `2.0` | Guidance — higher = obeys the prompt more |
| `inference_timesteps` | `VOXTTS_INFERENCE_TIMESTEPS` | `10` | Diffusion steps — higher = better/slower |
| `normalize` | `VOXTTS_NORMALIZE` | `true` | Text normalizer (numbers/punctuation → spoken) |
| `denoise` | `VOXTTS_DENOISE` | `false` | Denoise output (needs `load_denoiser: true`) |
| `retry_badcase` | `VOXTTS_RETRY_BADCASE` | `true` | Auto re-roll "bad" outputs — set `false` for strict, reproducible results |
| `retry_badcase_max_times` | `VOXTTS_RETRY_BADCASE_MAX_TIMES` | `3` | Max retries |
| `retry_badcase_ratio_threshold` | `VOXTTS_RETRY_BADCASE_RATIO_THRESHOLD` | `6.0` | Bad-case detection threshold |
| `max_chars` | `VOXTTS_MAX_CHARS` | `400` | Hard cap on input length; blank = no cap |

`_or_none` fields (`api_key`, `hf_home`, `device`, `matmul_precision`,
`reference_wav`, `reference_text`, `seed`, `max_chars`) treat a **blank** env
value as "disable / None". `VOXTTS_VOICE_PROMPT` and `VOXTTS_VOICE_PROMPT_SEPARATOR`
keep exact whitespace — quote them in the shell.

### Example `.env` (GB10, tuned for a stable Chinese voice)

```ini
VOXTTS_HF_HOME=/home/dgx1/repo/tts-server/hf_cache
VOXTTS_CUDNN=0
VOXTTS_OPTIMIZE=0
VOXTTS_MATMUL_PRECISION=highest
VOXTTS_VOICE_PROMPT=(溫柔甜美、輕柔的少女聲線，語氣親暱)
VOXTTS_CFG_VALUE=3.0
VOXTTS_SEED=20240601
VOXTTS_RETRY_BADCASE=0
```

---

## Voice tuning guide

- **Make it follow the description more strongly:** raise `cfg_value`
  (2.0 → 3.0–4.0). If the voice ignores "female/soft", this is the lever.
- **More detail / fewer artefacts:** raise `inference_timesteps` (10 → 16–24).
  Slower, but steadier.
- **A reproducible voice:** keep a fixed `seed` **and** set
  `retry_badcase: false` (the auto-retry otherwise re-rolls nondeterministically).
- **A rock-solid identity across machines:** use voice **cloning** instead of
  relying on prompt+seed. Drop a clean 5–15 s clip at `voice.reference_wav` (and
  optionally its transcript at `voice.reference_text`); the timbre then comes
  from the audio, independent of GPU/precision.

---

## Reproducing a voice generated elsewhere (e.g. ComfyUI)

Same prompt + seed but a *different* voice almost always means the `generate()`
call isn't byte-identical to the other tool — not a hardware issue. On the same
GPU, matching every input reproduces the result. Check against the other tool:

1. **Exact text string.** Voice-design prepends `(description)`; even one space
   (`(desc)text` vs `(desc) text`) changes tokenization. Match `prompt_separator`,
   or set `voice.prompt: ""` and send the full `(description)text` yourself.
2. **`retry_badcase`** — set `false` on both sides (its default `true` re-rolls).
3. **`cfg_value`, `inference_timesteps`, `normalize`, `denoise`** — match all four.
4. **`seed`** — set the same value.

The service logs the exact call (`generate seed=… cfg=… steps=… normalize=…
denoise=… retry_badcase=… text='…'`) at INFO — diff that line against the other
tool's parameters to find the mismatch.

---

## Deploy on a GPU server (systemd)

A unit is provided at [`systemd/voxcpm-tts.service`](systemd/voxcpm-tts.service)
— **not installed automatically**:

```bash
sudo cp systemd/voxcpm-tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voxcpm-tts
journalctl -u voxcpm-tts -f          # follow logs
```

Adjust `User`, `WorkingDirectory`, `ExecStart`, and the `Environment=` lines for
your box (these mirror the `.env` vars). Then point the waifu client's TTS URL at
`http://<server>:9824/tts`.

---

## Troubleshooting

### `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` on load (e.g. GB10 Blackwell)

The model loads, the diffusion warmup runs, then it crashes on the first conv in
the AudioVAE. Cause: a **system cuDNN** in `/lib` whose sublibraries get
`dlopen`-ed ahead of the pip wheel's, mixing versions. Confirm:

```bash
ldconfig -p | grep -i cudnn        # system cuDNN in /lib/... → the culprit
```

**Preferred fix** — force torch to use the venv's bundled cuDNN:

```bash
# nvidia.cudnn is a namespace package (__file__ is None) — use __path__:
CUDNN_LIB=$(.venv/bin/python -c "import nvidia.cudnn,os;print(os.path.join(list(nvidia.cudnn.__path__)[0],'lib'))")
LD_LIBRARY_PATH="$CUDNN_LIB" python -m voxcpm_tts
```

`LD_LIBRARY_PATH` must be set **before** launch (the linker reads it at process
start) so it **can't** come from `.env` — put it in the systemd `Environment=`
line, or use the simpler fallback: **`VOXTTS_CUDNN=0`** to bypass cuDNN entirely
(convs use the native CUDA path — slightly slower, no crash).

### Same seed, different/garbled or wrong-gender voice

Not hardware noise. Voice-design + seed is fragile and VoxCPM's `retry_badcase`
re-rolls nondeterministically. Set `VOXTTS_RETRY_BADCASE=0`, keep a fixed seed,
and prefer `highest` matmul precision; for a guaranteed identity use a reference
clip. See [Voice tuning](#voice-tuning-guide) and the ComfyUI section above.

### Windows install errors (`*.exe` file-lock / `~ip` warnings)

The global `C:\PythonXX\Scripts` dir can lock console-script executables mid-
install. Install into a **fresh venv** (see above). The `Ignoring invalid
distribution ~ip` warning is a leftover from an interrupted global pip and is
harmless from inside a venv.

### Slow generation

Without triton, `torch.compile` is off (Windows always; set `VOXTTS_OPTIMIZE=0`
to skip the attempt). The RTX 3060 runs ~RTF 2–3 (a few seconds per short reply);
a Linux GPU server with triton is much faster.

---

## Notes

- Synthesized audio is never retained on disk.
- Voice-design output can vary run to run; the fixed seed pins it, a reference
  clip nails it.

## License

Licensed under the **Apache License, Version 2.0** — see [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE).

```
Copyright 2026 Weil <me@weils.net>
```

VoxCPM2 itself is also Apache-2.0 (© OpenBMB).
