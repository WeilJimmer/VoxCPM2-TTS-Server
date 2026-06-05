#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Weil <me@weils.net>
"""Interactive VoxCPM2 parameter-tuning REPL.

Loads the model once and stays running so you can iterate on the voice without
reloading. Changes are in-memory only — your `.env` / `config.yaml` are never
touched. Every generated wav embeds the params it was made with in its metadata
(RIFF INFO comment), so you can always recover them from the file.

At the prompt, type either the explicit `key=value` form or a shorthand:

    seed=341329004      set the seed          (shorthand: a bare number)
    prompt=cheerful girl set the voice prompt  (shorthand: any other text)
    cfg=3.0             set cfg_value (prompt adherence; higher = obeys more)
    sample=這是要被念出來的文字   set the sentence that gets spoken

    /generate  /g       synthesize a wav (./tune_out/) with the current params
    /random    /r       roll a new random seed
    /show      /s       print the current params
    /help      /h       show this help
    /quit      /q       quit (or Ctrl+C / Ctrl+D)

Every change prints a timestamped line, e.g.  (14:03:51) set seed=1234.
Up/down arrows scroll input history — install prompt_toolkit for the best UX.
"""

import io
import re
import sys
import time
import random
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxcpm_tts.config import load_config          # noqa: E402
from voxcpm_tts.engine import VoxCPMEngine          # noqa: E402

DEFAULT_SAMPLE = "你好呀，我是你的老婆 Ariel，今天也要好好加油喔～"
OUT_DIR = PROJECT_ROOT / "tune_out"
HISTORY_FILE = PROJECT_ROOT / ".tune_history"
SEED_MAX = 2**31 - 1

# Explicit `key=value` assignment (keys are case-insensitive; `text` == `sample`).
ASSIGN_RE = re.compile(r"^(seed|prompt|cfg|sample|text)\s*=\s*(.*)$", re.IGNORECASE | re.DOTALL)


# ── timestamped logging ──────────────────────────────────────────────
def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"({_ts()}) {msg}")


# ── input parsing (pure, unit-testable) ──────────────────────────────
def parse_input(line: str):
    """Map a raw input line to an (action, value) pair:

    ("empty",  None)
    ("error",  message)
    ("cmd",    (name, arg|None))      a /command
    ("set",    (field, value))        field in {seed, cfg, prompt, sample}
    """
    s = line.strip()
    if not s:
        return ("empty", None)
    if s.startswith("/"):
        head, _, rest = s[1:].partition(" ")
        return ("cmd", (head.lower(), rest.strip() or None))

    m = ASSIGN_RE.match(s)
    if m:
        key = m.group(1).lower()
        val = m.group(2).strip()
        if key == "text":
            key = "sample"
        if key == "seed":
            try:
                return ("set", ("seed", int(val)))
            except ValueError:
                return ("error", f"seed must be an integer, got {val!r}")
        if key == "cfg":
            try:
                return ("set", ("cfg", float(val)))
            except ValueError:
                return ("error", f"cfg must be a number, got {val!r}")
        return ("set", (key, val))  # prompt / sample

    if re.fullmatch(r"-?\d+", s):          # shorthand: bare number → seed
        return ("set", ("seed", int(s)))
    return ("set", ("prompt", s))          # shorthand: anything else → prompt


def wrap_prompt(p: str) -> str:
    """Voice-design prompts must be parenthesized; wrap if the user omitted it."""
    p = p.strip()
    if p and not p.startswith("("):
        return f"({p})"
    return p


# ── input backend (history + arrow keys) ─────────────────────────────
def make_reader():
    """Return (read_fn, backend_name). read_fn(message) -> str; raises
    EOFError/KeyboardInterrupt on Ctrl-D/Ctrl-C."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        session = PromptSession(history=FileHistory(str(HISTORY_FILE)))
        return (lambda msg: session.prompt(msg)), "prompt_toolkit"
    except Exception:
        try:
            import readline  # noqa: F401  (enables history+arrows for input() on Unix)
            backend = "readline"
        except Exception:
            backend = "input (no history — pip install prompt_toolkit for arrows)"
        return (lambda msg: input(msg)), backend


# ── output: write wav + embed params in metadata ─────────────────────
def write_wav_with_meta(path: Path, wav_bytes: bytes, state: dict) -> int:
    """Re-encode the wav with the current params embedded in RIFF INFO tags."""
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")  # lossless decode
    comment = (
        f"seed={state['seed']}; cfg={state['cfg']}; "
        f"prompt={state['prompt']}; sample={state['sample']}"
    )
    meta = {
        "title": f"VoxCPM2 tune seed={state['seed']}",
        "artist": "VoxCPM2",
        "software": "voxcpm-tts tune.py",
        "date": time.strftime("%Y-%m-%d"),
        "comment": comment,
    }
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=1,
                      format="WAV", subtype="PCM_16") as f:
        for attr, val in meta.items():
            try:
                setattr(f, attr, val)
            except Exception:  # noqa: BLE001 - metadata is best-effort
                pass
        f.write(data)
    return len(wav_bytes)


# ── actions ──────────────────────────────────────────────────────────
def show(state: dict) -> None:
    log(f"seed   = {state['seed']}")
    log(f"cfg    = {state['cfg']}")
    log(f"prompt = {state['prompt']!r}")
    log(f"sample = {state['sample']!r}")


HELP = """\
set params (in-memory only — never writes .env/config):
  seed=1234            (or a bare number)      cfg=3.0
  prompt=cheerful girl (or any bare text)      sample=要被念出來的句子
commands:
  /generate /g   synthesize → ./tune_out/      /random /r   new random seed
  /show /s       show params                   /help /h     this help
  /quit /q       quit (or Ctrl+C)
each generated wav embeds its params in the file's metadata (comment tag)."""


def generate(engine: VoxCPMEngine, state: dict) -> None:
    # Inject current params in-memory only (never written to disk/config).
    engine.cfg.voice.prompt = state["prompt"]
    engine.cfg.generation.cfg_value = state["cfg"]
    log(f"generating… seed={state['seed']} cfg={state['cfg']} prompt={state['prompt']!r}")
    t0 = time.time()
    try:
        wav_bytes, sr = engine.synthesize(state["sample"], seed=state["seed"])
    except KeyboardInterrupt:
        log("generation interrupted")
        return
    except Exception as e:  # noqa: BLE001
        log(f"generation FAILED: {e}")
        return
    OUT_DIR.mkdir(exist_ok=True)
    fn = OUT_DIR / f"{time.strftime('%H%M%S')}_seed{state['seed']}.wav"
    nbytes = write_wav_with_meta(fn, wav_bytes, state)
    log(f"wrote {fn}  ({nbytes} bytes, {sr} Hz, {time.time() - t0:.1f}s) [params in metadata]")


# ── REPL ─────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config()
    engine = VoxCPMEngine(cfg)
    print("Loading VoxCPM2 … (first run downloads weights; Ctrl+C to abort)")
    try:
        engine.load()
    except KeyboardInterrupt:
        print("\naborted during load")
        return
    log(f"model ready on {engine.device}, sample_rate={engine.sample_rate} Hz")

    seed = cfg.generation.seed
    if seed is None:
        seed = random.randint(0, SEED_MAX)
    state = {
        "seed": seed,
        "cfg": cfg.generation.cfg_value,
        "prompt": cfg.voice.prompt,
        "sample": DEFAULT_SAMPLE,
    }
    show(state)
    print(HELP)

    read, backend = make_reader()
    log(f"input backend: {backend}")

    while True:
        try:
            line = read(f"\ntune[seed={state['seed']} cfg={state['cfg']}]> ")
        except (EOFError, KeyboardInterrupt):
            print()
            log("bye 👋")
            return

        action, value = parse_input(line)
        if action == "empty":
            continue
        if action == "error":
            log(value)
            continue
        if action == "set":
            field, v = value
            if field == "prompt":
                state["prompt"] = wrap_prompt(v)
                log(f"set prompt={state['prompt']!r}")
            elif field == "sample":
                state["sample"] = v
                log(f"set sample={v!r}")
            else:  # seed / cfg
                state[field] = v
                log(f"set {field}={v}")
        elif action == "cmd":
            name, arg = value
            if name in ("generate", "g"):
                generate(engine, state)
            elif name in ("random", "r"):
                state["seed"] = random.randint(0, SEED_MAX)
                log(f"set seed={state['seed']} (random)")
            elif name in ("show", "s"):
                show(state)
            elif name in ("help", "h", "?"):
                print(HELP)
            elif name in ("quit", "q", "exit"):
                log("bye 👋")
                return
            else:
                log(f"unknown command: /{name}  (try /help)")


if __name__ == "__main__":
    main()
