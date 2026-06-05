#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Weil <me@weils.net>
"""Interactive VoxCPM2 parameter-tuning REPL.

Loads the model once and stays running so you can iterate on the voice without
reloading. Changes are in-memory only — your `.env` / `config.yaml` are never
touched.

At the prompt, type:

    <number>            set the seed            e.g.  341329004
    <any other text>    set the voice prompt    e.g.  少女音, cheerful girl
    /generate  /g       synthesize a wav with the current params
    /random    /r       roll a new random seed
    /text ...  /t ...   set the sentence that gets spoken
    /show      /s       print the current params
    /help      /h       show this help
    /quit      /q       quit (or Ctrl+C / Ctrl+D)

Every change prints a timestamped line, e.g.  (14:03:51) set seed=1234.
Generated files go to ./tune_out/ . Input history (up/down arrows) is kept; for
the best experience install prompt_toolkit:  pip install prompt_toolkit
"""

import re
import sys
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxcpm_tts.config import load_config          # noqa: E402
from voxcpm_tts.engine import VoxCPMEngine          # noqa: E402

DEFAULT_TEXT = "你好呀，我是你的老婆 Ariel，今天也要好好加油喔～"
OUT_DIR = PROJECT_ROOT / "tune_out"
HISTORY_FILE = PROJECT_ROOT / ".tune_history"
SEED_MAX = 2**31 - 1


# ── timestamped logging ──────────────────────────────────────────────
def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"({_ts()}) {msg}")


# ── input parsing (pure, unit-testable) ──────────────────────────────
def parse_input(line: str):
    """Map a raw input line to an (action, value) pair:

    ("empty",  None)              blank line
    ("cmd",    (name, arg|None))  a /command (name lowercased, no slash)
    ("seed",   int)               a bare integer
    ("prompt", str)               anything else
    """
    s = line.strip()
    if not s:
        return ("empty", None)
    if s.startswith("/"):
        head, _, rest = s[1:].partition(" ")
        return ("cmd", (head.lower(), rest.strip() or None))
    if re.fullmatch(r"-?\d+", s):
        return ("seed", int(s))
    return ("prompt", s)


def wrap_prompt(p: str) -> str:
    """Voice-design prompts must be parenthesized; wrap if the user omitted it."""
    p = p.strip()
    if p and not p.startswith("("):
        return f"({p})"
    return p


# ── input backend (history + arrow keys) ─────────────────────────────
def make_reader():
    """Return (read_fn, backend_name). read_fn(message) -> str (raises
    EOFError/KeyboardInterrupt on Ctrl-D/Ctrl-C)."""
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


# ── actions ──────────────────────────────────────────────────────────
def show(state: dict) -> None:
    log(f"seed   = {state['seed']}")
    log(f"prompt = {state['prompt']!r}")
    log(f"text   = {state['text']!r}")


HELP = """\
commands:
  <number>          set seed              <any text>   set voice prompt
  /generate /g      synthesize a wav      /random /r   new random seed
  /text ... /t ...  set spoken sentence   /show /s     show current params
  /help /h          this help             /quit /q     quit (or Ctrl+C)"""


def generate(engine: VoxCPMEngine, state: dict) -> None:
    # Inject the current params in-memory only (never written to disk).
    engine.cfg.voice.prompt = state["prompt"]
    log(f"generating… seed={state['seed']} prompt={state['prompt']!r}")
    t0 = time.time()
    try:
        wav, sr = engine.synthesize(state["text"], seed=state["seed"])
    except KeyboardInterrupt:
        log("generation interrupted")
        return
    except Exception as e:  # noqa: BLE001
        log(f"generation FAILED: {e}")
        return
    OUT_DIR.mkdir(exist_ok=True)
    fn = OUT_DIR / f"{time.strftime('%H%M%S')}_seed{state['seed']}.wav"
    fn.write_bytes(wav)
    log(f"wrote {fn}  ({len(wav)} bytes, {sr} Hz, {time.time() - t0:.1f}s)")


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
    state = {"seed": seed, "prompt": cfg.voice.prompt, "text": DEFAULT_TEXT}
    show(state)
    print(HELP)

    read, backend = make_reader()
    log(f"input backend: {backend}")

    while True:
        try:
            line = read(f"\ntune[seed={state['seed']}]> ")
        except (EOFError, KeyboardInterrupt):
            print()
            log("bye 👋")
            return

        action, value = parse_input(line)
        if action == "empty":
            continue
        if action == "seed":
            state["seed"] = value
            log(f"set seed={value}")
        elif action == "prompt":
            state["prompt"] = wrap_prompt(value)
            log(f"set prompt={state['prompt']!r}")
        elif action == "cmd":
            name, arg = value
            if name in ("generate", "g"):
                generate(engine, state)
            elif name in ("random", "r"):
                state["seed"] = random.randint(0, SEED_MAX)
                log(f"set seed={state['seed']} (random)")
            elif name in ("text", "t"):
                if arg:
                    state["text"] = arg
                    log(f"set text={arg!r}")
                else:
                    log(f"text = {state['text']!r}")
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
