# LocalFlow

A fully-local macOS clone of [Wispr Flow](https://wisprflow.ai): hold a global
hotkey, speak, release — and cleaned-up, punctuated text is pasted at the
cursor in whatever app is focused. Nothing leaves your machine.

- **Speech recognition:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (`small.en`, int8) on-device.
- **Cleanup brain:** a local LLM served by [Ollama](https://ollama.com)
  (`qwen2.5:7b`, falls back to `qwen2.5:3b`). Ollama only cleans/formats — it
  does not transcribe.
- **Injection:** clipboard swap + `Cmd+V` into the frontmost app, with the
  previous clipboard restored.

## Features

- **Push-to-talk dictation** — hold `Cmd+Option`, speak, release.
- **Cleanup magic** — removes fillers (um/uh), false starts, applies
  punctuation/capitalization, honors "scratch that" self-corrections, turns
  spoken enumerations into numbered lists.
- **Per-app formatting profiles** — casual for Slack/Discord, polished for
  email, verbatim (no reflow, no auto-caps) for terminals and code editors.
- **Command Mode** — select text and speak an instruction ("make this more
  formal", "translate this to German"), or start any dictation with
  *"voice command …"*. The result replaces the selection.
- **Personal dictionary** — terms are injected into the cleanup prompt so
  proper nouns are spelled right; corrections you make after pasting can be
  learned automatically (`observe_correction`).
- **Menubar tray** — recording indicator (🎙/🔴), enable/disable toggle,
  dictionary editor, quit.
- **VAD auto-stop** — recording ends automatically after ~0.9 s of silence.

## Install

Requires macOS and Python 3.11+ ([uv](https://docs.astral.sh/uv/) is the
easiest way to get one: `uv venv --python 3.12 .venv`).

```bash
# 1. The cleanup model
ollama pull qwen2.5:7b        # or qwen2.5:3b on smaller machines
ollama serve                  # if not already running

# 2. LocalFlow
python3 -m venv .venv && source .venv/bin/activate
pip install -e .              # first run of the app downloads whisper small.en (~460 MB)
```

### macOS permissions (required)

In **System Settings → Privacy & Security**, grant your terminal (or the app
you launch LocalFlow from):

- **Microphone** — to record speech.
- **Accessibility** — to send the synthetic `Cmd+V` keystroke.
- **Input Monitoring** — for the global hotkey listener.

## Usage

```bash
python -m localflow
```

A 🎙 icon appears in the menubar. Hold **Cmd+Option**, speak, release. The
icon turns 🔴 while recording. Roughly 1.5 s after release (models warm), the
cleaned text is pasted at your cursor.

> **Why not the Fn key like Wispr?** Many Mac keyboards never deliver a
> key-*down* event for Fn to userspace listeners (only sporadic key-ups),
> which makes Fn unusable for push-to-talk. Any combo of ordinary modifiers
> works: set e.g. `hotkey.combo: "ctrl+alt"` in your user config.
> `keytest.py` in the repo root logs 30 s of raw key events to diagnose what
> your keyboard sends.

### Configuration

Defaults live in `localflow/config.defaults.yaml`; override any subset in
`~/.config/localflow/config.yaml`:

```yaml
hotkey:
  combo: "ctrl+alt"
ollama:
  model: "qwen2.5:3b"
vad:
  enabled: false
```

The personal dictionary is a JSON list at
`~/.config/localflow/dictionary.json` (editable from the menubar).

## Tests

All external services and hardware (Ollama, microphone, keyboard,
NSWorkspace, rumps) are mocked; tests run headless:

```bash
pip install -e ".[dev]"
pytest -v
ruff check localflow tests
```

Tests that hit real services (whisper model download, live Ollama) are
skipped unless `RUN_LIVE=1` is set.

## Manual end-to-end checklist

These need a microphone and the macOS permission grants above, so they cannot
run in CI:

1. `ollama pull qwen2.5:7b` and make sure `ollama serve` is running.
2. `pip install -e .` (first run downloads the whisper `small.en` model).
3. Grant **Microphone**, **Accessibility**, and **Input Monitoring** to your terminal.
4. `python -m localflow` → the 🎙 menubar icon appears.
5. Focus TextEdit; hold `Cmd+Option`, say *"hey team um lets ship this on
   friday"*, release → **"Hey team, let's ship this on Friday."** is pasted.
6. Command Mode: select a sentence, hold the hotkey, say *"voice command make
   this more formal"* → the selection is rewritten formally.
7. Dictionary: add a proper noun (menubar → *Edit dictionary…*) → it is
   spelled correctly in the next dictation.
8. Per-app: dictate into Terminal or VS Code → text arrives verbatim, without
   chat-style punctuation or auto-capitalization.

## Architecture

```
hotkey (push-to-talk, Fn+Ctrl)
  └─> audio.Recorder (ring buffer, 500 ms pre-roll)
        └─> vad.SilenceDetector (auto-stop)
              └─> asr.WhisperEngine ──► Transcript
                    └─> context.detect() + dictionary + profiles
                          ├─> cleanup.OllamaCleaner        (dictation)
                          ├─> command_mode.CommandRunner   (selection / "voice command")
                          └─────► inject.ClipboardInjector (Cmd+V, clipboard restored)
tray.LocalFlowTray: 🎙/🔴 state, toggle, dictionary, quit
app.LocalFlowApp: session state machine wiring it all together
```

Every module codes against the dataclasses/protocols in
`localflow/contracts.py` and is testable in isolation.
