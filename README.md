# LocalFlow

A fully-local macOS dictation app in the spirit of [Wispr Flow](https://wisprflow.ai):
hold a hotkey, speak, release — and your words are pasted at the cursor in
whatever app is focused. Nothing leaves your machine.

The default setup is tuned for **speed and accuracy**: OpenAI's
`whisper-large-v3-turbo` running on the Apple GPU via
[mlx-whisper](https://github.com/ml-explore/mlx-examples) transcribes a
dictation of any length in **~2 seconds** flat, with punctuation and
capitalization built in, and pastes the transcript directly. Optional LLM
post-processing via [Ollama](https://ollama.com) is available for formatting
magic (see *Cleanup modes*).

## Quick install (Apple Silicon Mac)

1. **Download:** green **Code** button → *Download ZIP* → unzip (keep the
   folder wherever you want it to live).
2. **Install:** right-click `Install.command` → *Open* → *Open*
   (first time only — the file is unsigned). It installs a private
   Python and all dependencies; no admin password needed.
3. **Permissions:** System Settings → Privacy & Security → grant **Terminal**:
   *Microphone*, *Accessibility*, and *Input Monitoring*.
4. **Start:** double-click `Start.command`. The first start downloads the
   transcription model (~1.6 GB, one time).

Intel Macs work too — the installer automatically falls back to the CPU
backend with a smaller model.

## Usage

- **Dictate:** hold **⌘ Cmd + ⌥ Option**, speak (as long as you like),
  release. ~2 s later the text is pasted at your cursor. A small waveform
  pill at the bottom of the screen shows that recording is live.
- **Fix a misheard name:** press **⌃ Ctrl + ⇧ Shift + C** — your last
  dictation opens in TextEdit. Correct it, hit **Cmd+S**, close. LocalFlow
  learns the corrected names and recognizes them from then on.
- **Command Mode** (requires Ollama): start a dictation with
  *"voice command …"* to have an instruction executed instead of pasted,
  e.g. *"voice command translate this to English"* with text selected.
- **Quit:** press Ctrl+C in the Terminal window that `Start.command` opened
  (or just close it).

Autostart at login: System Settings → General → *Login Items & Extensions* →
add `Start.command` under *Open at Login*.

## Configuration

Defaults live in `localflow/config.defaults.yaml`; override any subset in
`~/.config/localflow/config.yaml`. The interesting dials:

```yaml
hotkey:
  combo: "cmd+alt"              # push-to-talk keys
  correction_combo: "ctrl+shift+c"
whisper:
  backend: "mlx"                # "mlx" (Apple GPU) or "ctranslate2" (CPU)
  model: "large-v3-turbo"       # or "small" for lower RAM/disk use
  language: "auto"              # pin to "de"/"en"/… to skip detection
cleanup:
  mode: "off"                   # see below
```

### Cleanup modes (`cleanup.mode`)

| Mode | What it does | Needs Ollama | Extra latency |
|---|---|---|---|
| `off` | paste Whisper's transcript directly (default) | no | none |
| `light` | remove fillers, fix punctuation | yes | ~1–2 s |
| `format` | verbatim words + numbered lists, email greeting/paragraph structure | yes | ~1–3 s |
| `full` | fillers, self-corrections, spoken lists, per-app tone | yes | ~1–3 s |

For the LLM modes: install [Ollama](https://ollama.com/download) and run
`ollama pull qwen2.5:7b`.

The personal dictionary is a JSON list at
`~/.config/localflow/dictionary.json`; its terms are fed to Whisper as
recognition hints and (in LLM modes) spelling rules.

## Troubleshooting

- **Hotkey does nothing:** another tool may own the combo (e.g. Rectangle
  uses many Ctrl+Option shortcuts — that's why correction defaults to
  Ctrl+Shift+C). Pick any other modifier combo in the config. The Fn key
  cannot be used: most Mac keyboards never deliver it to apps.
- **Nothing pastes:** re-check the Accessibility grant for Terminal, then
  fully quit and reopen Terminal.
- **Logs:** the Terminal window shows per-dictation timings; a copy is
  written to `~/Library/Logs/LocalFlow.log`.
- `keytest.py` logs 30 s of raw key events to diagnose keyboard issues.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,mlx]"
.venv/bin/python -m pytest -v      # all external services/hardware mocked
.venv/bin/ruff check localflow tests
```

Architecture: `hotkey → audio (ring buffer w/ pre-roll) → asr (MLX or
faster-whisper) → [optional Ollama cleanup / command mode] → inject
(clipboard + Cmd+V, restored)`. Every module codes against
`localflow/contracts.py` and ships its own test file. An experimental
chunked streaming mode exists (`pipeline.streaming`) for the CPU backend;
with MLX it is unnecessary because per-pass cost is flat.
