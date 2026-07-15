#!/bin/zsh
# LocalFlow installer — double-click me.
# Sets up a private Python and installs LocalFlow with the fast GPU backend.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "==> Installing LocalFlow into: $DIR"
echo

# 1. uv — a small user-local tool that provides Python 3.12 (no admin needed).
UV="$HOME/.local/bin/uv"
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [ ! -x "$UV" ]; then
    echo "==> Installing uv (Python manager, user-local)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 2. Python environment + LocalFlow. On Apple Silicon, include the MLX
#    GPU backend (much faster and more accurate transcription).
echo "==> Setting up Python 3.12 environment…"
"$UV" venv --clear --python 3.12 "$DIR/.venv"
echo "==> Installing LocalFlow (this takes a few minutes)…"
if [ "$(uname -m)" = "arm64" ]; then
    "$UV" pip install --python "$DIR/.venv/bin/python" -e "$DIR[mlx]"
else
    echo "    (Intel Mac detected — using the CPU transcription backend.)"
    "$UV" pip install --python "$DIR/.venv/bin/python" -e "$DIR"
    mkdir -p "$HOME/.config/localflow"
    if [ ! -f "$HOME/.config/localflow/config.yaml" ]; then
        printf 'whisper:\n  backend: "ctranslate2"\n  model: "small"\n' \
            > "$HOME/.config/localflow/config.yaml"
    fi
fi

# 3. Optional: Ollama powers the LLM extras (cleanup modes, Command Mode).
#    The default setup pastes Whisper's transcript directly and does not
#    need it.
if ! command -v ollama >/dev/null 2>&1 && [ ! -d /Applications/Ollama.app ]; then
    echo
    echo "ℹ️  Ollama not found — that's fine for the default setup."
    echo "   If you later want LLM cleanup or Command Mode: install it from"
    echo "   https://ollama.com/download and run: ollama pull qwen2.5:7b"
fi

echo
echo "✅ Installed. Two things left to do by hand:"
echo
echo "1. PERMISSIONS — System Settings → Privacy & Security → grant Terminal:"
echo "     • Microphone   • Accessibility   • Input Monitoring"
echo "   (macOS will prompt for the microphone on first recording.)"
echo
echo "2. START — double-click Start.command in this folder."
echo "   The FIRST start downloads the transcription model (~1.6 GB, one time)."
echo "   Then: hold Cmd+Option, speak, release → your words paste at the cursor."
echo "   Fix a misheard name: press Ctrl+Shift+C, edit, save with Cmd+S."
