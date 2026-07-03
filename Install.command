#!/bin/zsh
# LocalFlow installer — double-click me.
# Sets up a private Python, installs LocalFlow, and pulls the cleanup model.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "==> Installing LocalFlow into: $DIR"
echo

# 1. Ollama must be installed first (it serves the local cleanup LLM).
if ! command -v ollama >/dev/null 2>&1 && [ ! -d /Applications/Ollama.app ]; then
    echo "⚠️  Ollama is not installed yet."
    echo "    1. Download it from https://ollama.com/download (opening now)"
    echo "    2. Install and launch it once"
    echo "    3. Double-click Install.command again"
    open "https://ollama.com/download" 2>/dev/null || true
    exit 1
fi

# 2. uv — a small user-local tool that provides Python 3.12 (no admin needed).
UV="$HOME/.local/bin/uv"
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [ ! -x "$UV" ]; then
    echo "==> Installing uv (Python manager, user-local)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 3. Python environment + LocalFlow and its dependencies.
echo "==> Setting up Python 3.12 environment…"
"$UV" venv --clear --python 3.12 "$DIR/.venv"
echo "==> Installing LocalFlow (this takes a few minutes)…"
"$UV" pip install --python "$DIR/.venv/bin/python" -e "$DIR"

# 4. The cleanup model (one-time, ~4.7 GB; use qwen2.5:3b on 8 GB Macs).
if command -v ollama >/dev/null 2>&1; then
    echo "==> Downloading the cleanup model qwen2.5:7b (~4.7 GB, one time)…"
    ollama pull qwen2.5:7b || echo "    (skipped — run 'ollama pull qwen2.5:7b' later)"
else
    echo "⚠️  Launch the Ollama app once, then run: ollama pull qwen2.5:7b"
fi

echo
echo "✅ Installed. Two things left to do by hand:"
echo
echo "1. PERMISSIONS — System Settings → Privacy & Security → grant Terminal:"
echo "     • Microphone   • Accessibility   • Input Monitoring"
echo "   (macOS will prompt for the microphone on first recording.)"
echo
echo "2. START — double-click Start.command in this folder."
echo "   Hold Cmd+Option, speak, release → cleaned text pastes at your cursor."
echo "   (First start downloads the speech model, ~460 MB, one time.)"
