#!/bin/zsh
# Double-click to start LocalFlow voice dictation.
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$DIR/.venv/bin/python" ]; then
    echo "LocalFlow is not installed yet — double-click Install.command first."
    exit 1
fi
exec "$DIR/.venv/bin/python" -m localflow
