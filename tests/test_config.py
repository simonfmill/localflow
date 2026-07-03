from localflow.config import _deep_merge, load_config


def test_defaults_load():
    cfg = load_config(user_path="/nonexistent/config.yaml")
    assert cfg["hotkey"]["combo"] == "cmd+alt"
    assert cfg["whisper"]["model"] == "small"
    assert cfg["ollama"]["model"] == "qwen2.5:7b"
    assert cfg["ollama"]["fallback_model"] == "qwen2.5:3b"
    assert cfg["ollama"]["base_url"] == "http://localhost:11434"
    assert cfg["audio"]["preroll_ms"] == 500
    assert cfg["vad"]["enabled"] is True


def test_user_override_merges_deeply(tmp_path):
    user = tmp_path / "config.yaml"
    user.write_text("ollama:\n  model: qwen2.5:3b\nhotkey:\n  combo: ctrl+alt\n")
    cfg = load_config(user_path=user)
    assert cfg["ollama"]["model"] == "qwen2.5:3b"
    # untouched sibling keys survive the merge
    assert cfg["ollama"]["base_url"] == "http://localhost:11434"
    assert cfg["hotkey"]["combo"] == "ctrl+alt"
    assert cfg["audio"]["samplerate"] == 16000


def test_deep_merge_replaces_scalars_and_merges_dicts():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 9}, "b": 4}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 9}, "b": 4}
    assert base["a"]["y"] == 2  # base not mutated
