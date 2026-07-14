import os

import pytest

from localflow.cleanup import SYSTEM_PROMPT, OllamaCleaner
from localflow.contracts import CleanupRequest, CleanupResult

from conftest import FakeOllamaSession


def req(**kwargs):
    defaults = dict(raw_text="um hello world", dictionary=["Qwen", "Sarah"],
                    profile="Keep it casual.", context_hint="Pasting into Slack.")
    defaults.update(kwargs)
    return CleanupRequest(**defaults)


def test_system_prompt_has_worked_examples():
    assert SYSTEM_PROMPT.count("Input:") == 7
    assert SYSTEM_PROMPT.count("Output:") == 7
    assert "erstens" in SYSTEM_PROMPT  # German list example


def test_build_messages_includes_dictionary_profile_and_context():
    cleaner = OllamaCleaner(session=FakeOllamaSession())
    messages = cleaner.build_messages(req())
    system = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "Qwen, Sarah" in system
    assert "Keep it casual." in system
    assert "Pasting into Slack." in system
    assert messages[1] == {"role": "user", "content": "um hello world"}


def test_build_messages_omits_empty_sections():
    cleaner = OllamaCleaner(session=FakeOllamaSession())
    messages = cleaner.build_messages(req(dictionary=[], profile="", context_hint=""))
    system = messages[0]["content"]
    assert "Personal dictionary" not in system
    assert "Formatting profile" not in system


def test_clean_posts_to_ollama_and_parses_response():
    session = FakeOllamaSession(content="Hello world.")
    cleaner = OllamaCleaner(base_url="http://localhost:11434", model="qwen2.5:7b",
                            timeout_s=30, session=session)
    result = cleaner.clean(req())
    assert result == CleanupResult(text="Hello world.")
    call = session.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["json"]["model"] == "qwen2.5:7b"
    assert call["json"]["stream"] is False
    assert call["json"]["keep_alive"] == "30m"
    assert call["timeout"] == 30


def test_warmup_posts_a_tiny_request_and_swallows_errors():
    session = FakeOllamaSession()
    cleaner = OllamaCleaner(session=session, keep_alive="1h")
    cleaner.warmup()
    call = session.calls[0]["json"]
    assert call["model"] == "qwen2.5:7b"
    assert call["keep_alive"] == "1h"
    assert call["options"]["num_predict"] == 1

    dead = FakeOllamaSession(error_models={"qwen2.5:7b"})
    OllamaCleaner(session=dead).warmup()  # must not raise


def test_falls_back_to_secondary_model_on_404():
    session = FakeOllamaSession(content="Fallback answer.", fail_models={"qwen2.5:7b"})
    cleaner = OllamaCleaner(model="qwen2.5:7b", fallback_model="qwen2.5:3b", session=session)
    result = cleaner.clean(req())
    assert result.text == "Fallback answer."
    assert [c["json"]["model"] for c in session.calls] == ["qwen2.5:7b", "qwen2.5:3b"]


def test_fails_open_to_raw_text_when_ollama_unreachable():
    session = FakeOllamaSession(error_models={"qwen2.5:7b", "qwen2.5:3b"})
    cleaner = OllamaCleaner(session=session)
    result = cleaner.clean(req(raw_text="um keep this text"))
    assert result.text == "um keep this text"


@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1 for live Ollama test")
def test_live_cleanup():
    cleaner = OllamaCleaner()
    result = cleaner.clean(req(raw_text="um so hey team uh lets ship this on friday"))
    assert "um" not in result.text.lower().split()
