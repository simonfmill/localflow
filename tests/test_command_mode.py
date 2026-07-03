from localflow.command_mode import (
    COMMAND_SYSTEM_PROMPT,
    CommandRunner,
    is_command,
    strip_trigger,
)
from localflow.contracts import CommandRequest

from conftest import FakeOllamaSession


def test_selection_routes_to_command():
    assert is_command("make this more formal", selection="hello there") is True


def test_trigger_phrase_routes_to_command():
    assert is_command("voice command translate this to german") is True
    assert is_command("Command mode, rewrite the last sentence") is True
    assert is_command("hey flow what is the capital of france") is True


def test_plain_dictation_is_not_command():
    assert is_command("hey team lets ship this on friday") is False
    assert is_command("hey team lets ship this on friday", selection=None) is False
    assert is_command("hey team", selection="") is False


def test_strip_trigger():
    assert strip_trigger("voice command translate this") == "translate this"
    assert strip_trigger("Voice command, make it formal") == "make it formal"
    assert strip_trigger("no trigger here") == "no trigger here"


def test_run_builds_prompt_with_selection():
    session = FakeOllamaSession(content="Sehr geehrte Damen und Herren")
    runner = CommandRunner(session=session)
    result = runner.run(CommandRequest(instruction="translate to german",
                                       selection="dear sir or madam"))
    assert result == "Sehr geehrte Damen und Herren"
    call = session.calls[0]["json"]
    assert call["messages"][0]["content"] == COMMAND_SYSTEM_PROMPT
    user = call["messages"][1]["content"]
    assert "translate to german" in user
    assert "SELECTION:\ndear sir or madam" in user


def test_run_without_selection():
    session = FakeOllamaSession(content="Paris")
    runner = CommandRunner(session=session)
    result = runner.run(CommandRequest(instruction="what is the capital of france"))
    assert result == "Paris"
    assert "SELECTION" not in session.calls[0]["json"]["messages"][1]["content"]


def test_run_falls_back_then_fails_closed():
    session = FakeOllamaSession(content="ok", fail_models={"qwen2.5:7b"})
    runner = CommandRunner(model="qwen2.5:7b", fallback_model="qwen2.5:3b", session=session)
    assert runner.run(CommandRequest(instruction="x")) == "ok"
    assert [c["json"]["model"] for c in session.calls] == ["qwen2.5:7b", "qwen2.5:3b"]

    dead = FakeOllamaSession(error_models={"qwen2.5:7b", "qwen2.5:3b"})
    runner = CommandRunner(session=dead)
    assert runner.run(CommandRequest(instruction="x", selection="y")) == ""
