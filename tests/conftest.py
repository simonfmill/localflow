"""Shared fixtures: a bundled short WAV and a fake Ollama HTTP session."""

import numpy as np
import pytest
import requests
import soundfile as sf

SAMPLERATE = 16000
FIXTURE_SECONDS = 1.5


def make_sine(seconds=FIXTURE_SECONDS, freq=440.0, samplerate=SAMPLERATE, amplitude=0.3):
    t = np.linspace(0, seconds, int(samplerate * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="session")
def wav_fixture_path(tmp_path_factory):
    """A 1.5 s 440 Hz sine wave WAV — the bundled audio fixture."""
    path = tmp_path_factory.mktemp("fixtures") / "tone.wav"
    sf.write(path, make_sine(), SAMPLERATE)
    return path


@pytest.fixture
def wav_fixture_bytes(wav_fixture_path):
    return wav_fixture_path.read_bytes()


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeOllamaSession:
    """Stands in for requests.Session against Ollama's /api/chat.

    Models in ``fail_models`` return 404; models in ``error_models`` raise a
    ConnectionError; everything else answers with ``content``.
    """

    def __init__(self, content="Cleaned text.", fail_models=(), error_models=()):
        self.content = content
        self.fail_models = set(fail_models)
        self.error_models = set(error_models)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        model = json["model"]
        if model in self.error_models:
            raise requests.ConnectionError("connection refused")
        if model in self.fail_models:
            return FakeResponse(404, {"error": f"model '{model}' not found"})
        return FakeResponse(200, {"message": {"role": "assistant", "content": self.content}})


@pytest.fixture
def fake_ollama():
    return FakeOllamaSession()
