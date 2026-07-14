import os
from types import SimpleNamespace

import numpy as np
import pytest

from localflow.asr import WhisperEngine
from localflow.contracts import Transcript


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, beam_size=1, language=None,
                   condition_on_previous_text=True, hotwords=None):
        self.calls.append({"audio": audio, "beam_size": beam_size,
                           "language": language, "hotwords": hotwords})
        segments = iter([
            SimpleNamespace(start=0.0, end=1.0, text=" hello"),
            SimpleNamespace(start=1.0, end=2.0, text="world "),
        ])
        info = SimpleNamespace(language="en", duration=2.0)
        return segments, info


def test_transcribe_builds_transcript():
    model = FakeModel()
    engine = WhisperEngine(model_factory=lambda *a: model)
    result = engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert isinstance(result, Transcript)
    assert result.text == "hello world"
    assert len(result.segments) == 2
    assert result.segments[0]["start"] == 0.0
    assert result.lang == "en"
    assert result.duration_s == 2.0


def test_model_loaded_once():
    created = []

    def factory(name, device, compute, cpu_threads):
        created.append((name, device, compute, cpu_threads))
        return FakeModel()

    engine = WhisperEngine(model_name="small.en", device="auto", compute_type="int8",
                           cpu_threads=8, model_factory=factory)
    engine.transcribe(np.zeros(16000, dtype=np.float32))
    engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert created == [("small.en", "auto", "int8", 8)]


def test_falls_back_to_cpu_int8():
    attempts = []

    def factory(name, device, compute, cpu_threads):
        attempts.append((device, compute))
        if device != "cpu":
            raise RuntimeError("no GPU available")
        return FakeModel()

    engine = WhisperEngine(device="cuda", compute_type="float16", model_factory=factory)
    engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert attempts == [("cuda", "float16"), ("cpu", "int8")]


def test_language_auto_passes_none_pinned_passes_code():
    model = FakeModel()
    engine = WhisperEngine(language="auto", model_factory=lambda *a: model)
    engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert model.calls[0]["language"] is None
    pinned = FakeModel()
    engine = WhisperEngine(language="de", model_factory=lambda *a: pinned)
    engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert pinned.calls[0]["language"] == "de"


def test_hotwords_are_passed_to_the_model():
    model = FakeModel()
    engine = WhisperEngine(model_factory=lambda *a: model)
    engine.transcribe(np.zeros(16000, dtype=np.float32), hotwords="Qwen, Sarah")
    assert model.calls[0]["hotwords"] == "Qwen, Sarah"
    engine.transcribe(np.zeros(16000, dtype=np.float32))
    assert model.calls[1]["hotwords"] is None


def test_hotwords_dropped_for_old_faster_whisper():
    class OldModel(FakeModel):
        def transcribe(self, audio, beam_size=1, language=None,
                       condition_on_previous_text=True, **kwargs):
            if "hotwords" in kwargs:
                raise TypeError("unexpected keyword argument 'hotwords'")
            return super().transcribe(audio, beam_size, language,
                                      condition_on_previous_text)

    engine = WhisperEngine(model_factory=lambda *a: OldModel())
    result = engine.transcribe(np.zeros(16000, dtype=np.float32), hotwords="Qwen")
    assert result.text == "hello world"  # falls back, still transcribes


def test_accepts_wav_bytes(wav_fixture_bytes):
    model = FakeModel()
    engine = WhisperEngine(model_factory=lambda *a: model)
    engine.transcribe(wav_fixture_bytes)
    audio = model.calls[0]["audio"]
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == int(1.5 * 16000)


@pytest.mark.skipif(not os.environ.get("RUN_LIVE"), reason="set RUN_LIVE=1 for live ASR test")
def test_live_transcription(wav_fixture_bytes):
    engine = WhisperEngine()
    result = engine.transcribe(wav_fixture_bytes)
    assert isinstance(result, Transcript)  # a pure tone yields empty/near-empty text
