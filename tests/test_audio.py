import numpy as np
import soundfile as sf

from localflow.audio import Recorder, to_wav_bytes


class FakeStream:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def make_recorder(preroll_ms=500):
    stream = FakeStream()
    rec = Recorder(samplerate=16000, channels=1, preroll_ms=preroll_ms,
                   stream_factory=lambda cb: stream)
    return rec, stream


def block(n=1600, value=0.5):
    return np.full((n, 1), value, dtype=np.float32)


def test_open_starts_stream_once():
    rec, stream = make_recorder()
    rec.open()
    rec.open()
    assert stream.started


def test_ring_buffer_trims_to_preroll():
    rec, _ = make_recorder(preroll_ms=500)  # 8000 frames
    rec.open()
    for _ in range(20):  # 20 * 1600 = 32000 frames fed while idle
        rec._on_audio(block(1600), 1600, None, None)
    buffered = sum(len(b) for b in rec._ring)
    assert 8000 <= buffered <= 8000 + 1600


def test_recording_includes_preroll():
    rec, _ = make_recorder(preroll_ms=100)  # 1600 frames preroll
    rec.open()
    rec._on_audio(block(1600, 0.1), 1600, None, None)  # idle: lands in ring
    rec.start()
    rec._on_audio(block(1600, 0.9), 1600, None, None)
    audio = rec.stop()
    assert len(audio) == 3200
    assert np.isclose(audio[0], 0.1)
    assert np.isclose(audio[-1], 0.9)


def test_stop_without_audio_returns_empty():
    rec, _ = make_recorder()
    rec.open()
    rec.start()
    audio = rec.stop()
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 0


def test_block_listeners_fire_only_while_recording():
    rec, _ = make_recorder()
    rec.open()
    seen = []
    rec.block_listeners.append(lambda b: seen.append(len(b)))
    rec._on_audio(block(), 1600, None, None)  # idle — no callback
    assert seen == []
    rec.start()
    rec._on_audio(block(), 1600, None, None)
    assert seen == [1600]


def test_to_wav_bytes_roundtrip():
    audio = (0.25 * np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
    wav = to_wav_bytes(audio, 16000)
    import io

    decoded, sr = sf.read(io.BytesIO(wav), dtype="float32")
    assert sr == 16000
    assert np.allclose(decoded, audio, atol=1e-3)  # PCM_16 quantization
