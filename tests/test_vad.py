import numpy as np

from localflow.vad import SilenceDetector, rms

SR = 16000


def speech_block(ms=100):
    return np.full(int(SR * ms / 1000), 0.3, dtype=np.float32)


def silence_block(ms=100):
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def test_rms():
    assert rms(np.zeros(100)) == 0.0
    assert abs(rms(np.full(100, 0.5)) - 0.5) < 1e-6
    assert rms(np.zeros(0)) == 0.0


def test_silence_only_never_triggers():
    det = SilenceDetector(samplerate=SR, silence_ms=300, min_speech_ms=200)
    for _ in range(50):
        assert det.feed(silence_block()) is False


def test_speech_then_silence_triggers():
    det = SilenceDetector(samplerate=SR, silence_ms=300, min_speech_ms=200)
    for _ in range(3):
        assert det.feed(speech_block()) is False
    assert det.feed(silence_block()) is False  # 100 ms silence
    assert det.feed(silence_block()) is False  # 200 ms
    assert det.feed(silence_block()) is True  # 300 ms → stop


def test_speech_resets_silence_run():
    det = SilenceDetector(samplerate=SR, silence_ms=300, min_speech_ms=200)
    for _ in range(3):
        det.feed(speech_block())
    det.feed(silence_block())
    det.feed(silence_block())
    det.feed(speech_block())  # speaker resumed — silence counter resets
    assert det.feed(silence_block()) is False
    assert det.feed(silence_block()) is False
    assert det.feed(silence_block()) is True


def test_reset_clears_state():
    det = SilenceDetector(samplerate=SR, silence_ms=300, min_speech_ms=200)
    for _ in range(3):
        det.feed(speech_block())
    det.reset()
    assert det.feed(silence_block()) is False
    assert det.feed(silence_block()) is False
    assert det.feed(silence_block()) is False  # no speech since reset
