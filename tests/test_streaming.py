import numpy as np

from localflow.contracts import Transcript
from localflow.streaming import StreamingTranscriber

SR = 16000


class FakeEngine:
    """Deterministic fake: one segment per full second of audio, texts seg0…segN."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, hotwords=None, initial_prompt=None):
        self.calls.append({"seconds": len(audio) / SR, "hotwords": hotwords,
                           "initial_prompt": initial_prompt})
        n = int(len(audio) // SR)
        segments = [{"start": float(i), "end": float(i + 1), "text": f"seg{i}"}
                    for i in range(n)]
        return Transcript(text=" ".join(s["text"] for s in segments),
                          segments=segments, lang="en", duration_s=len(audio) / SR)


def seconds(n):
    return np.zeros(int(n * SR), dtype=np.float32)


def make(chunk_s=2.0, tail_s=1.0):
    engine = FakeEngine()
    st = StreamingTranscriber(engine, samplerate=SR, chunk_s=chunk_s, tail_s=tail_s)
    return st, engine


def test_commits_one_bounded_chunk():
    st, engine = make()  # threshold = 3.0 s
    st.start(hotwords="Qwen")
    st.feed(seconds(4))
    st.process_pending()
    assert len(st._committed) == 1
    assert st._committed[0].startswith("seg0")
    # cut lands within chunk_s ± 0.6 s → 1.4–2.6 s of audio removed
    assert 1.4 * SR <= st._buffer.size <= 2.6 * SR
    assert engine.calls[0]["hotwords"] == "Qwen"
    assert engine.calls[0]["seconds"] <= 2.6  # bounded work per pass
    st.finish()


def test_below_threshold_does_not_transcribe():
    st, engine = make()
    st.start()
    st.feed(seconds(2.9))
    st.process_pending()
    assert engine.calls == []
    st.finish()


def test_finish_appends_tail_and_resets():
    st, engine = make()
    st.start()
    st.feed(seconds(4))
    st.process_pending()
    committed = list(st._committed)
    result = st.finish()
    assert len(engine.calls) == 2  # one partial pass + one tail pass
    assert result.startswith(committed[0])
    assert len(result) > len(committed[0])  # tail text appended
    assert st._committed == []
    assert st._buffer.size == 0


def test_committed_text_becomes_decoding_context():
    st, engine = make()
    st.start()
    st.feed(seconds(4))
    st.process_pending()
    assert engine.calls[0]["initial_prompt"] is None  # nothing committed yet
    st.feed(seconds(2))
    st.process_pending()
    assert engine.calls[1]["initial_prompt"] == st._committed[0]
    st.finish()
    assert engine.calls[-1]["initial_prompt"]  # tail pass gets context too


def test_quiet_cut_prefers_the_silence_gap():
    st, _ = make(chunk_s=2.0)
    buffer = 0.5 * np.sin(np.linspace(0, 800, 4 * SR)).astype(np.float32)
    gap = int(1.8 * SR)
    buffer[gap:gap + int(0.05 * SR)] = 0.0  # 50 ms pause at 1.8 s
    cut = st._quiet_cut(buffer)
    assert abs(cut / SR - 1.8) < 0.1


def test_tiny_tail_is_not_transcribed():
    st, engine = make()
    st.start()
    st.feed(seconds(0.1))
    assert st.finish() == ""
    assert engine.calls == []


def test_total_seconds_track_fed_audio():
    st, _ = make()
    st.start()
    st.feed(seconds(2))
    st.feed(seconds(1.5))
    assert abs(st.total_s - 3.5) < 1e-6
    st.finish()


def test_feed_after_finish_is_ignored():
    st, _ = make()
    st.start()
    st.feed(seconds(2))
    st.finish()
    st.feed(seconds(2))
    assert st._buffer.size == 0


def test_worker_thread_processes_in_background():
    import time

    st, _ = make()
    st.start()
    st.feed(seconds(4))  # crosses the threshold → wakes the worker
    deadline = time.time() + 2
    while time.time() < deadline and not st._committed:
        time.sleep(0.02)
    assert st._committed
    st.finish()


def test_restart_after_finish_works():
    st, _ = make()
    st.start()
    st.feed(seconds(4))
    st.process_pending()
    first = st.finish()
    st.start()
    st.feed(seconds(4))
    st.process_pending()
    assert st.finish() == first
