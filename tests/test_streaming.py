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


def make(chunk_s=3.0, tail_s=1.5):
    engine = FakeEngine()
    st = StreamingTranscriber(engine, samplerate=SR, chunk_s=chunk_s, tail_s=tail_s)
    return st, engine


def test_commits_stable_prefix_and_keeps_tail():
    st, engine = make(tail_s=1.5)
    st.start(hotwords="Qwen")
    st.feed(seconds(5))
    st.process_pending()
    # 5 s buffer, cutoff 3.5 s → segments ending at 1,2,3 are committed
    assert st._committed == ["seg0 seg1 seg2"]
    assert st._buffer.size == 2 * SR  # 3 s of audio dropped
    assert engine.calls[0]["hotwords"] == "Qwen"


def test_finish_appends_tail_and_resets():
    st, engine = make()
    st.start()
    st.feed(seconds(5))
    st.process_pending()
    result = st.finish()
    # committed prefix + transcription of the remaining 2 s tail
    assert result == "seg0 seg1 seg2 seg0 seg1"
    assert st._committed == []
    assert st._buffer.size == 0


def test_short_capture_skips_partial_passes():
    st, engine = make()
    st.start()
    st.feed(seconds(0.5))
    st.process_pending()  # < 1 s — no engine call
    assert engine.calls == []
    st.finish()
    assert len(engine.calls) == 1  # only the final tail pass


def test_tiny_tail_is_not_transcribed():
    st, engine = make()
    st.start()
    st.feed(seconds(0.1))
    assert st.finish() == ""
    assert engine.calls == []


def test_committed_text_becomes_context_for_next_passes():
    st, engine = make()
    st.start()
    st.feed(seconds(5))
    st.process_pending()
    assert engine.calls[0]["initial_prompt"] is None  # nothing committed yet
    st.feed(seconds(4))
    st.process_pending()
    assert engine.calls[1]["initial_prompt"] == "seg0 seg1 seg2"
    st.finish()
    assert engine.calls[-1]["initial_prompt"]  # tail pass gets context too


def test_total_seconds_track_fed_audio():
    st, _ = make()
    st.start()
    st.feed(seconds(2))
    st.feed(seconds(1.5))
    assert abs(st.total_s - 3.5) < 1e-6


def test_feed_after_finish_is_ignored():
    st, engine = make()
    st.start()
    st.feed(seconds(2))
    st.finish()
    st.feed(seconds(2))
    assert st._buffer.size == 0


def test_worker_thread_processes_in_background():
    st, engine = make(chunk_s=1.0)
    st.start()
    st.feed(seconds(4))
    st._wake.set()
    import time

    deadline = time.time() + 2
    while time.time() < deadline and not st._committed:
        time.sleep(0.02)
    assert st._committed  # worker picked up the pending chunk
    st.finish()


def test_restart_after_finish_works():
    st, _ = make()
    st.start()
    st.feed(seconds(5))
    st.process_pending()
    st.finish()
    st.start()
    st.feed(seconds(5))
    st.process_pending()
    assert st._committed == ["seg0 seg1 seg2"]
    assert st.finish() == "seg0 seg1 seg2 seg0 seg1"
