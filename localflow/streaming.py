"""Incremental transcription while recording (streaming ASR).

Audio blocks are fed into a background worker that repeatedly transcribes the
growing buffer. Segments that ended more than ``tail_s`` before the buffer end
are considered stable: their text is committed and their audio dropped. When
the user releases the hotkey, only the small uncommitted tail still needs
transcription — so key-release-to-text latency no longer grows with how long
the user spoke.
"""

import threading

import numpy as np


class StreamingTranscriber:
    def __init__(self, engine, samplerate=16000, chunk_s=3.0, tail_s=1.5):
        self.engine = engine
        self.samplerate = samplerate
        self.chunk_s = chunk_s
        self.tail_s = tail_s
        self.total_s = 0.0
        self._buffer = np.zeros(0, dtype=np.float32)
        self._committed: list = []
        self._hotwords = None
        self._processed_s = 0.0
        self._running = False
        self._lock = threading.Lock()
        self._proc_lock = threading.Lock()  # serializes partial passes
        self._wake = threading.Event()
        self._thread = None

    def start(self, hotwords=None):
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._committed = []
            self._processed_s = 0.0
            self.total_s = 0.0
            self._hotwords = hotwords
            self._running = True
        self._wake.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def feed(self, block):
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        with self._lock:
            if not self._running:
                return
            self._buffer = np.concatenate([self._buffer, block])
            self.total_s += block.size / self.samplerate
            pending = self._buffer.size / self.samplerate
        if pending - self._processed_s >= self.chunk_s:
            self._wake.set()

    def _worker(self):
        while True:
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            with self._lock:
                if not self._running:
                    return
                pending = self._buffer.size / self.samplerate
            if pending - self._processed_s >= self.chunk_s:
                self.process_pending()

    def process_pending(self):
        """Transcribe the pending buffer and commit its stable prefix."""
        with self._proc_lock:
            with self._lock:
                if not self._running:
                    return
                audio = self._buffer.copy()
            if audio.size < self.samplerate:  # < 1 s — not worth a pass yet
                return
            transcript = self.engine.transcribe(audio, hotwords=self._hotwords)
            cutoff = audio.size / self.samplerate - self.tail_s
            texts = []
            commit_end = 0.0
            for seg in transcript.segments:
                end = seg.get("end")
                if end is None or end > cutoff:
                    break
                text = (seg.get("text") or "").strip()
                if text:
                    texts.append(text)
                commit_end = end
            with self._lock:
                if not self._running:
                    return  # finish() took over while we were transcribing
                if texts:
                    self._committed.append(" ".join(texts))
                    self._buffer = self._buffer[int(commit_end * self.samplerate):]
                self._processed_s = self._buffer.size / self.samplerate

    def finish(self) -> str:
        """Stop streaming and return committed text plus the transcribed tail."""
        with self._lock:
            self._running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        with self._lock:
            audio = self._buffer
            self._buffer = np.zeros(0, dtype=np.float32)
            parts = list(self._committed)
            self._committed = []
        if audio.size >= int(0.2 * self.samplerate):
            tail = self.engine.transcribe(audio, hotwords=self._hotwords).text.strip()
            if tail:
                parts.append(tail)
        return " ".join(p for p in parts if p).strip()
