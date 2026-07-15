"""Incremental transcription while recording (streaming ASR).

Audio blocks are fed into a background worker. Whenever at least
``chunk_s + tail_s`` seconds are pending, the worker cuts a chunk of roughly
``chunk_s`` at the quietest point near its end (a likely speech gap),
transcribes it with the committed text as decoding context, and commits the
result. Work per pass is therefore bounded regardless of segment shapes, and
on key release only the small remaining tail needs transcription — so
release-to-text latency no longer grows with how long the user spoke.
"""

import threading

import numpy as np


class StreamingTranscriber:
    def __init__(self, engine, samplerate=16000, chunk_s=2.5, tail_s=1.2):
        self.engine = engine
        self.samplerate = samplerate
        self.chunk_s = chunk_s
        self.tail_s = tail_s
        self.total_s = 0.0
        self._buffer = np.zeros(0, dtype=np.float32)
        self._committed: list = []
        self._hotwords = None
        self._running = False
        self._lock = threading.Lock()
        self._proc_lock = threading.Lock()  # serializes partial passes
        self._wake = threading.Event()
        self._thread = None

    def start(self, hotwords=None):
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._committed = []
            self.total_s = 0.0
            self._hotwords = hotwords
            self._running = True
        self._wake.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def _threshold_s(self):
        return self.chunk_s + self.tail_s

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
        if pending >= self._threshold_s:
            self._wake.set()

    def _pending_s(self):
        with self._lock:
            return self._buffer.size / self.samplerate

    def _worker(self):
        while True:
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            with self._lock:
                if not self._running:
                    return
            # drain: the worker may lag behind fast speech, so keep cutting
            # chunks until the pending buffer is short again
            while self._running and self._pending_s() >= self._threshold_s:
                self.process_pending()

    def _quiet_cut(self, buffer) -> int:
        """Sample index near chunk_s with the least energy (a speech gap)."""
        sr = self.samplerate
        lo = max(0, int((self.chunk_s - 0.6) * sr))
        hi = min(int((self.chunk_s + 0.6) * sr), buffer.size)
        window = int(0.03 * sr)
        step = int(0.01 * sr)
        best, best_energy = hi, None
        segment = buffer[lo:hi]
        for offset in range(0, max(1, segment.size - window), step):
            energy = float(np.mean(np.abs(segment[offset:offset + window])))
            if best_energy is None or energy < best_energy:
                best_energy = energy
                best = lo + offset + window // 2
        return best

    def process_pending(self):
        """Cut one bounded chunk at a quiet point, transcribe and commit it."""
        with self._proc_lock:
            with self._lock:
                if not self._running:
                    return
                buffer = self._buffer
            if buffer.size / self.samplerate < self._threshold_s:
                return
            cut = self._quiet_cut(buffer)
            chunk = buffer[:cut].copy()
            transcript = self.engine.transcribe(chunk, hotwords=self._hotwords,
                                                initial_prompt=self._context())
            text = transcript.text.strip()
            with self._lock:
                if not self._running:
                    return  # finish() took over while we were transcribing
                if text:
                    self._committed.append(text)
                self._buffer = self._buffer[cut:]

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
            context = " ".join(parts)[-200:] if parts else None
            tail = self.engine.transcribe(audio, hotwords=self._hotwords,
                                          initial_prompt=context).text.strip()
            if tail:
                parts.append(tail)
        return " ".join(p for p in parts if p).strip()

    def _context(self):
        """Last ~200 chars of committed text — decoding context across cuts."""
        with self._lock:
            if not self._committed:
                return None
            return " ".join(self._committed)[-200:]
