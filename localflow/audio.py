"""Microphone capture with a pre-roll ring buffer.

The stream runs continuously so the ring buffer always holds the last
``preroll_ms`` of audio; speech that starts slightly before the hotkey press
is therefore not lost. The stream factory is injectable so tests never touch
real hardware.
"""

import io
import threading
from collections import deque

import numpy as np
import soundfile as sf


class Recorder:
    def __init__(self, samplerate=16000, channels=1, preroll_ms=500, stream_factory=None):
        self.samplerate = samplerate
        self.channels = channels
        self.preroll_frames = int(samplerate * preroll_ms / 1000)
        self.block_listeners: list = []  # called with each block while recording
        self._ring: deque = deque()
        self._ring_frames = 0
        self._chunks: list = []
        self._recording = False
        self._lock = threading.Lock()
        self._stream_factory = stream_factory or self._default_stream_factory
        self._stream = None

    def _default_stream_factory(self, callback):
        import sounddevice as sd

        return sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )

    def open(self):
        if self._stream is None:
            self._stream = self._stream_factory(self._on_audio)
            self._stream.start()

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream = None

    def _on_audio(self, indata, frames, time_info, status):
        block = np.asarray(indata, dtype=np.float32).reshape(-1, self.channels)[:, 0].copy()
        notify = []
        with self._lock:
            if self._recording:
                self._chunks.append(block)
                notify = list(self.block_listeners)
            else:
                self._ring.append(block)
                self._ring_frames += len(block)
                while self._ring and self._ring_frames - len(self._ring[0]) >= self.preroll_frames:
                    self._ring_frames -= len(self._ring.popleft())
        for cb in notify:
            cb(block)

    def start(self):
        """Begin a capture; the current ring-buffer contents become the pre-roll.

        Returns the pre-roll audio so streaming consumers can ingest it (block
        listeners only see blocks that arrive after this call).
        """
        self.open()
        with self._lock:
            self._chunks = list(self._ring)
            preroll = (np.concatenate(self._chunks) if self._chunks
                       else np.zeros(0, dtype=np.float32))
            self._ring.clear()
            self._ring_frames = 0
            self._recording = True
        return preroll

    def stop(self) -> np.ndarray:
        """End the capture and return the recorded mono float32 audio."""
        with self._lock:
            self._recording = False
            chunks = self._chunks
            self._chunks = []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    @property
    def is_recording(self) -> bool:
        return self._recording


def to_wav_bytes(audio: np.ndarray, samplerate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
