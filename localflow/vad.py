"""Voice-activity detection for auto-stopping a capture.

Uses a simple RMS energy threshold: once speech has been heard for at least
``min_speech_ms``, a run of ``silence_ms`` of below-threshold audio signals
that the capture should stop. (The optional ten-vad package can replace this,
but the energy detector has no extra dependency and works well at 16 kHz.)
"""

import numpy as np


def rms(frame) -> float:
    frame = np.asarray(frame, dtype=np.float32).reshape(-1)
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame))))


class SilenceDetector:
    def __init__(self, samplerate=16000, energy_threshold=0.01, silence_ms=900, min_speech_ms=200):
        self.samplerate = samplerate
        self.energy_threshold = energy_threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.reset()

    def reset(self):
        self._speech_frames = 0
        self._silence_frames = 0

    def feed(self, frame) -> bool:
        """Feed one audio block; returns True when the capture should stop."""
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if rms(frame) >= self.energy_threshold:
            self._speech_frames += frame.size
            self._silence_frames = 0
        else:
            self._silence_frames += frame.size
        min_speech = self.samplerate * self.min_speech_ms / 1000
        silence_needed = self.samplerate * self.silence_ms / 1000
        return self._speech_frames >= min_speech and self._silence_frames >= silence_needed
