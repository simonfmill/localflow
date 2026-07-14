"""faster-whisper wrapper producing Transcript objects.

The model is loaded lazily; if the configured device fails (e.g. a GPU
config on a CPU-only machine) it falls back to int8 on CPU. The model
factory is injectable for tests.
"""

import numpy as np

from localflow.contracts import Transcript


class WhisperEngine:
    def __init__(self, model_name="small.en", device="auto", compute_type="int8",
                 language=None, cpu_threads=8, model_factory=None):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        # Pinning a language ("de", "en", …) skips the per-utterance
        # language-detection pass; None/"auto" keeps auto-detection.
        self.language = None if language in (None, "", "auto") else language
        self.cpu_threads = cpu_threads
        self._model_factory = model_factory
        self._model = None

    @staticmethod
    def _default_factory(model_name, device, compute_type, cpu_threads):
        from faster_whisper import WhisperModel

        return WhisperModel(model_name, device=device, compute_type=compute_type,
                            cpu_threads=cpu_threads)

    def load(self):
        if self._model is None:
            factory = self._model_factory or self._default_factory
            try:
                self._model = factory(self.model_name, self.device, self.compute_type,
                                      self.cpu_threads)
            except Exception:
                self._model = factory(self.model_name, "cpu", "int8", self.cpu_threads)
        return self._model

    def warmup(self):
        """Force model download/load and prime caches with half a second of silence."""
        self.transcribe(np.zeros(int(0.5 * 16000), dtype=np.float32))

    def transcribe(self, wav, hotwords=None) -> Transcript:
        """Transcribe 16 kHz mono audio (float32 ndarray or WAV bytes).

        hotwords: personal-dictionary terms biased during decoding, so names
        and jargon are recognized instead of being fixed after the fact.
        """
        audio = self._to_audio(wav)
        kwargs = dict(beam_size=1, language=self.language,
                      condition_on_previous_text=False)
        if hotwords:
            kwargs["hotwords"] = hotwords
        try:
            segments, info = self.load().transcribe(audio, **kwargs)
        except TypeError:  # faster-whisper too old for hotwords
            kwargs.pop("hotwords", None)
            segments, info = self.load().transcribe(audio, **kwargs)
        seg_list = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        text = " ".join(s["text"].strip() for s in seg_list).strip()
        duration = float(getattr(info, "duration", audio.size / 16000))
        lang = getattr(info, "language", "en") or "en"
        return Transcript(text=text, segments=seg_list, lang=lang, duration_s=duration)

    @staticmethod
    def _to_audio(wav) -> np.ndarray:
        if isinstance(wav, (bytes, bytearray)):
            import io

            import soundfile as sf

            data, _sr = sf.read(io.BytesIO(bytes(wav)), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            return np.ascontiguousarray(data, dtype=np.float32)
        return np.asarray(wav, dtype=np.float32).reshape(-1)
