"""Orchestrator: wires hotkey → audio → vad → asr → context/profiles/dictionary
→ cleanup or command mode → inject into a push-to-talk session state machine."""

import logging
import threading
import time

from localflow import command_mode, profiles
from localflow.contracts import CleanupRequest, CommandRequest
from localflow.vad import rms

log = logging.getLogger("localflow")

IDLE = "idle"
RECORDING = "recording"
PROCESSING = "processing"


class LocalFlowApp:
    def __init__(self, *, recorder, asr, cleaner, commander, injector,
                 context_provider, dictionary, hotkey, tray=None, vad=None,
                 overlay=None, streamer=None, correction_combo=None,
                 editor_opener=None, selection_provider=None, min_duration_s=0.3,
                 samplerate=16000, threaded=True):
        self.recorder = recorder
        self.asr = asr
        self.cleaner = cleaner
        self.commander = commander
        self.injector = injector
        self.context_provider = context_provider
        self.dictionary = dictionary
        self.hotkey = hotkey
        self.tray = tray
        self.vad = vad
        self.overlay = overlay
        self.streamer = streamer
        self.correction_combo = correction_combo
        self._editor_opener = editor_opener or self._default_editor_opener
        self.selection_provider = selection_provider
        self.min_duration_s = min_duration_s
        self.samplerate = samplerate
        self.last_pasted: str | None = None
        self._threaded = threaded
        self._state = IDLE
        self._lock = threading.Lock()
        self.recorder.block_listeners.append(self._on_block)

    @staticmethod
    def _default_editor_opener(path):
        import subprocess

        subprocess.Popen(["open", "-t", path])

    def _request_correction(self):
        """Hotkey handler: open the last dictation in a text editor to learn fixes.

        Uses a plain file + default editor because on some macOS installs this
        unbundled process cannot display its own dialogs or status items.
        """
        last = self.last_pasted
        if not last:
            log.info("correction requested but nothing has been dictated yet")
            return
        log.info("correction hotkey pressed — opening editor")
        threading.Thread(target=self._correction_file_flow, args=(last,),
                         daemon=True).start()

    def _correction_file_flow(self, last, poll_interval=1.0, timeout_s=180):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.gettempdir()) / "localflow-correction.txt"
        try:
            path.write_text(last)
            self._editor_opener(str(path))
        except Exception:
            log.exception("could not open the correction editor")
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(poll_interval)
            try:
                current = path.read_text().strip()
            except OSError:
                continue
            if current and current != last.strip():
                added = self.dictionary.observe_correction(last, current)
                log.info("dictionary learned: %s",
                         ", ".join(added) if added else "nothing new")
                return
        log.info("correction editor closed without changes (timed out)")

    @property
    def state(self) -> str:
        return self._state

    @property
    def enabled(self) -> bool:
        return self.tray.enabled if self.tray is not None else True

    def _on_hotkey_press(self):
        try:
            if not self.enabled:
                return
            with self._lock:
                if self._state != IDLE:
                    return
                self._state = RECORDING
            log.info("hotkey pressed — recording")
            if self.vad is not None:
                self.vad.reset()
            if self.streamer is not None:
                self.streamer.start(hotwords=self._hotwords())
            preroll = self.recorder.start()
            if self.streamer is not None and preroll is not None and len(preroll):
                self.streamer.feed(preroll)
            if self.tray is not None:
                self.tray.set_recording(True)
            if self.overlay is not None:
                self.overlay.show()
        except Exception:
            log.exception("failed to start recording")
            if self.overlay is not None:
                self.overlay.hide()
            with self._lock:
                self._state = IDLE

    def _on_hotkey_release(self):
        try:
            with self._lock:
                if self._state != RECORDING:
                    return
                self._state = PROCESSING
            audio = self.recorder.stop()
            log.info("hotkey released — %.2fs of audio captured",
                     len(audio) / self.samplerate)
            if self.tray is not None:
                self.tray.set_recording(False)
            if self.overlay is not None:
                self.overlay.hide()
        except Exception:
            log.exception("failed to stop recording")
            if self.overlay is not None:
                self.overlay.hide()
            with self._lock:
                self._state = IDLE
            return
        if self._threaded:
            threading.Thread(target=self._process_safe, args=(audio,), daemon=True).start()
        else:
            self._process_safe(audio)

    def _process_safe(self, audio):
        try:
            if self.streamer is not None:
                t0 = time.perf_counter()
                text = self.streamer.finish().strip()
                log.info("transcript (streaming, tail %.2fs): %r",
                         time.perf_counter() - t0, text)
                if text and self.streamer.total_s >= self.min_duration_s:
                    self.deliver(text)
            else:
                self.process(audio)
        except Exception:
            log.exception("pipeline failed")
        finally:
            with self._lock:
                self._state = IDLE

    def _hotwords(self):
        return ", ".join(self.dictionary.terms) or None

    def _on_block(self, block):
        """Audio-thread callback: overlay levels, streaming ASR, VAD auto-stop."""
        if self._state != RECORDING:
            return
        if self.overlay is not None:
            self.overlay.feed(rms(block))
        if self.streamer is not None:
            self.streamer.feed(block)
        if self.vad is not None and self.vad.feed(block):
            self._on_hotkey_release()

    def process(self, audio):
        if len(audio) < self.min_duration_s * self.samplerate:
            log.info("capture too short (< %.1fs) — dropped", self.min_duration_s)
            return
        t0 = time.perf_counter()
        transcript = self.asr.transcribe(audio, hotwords=self._hotwords())
        text = transcript.text.strip()
        log.info("transcript (asr %.2fs): %r", time.perf_counter() - t0, text)
        if not text:
            return
        self.deliver(text)

    def deliver(self, text):
        ctx = self.context_provider()
        selection = self.selection_provider() if self.selection_provider else None
        t1 = time.perf_counter()
        if command_mode.is_command(text, selection):
            result = self.commander.run(
                CommandRequest(instruction=command_mode.strip_trigger(text),
                               selection=selection)
            )
        elif self.cleaner is None:  # cleanup off: paste the transcript as-is
            result = text
        else:
            req = CleanupRequest(
                raw_text=text,
                dictionary=list(self.dictionary.terms),
                profile=profiles.fragment_for(ctx.kind),
                context_hint=f"The text will be pasted into {ctx.app_name}." if ctx.app_name else "",
            )
            result = self.cleaner.clean(req).text
        if result:
            log.info("pasting into %s (%s) (llm %.2fs): %r",
                     ctx.app_name or "?", ctx.kind, time.perf_counter() - t1, result)
            self.injector.paste(result, ctx)
            self.last_pasted = result

    def run(self):
        if hasattr(self.cleaner, "warmup"):
            # Load the LLM into Ollama in parallel with the whisper warmup so
            # the very first dictation is already fast.
            threading.Thread(target=self.cleaner.warmup, daemon=True).start()
        if hasattr(self.asr, "warmup"):
            log.info("warming up whisper model…")
            try:
                self.asr.warmup()
                log.info("whisper ready")
            except Exception:
                log.exception("whisper warmup failed — will retry lazily")
        self.recorder.open()
        log.info("microphone stream open")
        self.hotkey.on_press(self._on_hotkey_press)
        self.hotkey.on_release(self._on_hotkey_release)
        if self.correction_combo and hasattr(self.hotkey, "add_chord"):
            # Same listener as push-to-talk: a second keyboard event tap
            # in one process aborts on macOS.
            self.hotkey.add_chord(self.correction_combo, self._request_correction)
        self.hotkey.start()
        log.info("hotkey listener started — waiting for push-to-talk")
        if self.tray is not None:
            self.tray.run()
        else:
            threading.Event().wait()


def build_default(cfg: dict) -> LocalFlowApp:
    """Construct the real component graph from a loaded config dict."""
    from localflow.asr import WhisperEngine
    from localflow.audio import Recorder
    from localflow.cleanup import OllamaCleaner
    from localflow.command_mode import CommandRunner
    from localflow.context import detect
    from localflow.dictionary import PersonalDictionary
    from localflow.hotkey import PushToTalkListener
    from localflow.inject import ClipboardInjector
    from localflow.overlay import RecordingOverlay
    from localflow.streaming import StreamingTranscriber
    from localflow.tray import LocalFlowTray
    from localflow.vad import SilenceDetector

    audio_cfg = cfg["audio"]
    recorder = Recorder(samplerate=audio_cfg["samplerate"], channels=audio_cfg["channels"],
                        preroll_ms=audio_cfg["preroll_ms"])
    vad_cfg = cfg["vad"]
    vad = None
    if vad_cfg.get("enabled", True):
        vad = SilenceDetector(samplerate=audio_cfg["samplerate"],
                              energy_threshold=vad_cfg["energy_threshold"],
                              silence_ms=vad_cfg["silence_ms"],
                              min_speech_ms=vad_cfg["min_speech_ms"])
    whisper_cfg = cfg["whisper"]
    use_mlx = whisper_cfg.get("backend", "ctranslate2") == "mlx"
    if use_mlx:
        try:
            import mlx_whisper  # noqa: F401 — availability check (Apple Silicon only)
        except ImportError:
            log.warning("mlx-whisper not installed — falling back to CPU backend")
            use_mlx = False
    if use_mlx:
        from localflow.asr import MlxWhisperEngine

        asr = MlxWhisperEngine(model_name=whisper_cfg["model"],
                               language=whisper_cfg.get("language", "auto"))
    else:
        asr = WhisperEngine(model_name=whisper_cfg["model"], device=whisper_cfg["device"],
                            compute_type=whisper_cfg["compute_type"],
                            language=whisper_cfg.get("language", "auto"),
                            cpu_threads=whisper_cfg.get("cpu_threads", 8))
    ollama_cfg = cfg["ollama"]
    keep_alive = ollama_cfg.get("keep_alive", "30m")
    cleanup_mode = cfg.get("cleanup", {}).get("mode", "full")
    cleaner = None
    if cleanup_mode != "off":
        cleaner = OllamaCleaner(base_url=ollama_cfg["base_url"], model=ollama_cfg["model"],
                                fallback_model=ollama_cfg["fallback_model"],
                                timeout_s=ollama_cfg["timeout_s"], keep_alive=keep_alive,
                                mode=cleanup_mode)
    commander = CommandRunner(base_url=ollama_cfg["base_url"], model=ollama_cfg["model"],
                              fallback_model=ollama_cfg["fallback_model"],
                              timeout_s=ollama_cfg["timeout_s"], keep_alive=keep_alive)
    dictionary = PersonalDictionary(cfg["dictionary"]["path"])
    injector = ClipboardInjector(
        restore_delay_s=cfg["paste"]["restore_clipboard_after_ms"] / 1000)
    hotkey = PushToTalkListener(combo=cfg["hotkey"]["combo"])
    overlay_cfg = cfg.get("overlay", {})
    overlay = None
    if overlay_cfg.get("enabled", True):
        overlay = RecordingOverlay(position=overlay_cfg.get("position", "bottom-center"))
    streamer = None
    if cfg["pipeline"].get("streaming", True):
        streamer = StreamingTranscriber(asr, samplerate=audio_cfg["samplerate"])
    app = LocalFlowApp(recorder=recorder, asr=asr, cleaner=cleaner, commander=commander,
                       injector=injector, context_provider=detect, dictionary=dictionary,
                       hotkey=hotkey, tray=None, vad=vad, overlay=overlay,
                       streamer=streamer,
                       correction_combo=cfg["hotkey"].get("correction_combo", "ctrl+alt+c"),
                       min_duration_s=cfg["pipeline"]["min_duration_s"],
                       samplerate=audio_cfg["samplerate"])

    def _learn_correction(old, new):
        added = dictionary.observe_correction(old, new)
        log.info("dictionary learned: %s", ", ".join(added) if added else "nothing new")

    app.tray = LocalFlowTray(dictionary_path=dictionary.path,
                             get_last_text=lambda: app.last_pasted,
                             on_correction=_learn_correction)
    return app
