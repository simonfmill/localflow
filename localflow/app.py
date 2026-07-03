"""Orchestrator: wires hotkey → audio → vad → asr → context/profiles/dictionary
→ cleanup or command mode → inject into a push-to-talk session state machine."""

import logging
import threading

from localflow import command_mode, profiles
from localflow.contracts import CleanupRequest, CommandRequest

log = logging.getLogger("localflow")

IDLE = "idle"
RECORDING = "recording"
PROCESSING = "processing"


class LocalFlowApp:
    def __init__(self, *, recorder, asr, cleaner, commander, injector,
                 context_provider, dictionary, hotkey, tray=None, vad=None,
                 selection_provider=None, min_duration_s=0.3, samplerate=16000,
                 threaded=True):
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
        self.selection_provider = selection_provider
        self.min_duration_s = min_duration_s
        self.samplerate = samplerate
        self.last_pasted: str | None = None
        self._threaded = threaded
        self._state = IDLE
        self._lock = threading.Lock()
        self.recorder.block_listeners.append(self._on_block)

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
            self.recorder.start()
            if self.tray is not None:
                self.tray.set_recording(True)
        except Exception:
            log.exception("failed to start recording")
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
        except Exception:
            log.exception("failed to stop recording")
            with self._lock:
                self._state = IDLE
            return
        if self._threaded:
            threading.Thread(target=self._process_safe, args=(audio,), daemon=True).start()
        else:
            self._process_safe(audio)

    def _process_safe(self, audio):
        try:
            self.process(audio)
        except Exception:
            log.exception("pipeline failed")
        finally:
            with self._lock:
                self._state = IDLE

    def _on_block(self, block):
        """Audio-thread callback: VAD auto-stop while recording."""
        if self._state == RECORDING and self.vad is not None and self.vad.feed(block):
            self._on_hotkey_release()

    def process(self, audio):
        if len(audio) < self.min_duration_s * self.samplerate:
            log.info("capture too short (< %.1fs) — dropped", self.min_duration_s)
            return
        transcript = self.asr.transcribe(audio)
        text = transcript.text.strip()
        log.info("transcript: %r", text)
        if not text:
            return
        ctx = self.context_provider()
        selection = self.selection_provider() if self.selection_provider else None
        if command_mode.is_command(text, selection):
            result = self.commander.run(
                CommandRequest(instruction=command_mode.strip_trigger(text),
                               selection=selection)
            )
        else:
            req = CleanupRequest(
                raw_text=text,
                dictionary=list(self.dictionary.terms),
                profile=profiles.fragment_for(ctx.kind),
                context_hint=f"The text will be pasted into {ctx.app_name}." if ctx.app_name else "",
            )
            result = self.cleaner.clean(req).text
        if result:
            log.info("pasting into %s (%s): %r", ctx.app_name or "?", ctx.kind, result)
            self.injector.paste(result, ctx)
            self.last_pasted = result

    def run(self):
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
    asr = WhisperEngine(model_name=whisper_cfg["model"], device=whisper_cfg["device"],
                        compute_type=whisper_cfg["compute_type"])
    ollama_cfg = cfg["ollama"]
    cleaner = OllamaCleaner(base_url=ollama_cfg["base_url"], model=ollama_cfg["model"],
                            fallback_model=ollama_cfg["fallback_model"],
                            timeout_s=ollama_cfg["timeout_s"])
    commander = CommandRunner(base_url=ollama_cfg["base_url"], model=ollama_cfg["model"],
                              fallback_model=ollama_cfg["fallback_model"],
                              timeout_s=ollama_cfg["timeout_s"])
    dictionary = PersonalDictionary(cfg["dictionary"]["path"])
    injector = ClipboardInjector(
        restore_delay_s=cfg["paste"]["restore_clipboard_after_ms"] / 1000)
    hotkey = PushToTalkListener(combo=cfg["hotkey"]["combo"])
    tray = LocalFlowTray(dictionary_path=dictionary.path)
    return LocalFlowApp(recorder=recorder, asr=asr, cleaner=cleaner, commander=commander,
                        injector=injector, context_provider=detect, dictionary=dictionary,
                        hotkey=hotkey, tray=tray, vad=vad,
                        min_duration_s=cfg["pipeline"]["min_duration_s"],
                        samplerate=audio_cfg["samplerate"])
