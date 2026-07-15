import numpy as np

from localflow.app import IDLE, PROCESSING, RECORDING, LocalFlowApp
from localflow.contracts import AppContext, CleanupResult, Transcript

SR = 16000


class FakeRecorder:
    def __init__(self, audio=None):
        self.block_listeners = []
        self.audio = audio if audio is not None else np.zeros(SR, dtype=np.float32)
        self.preroll = np.full(800, 0.1, dtype=np.float32)
        self.started = 0
        self.stopped = 0

    def open(self):
        pass

    def start(self):
        self.started += 1
        return self.preroll

    def stop(self):
        self.stopped += 1
        return self.audio


class FakeASR:
    def __init__(self, text="um hello world"):
        self.text = text
        self.calls = []
        self.hotwords = []

    def transcribe(self, wav, hotwords=None):
        self.calls.append(wav)
        self.hotwords.append(hotwords)
        return Transcript(text=self.text, segments=[], lang="en", duration_s=1.0)


class FakeCleaner:
    def __init__(self):
        self.requests = []

    def clean(self, req):
        self.requests.append(req)
        return CleanupResult(text="Hello world.")


class FakeCommander:
    def __init__(self, result="Rewritten."):
        self.requests = []
        self.result = result

    def run(self, req):
        self.requests.append(req)
        return self.result


class FakeInjector:
    def __init__(self):
        self.pasted = []

    def paste(self, text, ctx):
        self.pasted.append((text, ctx))


class FakeDictionary:
    def __init__(self):
        self.terms = ["Qwen"]
        self.corrections = []

    def observe_correction(self, old, new):
        self.corrections.append((old, new))
        return ["Sarah"]


class FakeHotkey:
    def __init__(self):
        self.press_cb = None
        self.release_cb = None
        self.chords = []
        self.started = False

    def on_press(self, cb):
        self.press_cb = cb

    def on_release(self, cb):
        self.release_cb = cb

    def add_chord(self, combo, callback):
        self.chords.append((combo, callback))

    def start(self):
        self.started = True


class FakeTray:
    def __init__(self):
        self.enabled = True
        self.recording_states = []
        self.ran = False

    def set_recording(self, value):
        self.recording_states.append(value)

    def run(self):
        self.ran = True


class FakeVad:
    def __init__(self, stop_after=None):
        self.fed = 0
        self.resets = 0
        self.stop_after = stop_after

    def reset(self):
        self.resets += 1

    def feed(self, block):
        self.fed += 1
        return self.stop_after is not None and self.fed >= self.stop_after


class FakeOverlay:
    def __init__(self):
        self.events = []

    def show(self):
        self.events.append("show")

    def hide(self):
        self.events.append("hide")

    def feed(self, level):
        self.events.append(("feed", round(level, 3)))


CTX = AppContext("com.tinyspeck.slackmacgap", "Slack", "chat")


def make_app(**overrides):
    parts = dict(
        recorder=FakeRecorder(),
        asr=FakeASR(),
        cleaner=FakeCleaner(),
        commander=FakeCommander(),
        injector=FakeInjector(),
        context_provider=lambda: CTX,
        dictionary=FakeDictionary(),
        hotkey=FakeHotkey(),
        tray=FakeTray(),
        vad=None,
        threaded=False,
    )
    parts.update(overrides)
    return LocalFlowApp(**parts), parts


def test_press_starts_recording():
    app, p = make_app()
    app._on_hotkey_press()
    assert app.state == RECORDING
    assert p["recorder"].started == 1
    assert p["tray"].recording_states == [True]


def test_release_runs_dictation_pipeline():
    app, p = make_app()
    app._on_hotkey_press()
    app._on_hotkey_release()
    assert app.state == IDLE
    assert p["recorder"].stopped == 1
    assert p["tray"].recording_states == [True, False]
    assert len(p["asr"].calls) == 1
    req = p["cleaner"].requests[0]
    assert req.raw_text == "um hello world"
    assert req.dictionary == ["Qwen"]
    assert "chat" in req.profile
    assert "Slack" in req.context_hint
    assert p["injector"].pasted == [("Hello world.", CTX)]
    assert p["commander"].requests == []
    assert app.last_pasted == "Hello world."


def test_trigger_phrase_routes_to_command_mode():
    app, p = make_app(asr=FakeASR(text="voice command make this more formal"))
    app._on_hotkey_press()
    app._on_hotkey_release()
    assert p["cleaner"].requests == []
    cmd = p["commander"].requests[0]
    assert cmd.instruction == "make this more formal"
    assert cmd.selection is None
    assert p["injector"].pasted == [("Rewritten.", CTX)]


def test_selection_routes_to_command_mode():
    app, p = make_app(asr=FakeASR(text="translate this to german"),
                      selection_provider=lambda: "good morning")
    app._on_hotkey_press()
    app._on_hotkey_release()
    cmd = p["commander"].requests[0]
    assert cmd.instruction == "translate this to german"
    assert cmd.selection == "good morning"


def test_short_audio_is_dropped():
    short = np.zeros(int(0.1 * SR), dtype=np.float32)
    app, p = make_app(recorder=FakeRecorder(audio=short))
    app._on_hotkey_press()
    app._on_hotkey_release()
    assert p["asr"].calls == []
    assert p["injector"].pasted == []
    assert app.state == IDLE


def test_empty_transcript_is_dropped():
    app, p = make_app(asr=FakeASR(text="   "))
    app._on_hotkey_press()
    app._on_hotkey_release()
    assert p["cleaner"].requests == []
    assert p["injector"].pasted == []


def test_empty_command_result_is_not_pasted():
    app, p = make_app(asr=FakeASR(text="voice command do something"),
                      commander=FakeCommander(result=""))
    app._on_hotkey_press()
    app._on_hotkey_release()
    assert p["injector"].pasted == []


def test_disabled_via_tray_ignores_press():
    app, p = make_app()
    p["tray"].enabled = False
    app._on_hotkey_press()
    assert app.state == IDLE
    assert p["recorder"].started == 0


def test_vad_auto_stops_recording():
    vad = FakeVad(stop_after=3)
    app, p = make_app(vad=vad)
    app._on_hotkey_press()
    assert vad.resets == 1
    block = np.zeros(1600, dtype=np.float32)
    for _ in range(3):
        app._on_block(block)
    assert app.state == IDLE  # release path ran via VAD
    assert p["recorder"].stopped == 1
    assert p["injector"].pasted  # pipeline completed


def test_release_without_recording_is_noop():
    app, p = make_app()
    app._on_hotkey_release()
    assert p["recorder"].stopped == 0
    assert app.state == IDLE


def test_run_wires_hotkey_and_tray():
    app, p = make_app(asr=FakeASR())
    app.run()
    hk = p["hotkey"]
    assert hk.started
    assert hk.press_cb == app._on_hotkey_press
    assert hk.release_cb == app._on_hotkey_release
    assert p["tray"].ran


def test_correction_chord_shares_the_single_hotkey_listener():
    app, p = make_app(correction_combo="ctrl+shift+c")
    app.run()
    assert p["hotkey"].chords == [("ctrl+shift+c", app._request_correction)]


def test_correction_file_flow_learns_from_saved_edit(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    def edit_file(path):
        from pathlib import Path

        assert Path(path).read_text() == "meet with sara"
        Path(path).write_text("meet with Sarah")

    app, p = make_app(editor_opener=edit_file)
    app.last_pasted = "meet with sara"
    app._correction_file_flow("meet with sara", poll_interval=0.01, timeout_s=2)
    assert p["dictionary"].corrections == [("meet with sara", "meet with Sarah")]


def test_correction_file_flow_times_out_without_edit(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    app, p = make_app(editor_opener=lambda path: None)
    app._correction_file_flow("unchanged text", poll_interval=0.01, timeout_s=0.05)
    assert p["dictionary"].corrections == []


def test_correction_without_dictation_is_noop():
    opened = []
    app, _ = make_app(editor_opener=opened.append)
    app._request_correction()
    assert opened == []


def test_no_correction_combo_registers_no_chord():
    app, p = make_app()
    app.run()
    assert p["hotkey"].chords == []


def test_double_press_is_ignored_while_recording():
    app, p = make_app()
    app._on_hotkey_press()
    app._on_hotkey_press()
    assert p["recorder"].started == 1


def test_state_constants():
    assert (IDLE, RECORDING, PROCESSING) == ("idle", "recording", "processing")


class FakeStreamer:
    def __init__(self, text="um hello world", total_s=2.0):
        self.text = text
        self.total_s = total_s
        self.started_with = []
        self.fed = []
        self.finished = 0

    def start(self, hotwords=None):
        self.started_with.append(hotwords)

    def feed(self, block):
        self.fed.append(len(block))

    def finish(self):
        self.finished += 1
        return self.text


def test_streaming_pipeline_uses_streamer_text():
    streamer = FakeStreamer(text="um hello world")
    app, p = make_app(streamer=streamer)
    app._on_hotkey_press()
    assert streamer.started_with == ["Qwen"]  # dictionary as hotwords
    assert streamer.fed == [800]  # preroll ingested
    block = np.zeros(1600, dtype=np.float32)
    app._on_block(block)
    assert streamer.fed == [800, 1600]
    app._on_hotkey_release()
    assert streamer.finished == 1
    assert p["asr"].calls == []  # no separate full-pass transcription
    assert p["cleaner"].requests[0].raw_text == "um hello world"
    assert p["injector"].pasted == [("Hello world.", CTX)]


def test_streaming_short_capture_is_dropped():
    streamer = FakeStreamer(text="hi", total_s=0.1)
    app, p = make_app(streamer=streamer)
    app._on_hotkey_press()
    app._on_hotkey_release()
    assert streamer.finished == 1
    assert p["injector"].pasted == []


def test_overlay_shows_on_press_and_hides_on_release():
    overlay = FakeOverlay()
    app, _ = make_app(overlay=overlay)
    app._on_hotkey_press()
    assert overlay.events == ["show"]
    app._on_hotkey_release()
    assert overlay.events == ["show", "hide"]


def test_overlay_receives_mic_levels_while_recording():
    overlay = FakeOverlay()
    app, _ = make_app(overlay=overlay)
    block = np.full(1600, 0.5, dtype=np.float32)
    app._on_block(block)  # idle — ignored
    assert overlay.events == []
    app._on_hotkey_press()
    app._on_block(block)
    assert ("feed", 0.5) in overlay.events


def test_dictionary_terms_become_asr_hotwords():
    app, p = make_app()
    app._on_hotkey_press()
    app._on_hotkey_release()
    assert p["asr"].hotwords == ["Qwen"]


def test_no_overlay_is_fine():
    app, p = make_app(overlay=None)
    app._on_hotkey_press()
    app._on_hotkey_release()
    assert p["injector"].pasted  # pipeline unaffected
