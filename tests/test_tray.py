from types import SimpleNamespace

from localflow.tray import IDLE_ICON, RECORDING_ICON, LocalFlowTray


class FakeApp:
    def __init__(self, name, title=None, quit_button=None):
        self.name = name
        self.title = title
        self.quit_button = quit_button
        self.menu = None
        self.ran = False

    def run(self):
        self.ran = True


class FakeMenuItem:
    def __init__(self, title, callback=None):
        self.title = title
        self.callback = callback
        self.state = 0


class FakeWindow:
    next_response = SimpleNamespace(clicked=1, text="")
    created = []

    def __init__(self, message="", title="", default_text="", ok=None, cancel=None,
                 dimensions=None):
        self.default_text = default_text
        FakeWindow.created.append(self)

    def run(self):
        return FakeWindow.next_response


def make_tray(**kwargs):
    quits = []
    alerts = []
    FakeWindow.created = []
    fake_rumps = SimpleNamespace(App=FakeApp, MenuItem=FakeMenuItem, Window=FakeWindow,
                                 alert=alerts.append,
                                 quit_application=lambda: quits.append(1))
    tray = LocalFlowTray(rumps_module=fake_rumps, **kwargs)
    return tray, quits, alerts


def test_menu_is_built():
    tray, _, _ = make_tray()
    assert tray.app.title == IDLE_ICON
    titles = [item.title for item in tray.app.menu if item is not None]
    assert titles == ["Dictation enabled", "Correct last dictation…",
                      "Edit dictionary…", "Quit LocalFlow"]
    assert None in tray.app.menu  # separator
    assert tray.toggle_item.state == 1


def test_correct_learns_from_edited_text():
    learned = []
    tray, _, _ = make_tray(get_last_text=lambda: "meet with sara",
                           on_correction=lambda old, new: learned.append((old, new)))
    FakeWindow.next_response = SimpleNamespace(clicked=1, text="meet with Sarah")
    tray._correct(None)
    assert learned == [("meet with sara", "meet with Sarah")]
    assert FakeWindow.created[0].default_text == "meet with sara"


def test_correct_ignores_cancel_and_unchanged_text():
    learned = []
    tray, _, _ = make_tray(get_last_text=lambda: "hello world",
                           on_correction=lambda old, new: learned.append((old, new)))
    FakeWindow.next_response = SimpleNamespace(clicked=0, text="hello universe")
    tray._correct(None)  # cancelled
    FakeWindow.next_response = SimpleNamespace(clicked=1, text="hello world")
    tray._correct(None)  # unchanged
    assert learned == []


def test_correct_without_dictation_shows_alert():
    tray, _, alerts = make_tray(get_last_text=lambda: None)
    tray._correct(None)
    assert len(alerts) == 1
    assert FakeWindow.created == []


def test_set_recording_switches_icon():
    tray, _, _ = make_tray()
    tray.set_recording(True)
    assert tray.app.title == RECORDING_ICON
    tray.set_recording(False)
    assert tray.app.title == IDLE_ICON


def test_toggle_flips_state_and_notifies():
    events = []
    tray, _, _ = make_tray(on_toggle=events.append)
    tray._toggle(tray.toggle_item)
    assert tray.enabled is False
    assert tray.toggle_item.state == 0
    tray._toggle(tray.toggle_item)
    assert tray.enabled is True
    assert events == [False, True]


def test_quit_notifies_and_quits_rumps():
    events = []
    tray, quits, _ = make_tray(on_quit=lambda: events.append("quit"))
    tray._quit(tray.quit_item)
    assert events == ["quit"]
    assert quits == [1]


def test_edit_dictionary_opens_file(monkeypatch, tmp_path):
    opened = []
    import localflow.tray as tray_mod

    monkeypatch.setattr(tray_mod.subprocess, "Popen", lambda args: opened.append(args))
    path = tmp_path / "dictionary.json"
    tray, _, _ = make_tray(dictionary_path=path)
    tray._edit_dictionary(None)
    assert opened == [["open", "-t", str(path)]]


def test_run_delegates_to_rumps_app():
    tray, _, _ = make_tray()
    tray.run()
    assert tray.app.ran
