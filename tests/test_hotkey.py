from types import SimpleNamespace

from localflow.hotkey import PushToTalkListener, _token_for

CTRL = SimpleNamespace(name="ctrl_l")
CTRL_R = SimpleNamespace(name="ctrl_r")
FN = SimpleNamespace(name=None, char=None, vk=63)
FN_GLOBE = SimpleNamespace(name=None, char=None, vk=179)
A_KEY = SimpleNamespace(name=None, char="a", vk=0)


class FakeListener:
    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def make_listener(combo="fn+ctrl"):
    created = {}

    def factory(on_press, on_release):
        created["listener"] = FakeListener(on_press, on_release)
        return created["listener"]

    ptt = PushToTalkListener(combo=combo, listener_factory=factory)
    presses, releases = [], []
    ptt.on_press(lambda: presses.append(1))
    ptt.on_release(lambda: releases.append(1))
    return ptt, presses, releases, created


def test_token_mapping():
    assert _token_for(CTRL) == "ctrl"
    assert _token_for(CTRL_R) == "ctrl"
    assert _token_for(FN) == "fn"
    assert _token_for(FN_GLOBE) == "fn"
    assert _token_for(A_KEY) == "a"


def test_full_combo_fires_press_once():
    ptt, presses, releases, _ = make_listener()
    ptt._handle_press(CTRL)
    assert presses == []
    ptt._handle_press(FN)
    assert presses == [1]
    ptt._handle_press(CTRL)  # key repeat while held — no re-fire
    assert presses == [1]
    assert releases == []


def test_releasing_any_combo_key_fires_release_once():
    ptt, presses, releases, _ = make_listener()
    ptt._handle_press(CTRL)
    ptt._handle_press(FN)
    ptt._handle_release(FN)
    assert releases == [1]
    ptt._handle_release(CTRL)
    assert releases == [1]


def test_combo_can_retrigger():
    ptt, presses, releases, _ = make_listener()
    for _ in range(2):
        ptt._handle_press(CTRL)
        ptt._handle_press(FN)
        ptt._handle_release(FN)
        ptt._handle_release(CTRL)
    assert presses == [1, 1]
    assert releases == [1, 1]


def test_unrelated_keys_are_ignored():
    ptt, presses, releases, _ = make_listener()
    ptt._handle_press(A_KEY)
    ptt._handle_press(CTRL)
    ptt._handle_press(A_KEY)
    assert presses == []
    ptt._handle_release(A_KEY)
    assert releases == []


def test_start_and_stop_manage_backend_listener():
    ptt, _, _, created = make_listener()
    ptt.start()
    assert created["listener"].started
    ptt.start()  # idempotent
    ptt.stop()
    assert created["listener"].stopped


def test_custom_combo():
    ptt, presses, _, _ = make_listener(combo="ctrl+alt")
    ptt._handle_press(CTRL)
    ptt._handle_press(SimpleNamespace(name="alt_l"))
    assert presses == [1]


ALT = SimpleNamespace(name="alt_l")
C_KEY = SimpleNamespace(name=None, char="c", vk=8)


def test_chord_fires_once_and_rearms_on_release():
    ptt, presses, _, _ = make_listener(combo="cmd+alt")
    chords = []
    ptt.add_chord("ctrl+alt+c", lambda: chords.append(1))
    for key in (CTRL, ALT, C_KEY):
        ptt._handle_press(key)
    assert chords == [1]
    assert presses == []  # push-to-talk combo (cmd+alt) did not fire
    ptt._handle_press(C_KEY)  # key repeat while held — no re-fire
    assert chords == [1]
    ptt._handle_release(C_KEY)
    ptt._handle_press(C_KEY)  # re-pressed — fires again
    assert chords == [1, 1]


def test_chord_does_not_break_push_to_talk():
    ptt, presses, releases, _ = make_listener(combo="fn+ctrl")
    ptt.add_chord("ctrl+alt+c", lambda: None)
    ptt._handle_press(CTRL)
    ptt._handle_press(FN)
    ptt._handle_release(FN)
    assert presses == [1]
    assert releases == [1]
