"""Global push-to-talk hotkey via pynput.

Fires on_press once when the full combo is held and on_release when any key
of the combo is let go. The listener factory is injectable for tests.

Note: some macOS keyboards do not deliver the Fn key to pynput; the Fn key,
when it does arrive, shows up as virtual keycode 63. If "fn+ctrl" does not
trigger on your machine, set hotkey.combo to "ctrl+alt" in the user config.
"""

_TOKEN_ALIASES = {
    "ctrl_l": "ctrl",
    "ctrl_r": "ctrl",
    "alt_l": "alt",
    "alt_r": "alt",
    "alt_gr": "alt",
    "cmd_l": "cmd",
    "cmd_r": "cmd",
    "shift_l": "shift",
    "shift_r": "shift",
}

_FN_VIRTUAL_KEYCODES = {63, 179}  # macOS Fn/Globe key (varies by keyboard)


def _token_for(key) -> str | None:
    name = getattr(key, "name", None)
    if name:
        return _TOKEN_ALIASES.get(name, name)
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    if getattr(key, "vk", None) in _FN_VIRTUAL_KEYCODES:
        return "fn"
    return None


class PushToTalkListener:
    """One global listener: a push-to-talk combo plus optional extra chords.

    macOS aborts when a process installs several keyboard event taps, so all
    hotkeys must share this single pynput listener (see add_chord).
    """

    def __init__(self, combo="fn+ctrl", listener_factory=None):
        self._required = {t.strip().lower() for t in combo.split("+") if t.strip()}
        self._watched = set(self._required)
        self._chords: list = []
        self._pressed: set = set()
        self._active = False
        self._press_cb = None
        self._release_cb = None
        self._listener_factory = listener_factory or self._default_factory
        self._listener = None

    @staticmethod
    def _default_factory(on_press, on_release):
        from pynput import keyboard

        return keyboard.Listener(on_press=on_press, on_release=on_release)

    def on_press(self, cb):
        self._press_cb = cb

    def on_release(self, cb):
        self._release_cb = cb

    def add_chord(self, combo, callback):
        """Fire `callback` once whenever `combo` is fully pressed."""
        required = {t.strip().lower() for t in combo.split("+") if t.strip()}
        if required:
            self._chords.append({"required": required, "cb": callback, "active": False})
            self._watched |= required

    def _handle_press(self, key):
        token = _token_for(key)
        if token not in self._watched:
            return
        self._pressed.add(token)
        if self._required <= self._pressed and not self._active:
            self._active = True
            if self._press_cb:
                self._press_cb()
        for chord in self._chords:
            if chord["required"] <= self._pressed and not chord["active"]:
                chord["active"] = True
                chord["cb"]()

    def _handle_release(self, key):
        token = _token_for(key)
        if token not in self._watched:
            return
        self._pressed.discard(token)
        if self._active and token in self._required:
            self._active = False
            if self._release_cb:
                self._release_cb()
        for chord in self._chords:
            if chord["active"] and token in chord["required"]:
                chord["active"] = False

    def start(self):
        if self._listener is None:
            self._listener = self._listener_factory(self._handle_press, self._handle_release)
            self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
