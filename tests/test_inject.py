import pytest

from localflow.contracts import AppContext
from localflow.inject import ClipboardInjector


class Harness:
    def __init__(self, clipboard="old content", paste_fails=False):
        self.ops = []
        self.clipboard = clipboard
        self.paste_fails = paste_fails

    def get(self):
        self.ops.append(("get", self.clipboard))
        return self.clipboard

    def set(self, text):
        self.ops.append(("set", text))
        self.clipboard = text

    def send_paste(self):
        if self.paste_fails:
            raise RuntimeError("no accessibility grant")
        self.ops.append(("paste", None))

    def type_text(self, text):
        self.ops.append(("type", text))

    def sleep(self, seconds):
        self.ops.append(("sleep", seconds))

    def injector(self, **kwargs):
        return ClipboardInjector(clipboard_get=self.get, clipboard_set=self.set,
                                 send_paste=self.send_paste, type_text=self.type_text,
                                 sleep=self.sleep, restore_delay_s=0.3, **kwargs)


GENERIC = AppContext("com.apple.TextEdit", "TextEdit", "generic")
TERMINAL = AppContext("com.apple.Terminal", "Terminal", "terminal")
CODE = AppContext("com.microsoft.VSCode", "Code", "code")


def test_paste_sets_clipboard_sends_cmd_v_and_restores():
    h = Harness(clipboard="previous")
    h.injector().paste("Hello world.", GENERIC)
    assert h.ops == [
        ("get", "previous"),
        ("set", "Hello world."),
        ("paste", None),
        ("sleep", 0.3),
        ("set", "previous"),
    ]
    assert h.clipboard == "previous"


def test_falls_back_to_typing_when_paste_fails():
    h = Harness(paste_fails=True)
    h.injector().paste("Hello.", GENERIC)
    assert ("type", "Hello.") in h.ops
    assert h.ops[-1] == ("set", "old content")  # clipboard still restored


def test_auto_capitalizes_for_generic_apps():
    h = Harness()
    h.injector().paste("hello there", GENERIC)
    assert ("set", "Hello there") in h.ops


@pytest.mark.parametrize("ctx", [TERMINAL, CODE])
def test_terminal_and_code_get_verbatim_text(ctx):
    h = Harness()
    h.injector().paste("ls -la ~/Downloads", ctx)
    assert ("set", "ls -la ~/Downloads") in h.ops


def test_empty_text_is_not_pasted():
    h = Harness()
    h.injector().paste("   ", GENERIC)
    assert h.ops == []


def test_none_clipboard_skips_restore():
    h = Harness()
    h.clipboard = None

    def get_none():
        h.ops.append(("get", None))
        return None

    inj = ClipboardInjector(clipboard_get=get_none, clipboard_set=h.set,
                            send_paste=h.send_paste, type_text=h.type_text, sleep=h.sleep)
    inj.paste("Hi", GENERIC)
    assert h.ops[-1] == ("sleep", 0.3)  # no restoring set() after the sleep
