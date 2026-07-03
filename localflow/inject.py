"""Text injection into the focused app via clipboard swap + Cmd+V.

Saves the clipboard, sets the text, sends Cmd+V, then restores the previous
clipboard after a short delay. If the paste keystroke fails (e.g. missing
Accessibility grant for synthetic events), falls back to typing the text
directly. For terminal/code apps the text is injected verbatim — no
auto-capitalization or whitespace tweaks.
"""

import time

from localflow.contracts import AppContext

RAW_KINDS = ("terminal", "code")


class ClipboardInjector:
    def __init__(self, clipboard_get=None, clipboard_set=None, send_paste=None,
                 type_text=None, restore_delay_s=0.3, sleep=time.sleep,
                 auto_capitalize=True):
        self._get = clipboard_get or self._default_get
        self._set = clipboard_set or self._default_set
        self._send_paste = send_paste or self._default_send_paste
        self._type = type_text or self._default_type_text
        self._restore_delay_s = restore_delay_s
        self._sleep = sleep
        self._auto_capitalize = auto_capitalize

    @staticmethod
    def _default_get():
        import pyperclip

        try:
            return pyperclip.paste()
        except Exception:
            return None

    @staticmethod
    def _default_set(text):
        import pyperclip

        pyperclip.copy(text)

    @staticmethod
    def _default_send_paste():
        from pynput.keyboard import Controller, Key

        kb = Controller()
        with kb.pressed(Key.cmd):
            kb.press("v")
            kb.release("v")

    @staticmethod
    def _default_type_text(text):
        from pynput.keyboard import Controller

        Controller().type(text)

    def prepare(self, text: str, ctx: AppContext | None) -> str:
        if ctx is not None and ctx.kind in RAW_KINDS:
            return text  # verbatim for terminals and code editors
        text = text.strip()
        if self._auto_capitalize and text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    def paste(self, text: str, ctx: AppContext | None = None) -> None:
        text = self.prepare(text, ctx)
        if not text:
            return
        try:
            previous = self._get()
        except Exception:
            previous = None
        self._set(text)
        try:
            self._send_paste()
        except Exception:
            self._type(text)
        self._sleep(self._restore_delay_s)
        if previous is not None:
            try:
                self._set(previous)
            except Exception:
                pass
