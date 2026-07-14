"""rumps menubar tray: recording indicator plus toggle/dictionary/quit menu.

The rumps module is injectable so tests run without a GUI.
"""

import subprocess

IDLE_ICON = "🎙"
RECORDING_ICON = "🔴"


class LocalFlowTray:
    def __init__(self, dictionary_path=None, on_toggle=None, on_quit=None,
                 get_last_text=None, on_correction=None, rumps_module=None):
        if rumps_module is None:
            import rumps as rumps_module
        self._rumps = rumps_module
        self._dictionary_path = dictionary_path
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._get_last_text = get_last_text
        self._on_correction = on_correction
        self.enabled = True
        self.app = self._rumps.App("LocalFlow", title=IDLE_ICON, quit_button=None)
        self.toggle_item = self._rumps.MenuItem("Dictation enabled", callback=self._toggle)
        self.toggle_item.state = 1
        self.correct_item = self._rumps.MenuItem("Correct last dictation…",
                                                 callback=self._correct)
        self.dictionary_item = self._rumps.MenuItem("Edit dictionary…",
                                                    callback=self._edit_dictionary)
        self.quit_item = self._rumps.MenuItem("Quit LocalFlow", callback=self._quit)
        self.app.menu = [self.toggle_item, self.correct_item, self.dictionary_item,
                         None, self.quit_item]

    def set_recording(self, recording: bool):
        self.app.title = RECORDING_ICON if recording else IDLE_ICON

    def _toggle(self, item):
        self.enabled = not self.enabled
        item.state = 1 if self.enabled else 0
        if self._on_toggle:
            self._on_toggle(self.enabled)

    def _correct(self, _item):
        """Let the user fix the last pasted text; learned terms go to the dictionary."""
        last = self._get_last_text() if self._get_last_text else None
        if not last:
            self._rumps.alert("Nothing to correct yet — dictate something first.")
            return
        window = self._rumps.Window(
            message="Fix any misrecognized words. LocalFlow learns the corrections "
                    "and recognizes them next time.",
            title="Correct last dictation", default_text=last,
            ok="Learn", cancel="Cancel", dimensions=(420, 120))
        response = window.run()
        corrected = (response.text or "").strip()
        if getattr(response, "clicked", 0) and corrected and corrected != last:
            if self._on_correction:
                self._on_correction(last, corrected)

    def _edit_dictionary(self, _item):
        if self._dictionary_path:
            subprocess.Popen(["open", "-t", str(self._dictionary_path)])

    def _quit(self, _item):
        if self._on_quit:
            self._on_quit()
        self._rumps.quit_application()

    def run(self):
        self.app.run()
