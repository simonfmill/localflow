import pytest

from localflow.context import classify, detect


class FakeApp:
    def __init__(self, bundle_id, name):
        self._bundle = bundle_id
        self._name = name

    def bundleIdentifier(self):
        return self._bundle

    def localizedName(self):
        return self._name


class FakeWorkspace:
    def __init__(self, app):
        self._app = app

    def frontmostApplication(self):
        return self._app


@pytest.mark.parametrize("bundle,name,kind", [
    ("com.apple.mail", "Mail", "email"),
    ("com.microsoft.Outlook", "Microsoft Outlook", "email"),
    ("com.tinyspeck.slackmacgap", "Slack", "chat"),
    ("com.hnc.Discord", "Discord", "chat"),
    ("com.microsoft.VSCode", "Code", "code"),
    ("com.jetbrains.pycharm", "PyCharm", "code"),
    ("com.apple.Terminal", "Terminal", "terminal"),
    ("com.googlecode.iterm2", "iTerm2", "terminal"),
    ("com.apple.TextEdit", "TextEdit", "generic"),
    ("", "", "generic"),
])
def test_classify(bundle, name, kind):
    assert classify(bundle, name) == kind


def test_classify_falls_back_to_name_hint():
    assert classify("org.unknown.app", "SuperSlack Client") == "chat"
    assert classify("org.unknown.app", "My Terminal Emulator") == "terminal"


def test_detect_builds_app_context():
    ws = FakeWorkspace(FakeApp("com.tinyspeck.slackmacgap", "Slack"))
    ctx = detect(workspace=ws)
    assert ctx.bundle_id == "com.tinyspeck.slackmacgap"
    assert ctx.app_name == "Slack"
    assert ctx.kind == "chat"


def test_detect_handles_no_frontmost_app():
    ctx = detect(workspace=FakeWorkspace(None))
    assert ctx.kind == "generic"
    assert ctx.bundle_id == ""


def test_detect_handles_none_bundle_fields():
    ctx = detect(workspace=FakeWorkspace(FakeApp(None, None)))
    assert ctx.bundle_id == ""
    assert ctx.app_name == ""
    assert ctx.kind == "generic"
