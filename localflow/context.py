"""Frontmost-app detection via NSWorkspace, classified into an AppContext kind."""

from localflow.contracts import AppContext

_BUNDLE_KINDS = {
    "com.apple.mail": "email",
    "com.microsoft.outlook": "email",
    "com.readdle.smartemail-mac": "email",
    "com.tinyspeck.slackmacgap": "chat",
    "com.hnc.discord": "chat",
    "ru.keepcoder.telegram": "chat",
    "net.whatsapp.whatsapp": "chat",
    "com.apple.mobilesms": "chat",
    "com.microsoft.vscode": "code",
    "com.todesktop.230313mzl4w4u92": "code",  # Cursor
    "com.apple.dt.xcode": "code",
    "com.apple.terminal": "terminal",
    "com.googlecode.iterm2": "terminal",
    "dev.warp.warp-stable": "terminal",
    "com.mitchellh.ghostty": "terminal",
}

_BUNDLE_PREFIX_KINDS = (
    ("com.jetbrains", "code"),
    ("com.sublimetext", "code"),
)

_NAME_HINTS = (
    ("outlook", "email"),
    ("mail", "email"),
    ("slack", "chat"),
    ("discord", "chat"),
    ("telegram", "chat"),
    ("messages", "chat"),
    ("whatsapp", "chat"),
    ("iterm", "terminal"),
    ("terminal", "terminal"),
    ("warp", "terminal"),
    ("ghostty", "terminal"),
    ("xcode", "code"),
    ("code", "code"),
    ("cursor", "code"),
    ("intellij", "code"),
    ("pycharm", "code"),
)


def classify(bundle_id: str, app_name: str) -> str:
    bundle = (bundle_id or "").lower()
    name = (app_name or "").lower()
    if bundle in _BUNDLE_KINDS:
        return _BUNDLE_KINDS[bundle]
    for prefix, kind in _BUNDLE_PREFIX_KINDS:
        if bundle.startswith(prefix):
            return kind
    for hint, kind in _NAME_HINTS:
        if hint in name:
            return kind
    return "generic"


def detect(workspace=None) -> AppContext:
    """Return an AppContext for the frontmost application."""
    if workspace is None:
        from AppKit import NSWorkspace

        workspace = NSWorkspace.sharedWorkspace()
    app = workspace.frontmostApplication()
    if app is None:
        return AppContext(bundle_id="", app_name="", kind="generic")
    bundle_id = app.bundleIdentifier() or ""
    app_name = app.localizedName() or ""
    return AppContext(bundle_id=bundle_id, app_name=app_name,
                      kind=classify(bundle_id, app_name))
