"""Shared dataclasses and protocols that every LocalFlow module codes against."""

from dataclasses import dataclass, field
from typing import Callable, Protocol

APP_KINDS = ("email", "chat", "code", "terminal", "generic")


@dataclass
class Transcript:
    text: str
    segments: list = field(default_factory=list)
    lang: str = "en"
    duration_s: float = 0.0


@dataclass
class AppContext:
    bundle_id: str
    app_name: str
    kind: str  # one of APP_KINDS


@dataclass
class CleanupRequest:
    raw_text: str
    dictionary: list
    profile: str
    context_hint: str = ""


@dataclass
class CleanupResult:
    text: str


@dataclass
class CommandRequest:
    instruction: str
    selection: str | None = None


class ASREngine(Protocol):
    def transcribe(self, wav) -> Transcript: ...


class Cleaner(Protocol):
    def clean(self, req: CleanupRequest) -> CleanupResult: ...


class Injector(Protocol):
    def paste(self, text: str, ctx: AppContext) -> None: ...


class HotkeyListener(Protocol):
    def on_press(self, cb: Callable[[], None]) -> None: ...

    def on_release(self, cb: Callable[[], None]) -> None: ...
