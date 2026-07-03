"""Command Mode: route dictation vs. spoken commands and execute them via Ollama.

A capture is treated as a command when text is selected in the frontmost app,
or when the transcript starts with a trigger phrase ("voice command ...").
The command runner rewrites/translates/answers and returns only the result.
Fails closed: on Ollama failure it returns "" so nothing is pasted.
"""

import requests

from localflow.contracts import CommandRequest

TRIGGERS = ("voice command", "command mode", "hey flow")

COMMAND_SYSTEM_PROMPT = """You execute a spoken editing command from a dictation \
app. Return ONLY the resulting text — no explanations, no quotes, no preamble.

If a SELECTION is provided, apply the instruction to the selection (rewrite, \
translate, reformat, answer, etc.) and return only the transformed text that \
should replace the selection. If there is no selection, the instruction stands \
alone (e.g. a question to answer or text to produce) — return only the result."""


def is_command(text: str, selection: str | None = None) -> bool:
    if selection:
        return True
    lowered = text.lower().lstrip(" ,.")
    return any(lowered.startswith(t) for t in TRIGGERS)


def strip_trigger(text: str) -> str:
    stripped = text.lstrip(" ,.")
    lowered = stripped.lower()
    for trigger in TRIGGERS:
        if lowered.startswith(trigger):
            return stripped[len(trigger):].lstrip(" ,.:;-—").strip()
    return text.strip()


class CommandRunner:
    def __init__(self, base_url="http://localhost:11434", model="qwen2.5:7b",
                 fallback_model="qwen2.5:3b", timeout_s=30, session=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_model = fallback_model
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    def build_messages(self, req: CommandRequest) -> list:
        if req.selection:
            user = f"Instruction: {req.instruction}\n\nSELECTION:\n{req.selection}"
        else:
            user = f"Instruction: {req.instruction}"
        return [
            {"role": "system", "content": COMMAND_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _model_chain(self) -> list:
        chain = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            chain.append(self.fallback_model)
        return chain

    def _chat(self, model: str, messages: list) -> str | None:
        try:
            resp = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=self.timeout_s,
            )
            if resp.status_code >= 400:
                return None
            content = resp.json().get("message", {}).get("content", "").strip()
            return content or None
        except requests.RequestException:
            return None

    def run(self, req: CommandRequest) -> str:
        messages = self.build_messages(req)
        for model in self._model_chain():
            text = self._chat(model, messages)
            if text is not None:
                return text
        return ""
