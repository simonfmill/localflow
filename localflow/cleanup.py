"""Ollama-backed transcript cleanup: fillers out, punctuation in, nothing added.

Talks to Ollama's /api/chat endpoint with a strict system prompt plus six
worked examples. Fails open: if the primary and fallback models are both
unreachable, the raw transcript is returned unchanged so dictation still works.
"""

import requests

from localflow.contracts import CleanupRequest, CleanupResult

SYSTEM_PROMPT = """You are a dictation cleanup engine. The user message is a raw \
speech-to-text transcript. Return ONLY the cleaned-up text — no explanations, \
no quotes, no preamble.

Rules:
1. Remove filler words and false starts: "um", "uh", "er", "you know", "like" \
(when used as filler), and immediately repeated words.
2. Add correct punctuation and capitalization.
3. Never add new information, never answer questions contained in the text, \
never paraphrase or summarize. Every content word the speaker said stays, in \
the speaker's order.
4. Apply explicit self-corrections: if the speaker says "scratch that" or \
"no wait", keep only the corrected wording.
5. Turn clearly spoken enumerations ("first ... second ... third ...") into a \
numbered list, one item per line.
6. Spell personal-dictionary terms exactly as given, even if the transcript \
spelled them differently.
7. Reply in the same language as the transcript. Never translate: a German \
transcript stays German, cleaned up with German punctuation and capitalization.

Examples:
Input: um so hey team uh lets ship this on friday
Output: Hey team, let's ship this on Friday.

Input: i think we should um you know try the second option
Output: I think we should try the second option.

Input: the meeting is at 3 pm no wait scratch that 4 pm tomorrow
Output: The meeting is at 4 PM tomorrow.

Input: we need three things first the budget second the timeline third the staffing plan
Output: We need three things:
1. The budget
2. The timeline
3. The staffing plan

Input: can you send the deck to sarah before the standup
Output: Can you send the deck to Sarah before the standup?

Input: lets circle back on the qwen rollout with devops
Output: Let's circle back on the Qwen rollout with DevOps."""


class OllamaCleaner:
    def __init__(self, base_url="http://localhost:11434", model="qwen2.5:7b",
                 fallback_model="qwen2.5:3b", timeout_s=30, session=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_model = fallback_model
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    def build_messages(self, req: CleanupRequest) -> list:
        parts = [SYSTEM_PROMPT]
        if req.dictionary:
            parts.append(
                "Personal dictionary — always spell these exactly as written: "
                + ", ".join(req.dictionary)
            )
        if req.profile:
            parts.append("Formatting profile for the target app: " + req.profile)
        if req.context_hint:
            parts.append("Context: " + req.context_hint)
        return [
            {"role": "system", "content": "\n\n".join(parts)},
            {"role": "user", "content": req.raw_text},
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

    def clean(self, req: CleanupRequest) -> CleanupResult:
        messages = self.build_messages(req)
        for model in self._model_chain():
            text = self._chat(model, messages)
            if text is not None:
                return CleanupResult(text=text)
        return CleanupResult(text=req.raw_text)
