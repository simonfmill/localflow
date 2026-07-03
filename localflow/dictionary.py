"""Personal dictionary: a JSON list of terms the cleanup prompt must spell exactly.

Learns from corrections: when the user edits pasted text, proper nouns that
appear in the corrected version but not in what was pasted are added
automatically.
"""

import json
from pathlib import Path

_STRIP_CHARS = ".,!?;:\"'()[]{}"


def _core(token: str) -> str:
    return token.strip(_STRIP_CHARS)


class PersonalDictionary:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.terms: list = []
        self.load()

    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.terms = [t for t in data if isinstance(t, str)]
            except (json.JSONDecodeError, OSError):
                self.terms = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.terms, indent=2, ensure_ascii=False))

    def add(self, term: str) -> bool:
        term = term.strip()
        if not term or term in self.terms:
            return False
        self.terms.append(term)
        self.save()
        return True

    def remove(self, term: str) -> bool:
        if term in self.terms:
            self.terms.remove(term)
            self.save()
            return True
        return False

    def render(self) -> str:
        return ", ".join(self.terms)

    def observe_correction(self, pasted: str, corrected: str) -> list:
        """Learn proper nouns from a user's post-paste edit.

        Adds capitalized words that are new or re-cased relative to the pasted
        text, skipping plain sentence-start capitalizations.
        """
        pasted_tokens = [_core(t) for t in pasted.split()]
        pasted_exact = set(pasted_tokens)
        pasted_lower = {t.lower() for t in pasted_tokens}
        added = []
        raw_tokens = corrected.split()
        for idx, raw in enumerate(raw_tokens):
            word = _core(raw)
            if len(word) < 2 or not word[0].isupper():
                continue
            if word in pasted_exact or word in self.terms or word in added:
                continue
            sentence_start = idx == 0 or raw_tokens[idx - 1].endswith((".", "!", "?"))
            if word.lower() in pasted_lower:
                # Re-cased existing word — a proper-noun correction, unless it is
                # just first-letter capitalization at a sentence start.
                if sentence_start and word == word.lower().capitalize():
                    continue
                added.append(word)
            elif not sentence_start:
                # Brand-new capitalized word mid-sentence — likely a proper noun.
                added.append(word)
        for word in added:
            self.add(word)
        return added
