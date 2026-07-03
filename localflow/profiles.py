"""Per-app formatting profiles rendered as cleanup-prompt fragments."""

PROFILES = {
    "email": (
        "The text goes into an email. Use a polished, professional tone: full "
        "sentences, proper greetings kept as spoken, standard punctuation and "
        "paragraph breaks where natural."
    ),
    "chat": (
        "The text goes into a chat message (Slack/Discord style). Keep it casual "
        "and conversational: contractions are fine, light punctuation, no stiff "
        "formalities, keep it as one short message unless the speaker clearly "
        "dictated multiple lines."
    ),
    "code": (
        "The text goes into a code editor. Return the spoken words verbatim: do "
        "NOT auto-capitalize, do NOT add sentence punctuation, do NOT reflow — "
        "only remove filler words."
    ),
    "terminal": (
        "The text goes into a terminal. Treat it as a shell command: verbatim "
        "and lowercase unless the speaker spelled a capital, no punctuation "
        "added, no trailing period."
    ),
    "generic": (
        "Use neutral, clear prose with standard punctuation and capitalization."
    ),
}


def fragment_for(kind: str) -> str:
    return PROFILES.get(kind, PROFILES["generic"])
