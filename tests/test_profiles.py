from localflow.contracts import APP_KINDS
from localflow.profiles import PROFILES, fragment_for


def test_every_app_kind_has_a_profile():
    for kind in APP_KINDS:
        assert kind in PROFILES
        assert PROFILES[kind].strip()


def test_fragment_for_known_kinds():
    assert "email" in fragment_for("email")
    assert "chat" in fragment_for("chat")
    assert fragment_for("code") == PROFILES["code"]


def test_unknown_kind_falls_back_to_generic():
    assert fragment_for("spreadsheet") == PROFILES["generic"]
    assert fragment_for("") == PROFILES["generic"]


def test_raw_profiles_forbid_reformatting():
    for kind in ("code", "terminal"):
        fragment = fragment_for(kind).lower()
        assert "verbatim" in fragment
