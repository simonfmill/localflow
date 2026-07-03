import json

from localflow.dictionary import PersonalDictionary


def make(tmp_path):
    return PersonalDictionary(tmp_path / "dictionary.json")


def test_starts_empty_and_persists_adds(tmp_path):
    d = make(tmp_path)
    assert d.terms == []
    assert d.add("Qwen") is True
    assert d.add("Sarah") is True
    reloaded = make(tmp_path)
    assert reloaded.terms == ["Qwen", "Sarah"]


def test_add_deduplicates(tmp_path):
    d = make(tmp_path)
    d.add("Qwen")
    assert d.add("Qwen") is False
    assert d.terms == ["Qwen"]


def test_remove(tmp_path):
    d = make(tmp_path)
    d.add("Qwen")
    assert d.remove("Qwen") is True
    assert d.remove("Qwen") is False
    assert make(tmp_path).terms == []


def test_render(tmp_path):
    d = make(tmp_path)
    d.add("Qwen")
    d.add("Sarah")
    assert d.render() == "Qwen, Sarah"


def test_corrupt_file_yields_empty_dictionary(tmp_path):
    path = tmp_path / "dictionary.json"
    path.write_text("{not json")
    assert PersonalDictionary(path).terms == []


def test_saved_file_is_json_list(tmp_path):
    d = make(tmp_path)
    d.add("Qwen")
    assert json.loads((tmp_path / "dictionary.json").read_text()) == ["Qwen"]


def test_observe_correction_learns_proper_nouns(tmp_path):
    d = make(tmp_path)
    added = d.observe_correction(
        pasted="meet with sara about the qwen rollout",
        corrected="Meet with Sarah about the Qwen rollout",
    )
    # "Sarah" is a new mid-sentence capitalized word; "Qwen" is a re-cased word.
    # "Meet" is plain sentence-start capitalization and must be skipped.
    assert added == ["Sarah", "Qwen"]
    assert make(tmp_path).terms == ["Sarah", "Qwen"]


def test_observe_correction_ignores_unchanged_text(tmp_path):
    d = make(tmp_path)
    text = "Ship it on Friday."
    assert d.observe_correction(text, text) == []


def test_observe_correction_skips_known_terms(tmp_path):
    d = make(tmp_path)
    d.add("Qwen")
    added = d.observe_correction("the qwen rollout", "the Qwen rollout")
    assert added == []
