import json
from pathlib import Path
from types import SimpleNamespace

from yt_summarize.captions import cues_to_paragraphs, json3_to_text, snippets_to_text

FIXTURE = Path(__file__).parent / "fixtures" / "sample.en.json3"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_json3_fixture_has_no_rolling_duplicates():
    text = json3_to_text(load_fixture())
    assert text.startswith("[Music]")
    assert "We're no strangers to love. You know the rules and so do I." in text
    # The raw joined segments contain 291 words; paragraphing must not add or drop any.
    raw = "".join(s["utf8"] for e in load_fixture()["events"] for s in e.get("segs", []))
    assert text.split() == raw.split()
    assert text.count("We're no strangers to love") == 1


def test_json3_fixture_paragraphs():
    text = json3_to_text(load_fixture())
    paragraphs = text.split("\n\n")
    assert len(paragraphs) > 1
    assert paragraphs[0] == "[Music]"  # followed by a >2 s pause
    assert all("\n" not in p and p == p.strip() and "  " not in p for p in paragraphs)


def test_json3_skips_events_without_segs_and_newlines():
    data = {
        "events": [
            {"tStartMs": 0, "dDurationMs": 10000},
            {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "Hello"}, {"utf8": " world"}]},
            {"tStartMs": 900, "aAppend": 1, "segs": [{"utf8": "\n"}]},
            {"tStartMs": 1000, "dDurationMs": 1000, "segs": [{"utf8": "again."}]},
        ]
    }
    assert json3_to_text(data) == "Hello world again."


def test_json3_empty():
    assert json3_to_text({"events": []}) == ""
    assert json3_to_text({}) == ""


def test_pause_starts_new_paragraph():
    cues = [(0, 1000, "one"), (1200, 2000, "two"), (5000, 6000, "three")]
    assert cues_to_paragraphs(cues) == "one two\n\nthree"


def test_long_paragraph_breaks_at_sentence_end():
    sentence = "This is a sentence of moderate length that keeps going."
    cues = [(i * 1000, i * 1000 + 1000, sentence) for i in range(30)]
    paragraphs = cues_to_paragraphs(cues).split("\n\n")
    assert len(paragraphs) > 1
    assert all(p.endswith(".") for p in paragraphs)


def test_unpunctuated_text_hard_breaks():
    cues = [(i * 1000, i * 1000 + 1000, "words without any punctuation at all") for i in range(100)]
    paragraphs = cues_to_paragraphs(cues).split("\n\n")
    assert len(paragraphs) > 1
    assert all(len(p) < 1300 for p in paragraphs)


def test_snippets_objects_and_dicts():
    snippets = [
        SimpleNamespace(text="[♪♪♪]", start=1.36, duration=1.68),
        SimpleNamespace(text="We're no strangers\nto love", start=18.8, duration=3.0),
        {"text": "You know the rules", "start": 21.9, "duration": 2.0},
    ]
    assert snippets_to_text(snippets) == "[♪♪♪]\n\nWe're no strangers to love You know the rules"
