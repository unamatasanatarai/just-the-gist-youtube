from pathlib import Path

import pytest

from yt_summarize.naming import make_slug, summary_path, transcript_path

ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "title, expected",
    [
        (
            "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)",
            "rick-astley-never-gonna-give-you-up-official-video-4k-remaster",
        ),
        ("Zażółć gęślą jaźń — Łódź!", "zazolc-gesla-jazn-lodz"),
        ("Straße über", "strasse-uber"),
        ("日本語のタイトル", "ri-ben-yu-notaitoru"),
        ("Привет мир", "privet-mir"),
    ],
)
def test_slug_transliterates(title, expected):
    assert make_slug(title, ID) == expected


@pytest.mark.parametrize("title", ["🚀🔥", "", None, "!!! ???"])
def test_empty_slug_falls_back_to_video_id(title):
    assert make_slug(title, ID) == ID


def test_long_title_capped_on_word_boundary():
    slug = make_slug("word " * 100, ID)
    assert len(slug) <= 80
    assert not slug.endswith("-")
    assert all(part == "word" for part in slug.split("-"))


def test_long_single_word_capped():
    assert make_slug("x" * 300, ID) == "x" * 80


def test_paths():
    assert transcript_path("transcriptions", "abc") == Path("transcriptions/abc.txt")
    assert summary_path(Path("summaries"), "abc") == Path("summaries/abc.md")
