import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import youtube_transcript_api as yta
from yt_dlp.utils import DownloadError

from yt_summarize import transcript as T
from yt_summarize.transcript import (
    NoCaptionsError,
    Track,
    TranscriptError,
    TranscriptResult,
    choose_track,
    fetch_transcript,
    format_duration,
    render_transcript_file,
    save_transcript,
)

ID = "dQw4w9WgXcQ"
FIXTURE = Path(__file__).parent / "fixtures" / "sample.en.json3"
TITLE = "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)"
SLUG = "rick-astley-never-gonna-give-you-up-official-video-4k-remaster"


def fmts(lang):
    return [
        {"ext": "vtt", "url": f"https://x/{lang}.vtt"},
        {"ext": "json3", "url": f"https://x/{lang}.json3"},
    ]


def tracks(*langs):
    return {lang: fmts(lang) for lang in langs}


def info(subs=(), auto=(), language="en", **extra):
    return {
        "id": ID,
        "title": TITLE,
        "channel": "Rick Astley",
        "duration": 213,
        "language": language,
        "subtitles": tracks(*subs),
        "automatic_captions": tracks(*auto),
        **extra,
    }


# --------------------------------------------------------------------------- choose_track


class TestChooseTrack:
    def test_manual_english_first(self):
        assert choose_track(info(subs=["de-DE", "en", "ja"], auto=["en", "de"])) == Track("en", "manual")

    def test_manual_english_regional_variant(self):
        assert choose_track(info(subs=["de", "en-GB", "en-US"])) == Track("en-GB", "manual")

    def test_auto_english_asr(self):
        assert choose_track(info(auto=["de", "en"], language="en")) == Track("en", "auto")
        assert choose_track(info(auto=["de", "en", "en-orig"], language="en-US")) == Track("en-orig", "auto")

    def test_original_language_manual_before_translated_english(self):
        assert choose_track(info(subs=["pl"], auto=["en", "pl-orig"], language="pl")) == Track("pl", "manual")

    def test_original_language_auto(self):
        assert choose_track(info(auto=["en", "pl", "pl-orig"], language="pl")) == Track("pl-orig", "auto")
        assert choose_track(info(auto=["en", "pl"], language="pl")) == Track("pl", "auto")

    def test_any_other_manual_before_translated_english(self):
        assert choose_track(info(subs=["fr", "de"], auto=["en"], language="pl")) == Track("de", "manual")

    def test_machine_translated_english_last_resort(self):
        track = choose_track(info(auto=["en", "fr"], language="pl"))
        assert track == Track("en", "auto", translated=True)
        assert track.label == "en (auto-translated)"

    def test_live_chat_never_chosen(self):
        data = info(language=None)
        data["subtitles"] = {"live_chat": [{"ext": "json", "url": "https://x/chat"}]}
        assert choose_track(data) is None

    def test_tracks_without_json3_ignored(self):
        data = info(language="en")
        data["subtitles"] = {"en": [{"ext": "vtt", "url": "https://x/en.vtt"}]}
        assert choose_track(data) is None

    def test_no_tracks(self):
        assert choose_track(info()) is None
        assert choose_track({"id": ID}) is None

    def test_user_lang_order(self):
        data = info(subs=["en"], auto=["pl-orig", "pl", "en"], language="pl")
        assert choose_track(data, ["pl", "en"]) == Track("pl-orig", "auto")
        assert choose_track(info(subs=["en", "pl"]), ["pl", "en"]) == Track("pl", "manual")
        assert choose_track(info(subs=["en"]), ["de"]) is None

    def test_labels(self):
        assert Track("en", "manual").label == "en (manual)"
        assert Track("en-orig", "auto").label == "en-orig (auto)"


# --------------------------------------------------------------------------- fakes


class FakeYDL:
    def __init__(self, info=None, extract_error=None, urlopen_error=None, payload=None):
        self.info = info
        self.extract_error = extract_error
        self.urlopen_error = urlopen_error
        self.payload = payload if payload is not None else FIXTURE.read_bytes()
        self.opts = None
        self.extracted_url = None
        self.opened = []

    def __call__(self, opts):
        self.opts = opts
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        assert download is False
        self.extracted_url = url
        if self.extract_error:
            raise self.extract_error
        return self.info

    def urlopen(self, url):
        self.opened.append(url)
        if self.urlopen_error:
            raise self.urlopen_error
        return io.BytesIO(self.payload)


def api_transcript(code, generated, snippets=None):
    snippets = snippets or [SimpleNamespace(text="Hello there.", start=0.0, duration=1.0)]
    return SimpleNamespace(
        language_code=code,
        is_generated=generated,
        fetch=lambda: SimpleNamespace(snippets=snippets),
    )


class FakeAPI:
    def __init__(self, transcripts=(), error=None):
        self.transcripts = list(transcripts)
        self.error = error
        self.listed = []

    def __call__(self):
        return self

    def list(self, video_id):
        self.listed.append(video_id)
        if self.error:
            raise self.error
        return iter(self.transcripts)


def no_api():
    raise AssertionError("youtube-transcript-api must not be used")


def run(ydl, api=no_api, langs=None):
    logs, warns = [], []
    result = fetch_transcript(ID, langs, ydl_factory=ydl, api_factory=api, log=logs.append, warn=warns.append)
    return result, logs, warns


# --------------------------------------------------------------------------- fetch_transcript


def test_ytdlp_main_path():
    ydl = FakeYDL(info(subs=["en"], auto=["en"]))
    result, logs, warns = run(ydl)
    assert ydl.extracted_url == f"https://www.youtube.com/watch?v={ID}"
    assert ydl.opts["noplaylist"] is True and ydl.opts["skip_download"] is True
    assert ydl.opened == ["https://x/en.json3"]
    assert result.title == TITLE and result.channel == "Rick Astley" and result.duration == 213
    assert result.captions == "en (manual)"
    assert "We're no strangers to love." in result.text
    assert logs == [f"Fetching: {TITLE}"]
    assert warns == []


def test_ytdlp_auto_track_fetches_from_automatic_captions():
    ydl = FakeYDL(info(auto=["en"]))
    result, _, _ = run(ydl)
    assert result.captions == "en (auto)"
    assert ydl.opened == ["https://x/en.json3"]


def test_no_captions_does_not_try_fallback():
    with pytest.raises(NoCaptionsError, match="no captions"):
        run(FakeYDL(info()))


def test_extract_failure_falls_back_with_local_id():
    api = FakeAPI([api_transcript("de", False), api_transcript("en", True)])
    ydl = FakeYDL(extract_error=DownloadError("ERROR: [youtube] boom"))
    result, logs, warns = run(ydl, api)
    assert api.listed == [ID]
    assert result.title is None
    assert result.captions == "en (auto, via youtube-transcript-api)"
    assert result.text == "Hello there."
    assert any("trying youtube-transcript-api" in w and "[youtube] boom" in w for w in warns)
    assert any("title unknown" in w for w in warns)


def test_json3_failure_falls_back_and_keeps_metadata():
    api = FakeAPI([api_transcript("en", False)])
    ydl = FakeYDL(info(subs=["en"]), urlopen_error=OSError("HTTP Error 429"))
    result, _, warns = run(ydl, api)
    assert result.title == TITLE and result.channel == "Rick Astley"
    assert result.captions == "en (manual, via youtube-transcript-api)"
    assert not any("title unknown" in w for w in warns)


def test_unparsable_json3_falls_back():
    api = FakeAPI([api_transcript("en", False)])
    result, _, _ = run(FakeYDL(info(subs=["en"]), payload=b"<html>"), api)
    assert result.captions.endswith("via youtube-transcript-api)")


def test_fallback_track_priority():
    api = FakeAPI(
        [api_transcript("de", False), api_transcript("en", True), api_transcript("en-US", False)]
    )
    result, _, _ = run(FakeYDL(extract_error=DownloadError("x")), api)
    assert result.captions == "en-US (manual, via youtube-transcript-api)"

    api = FakeAPI([api_transcript("pl", True), api_transcript("de", False)])
    result, _, _ = run(FakeYDL(extract_error=DownloadError("x")), api)
    assert result.captions == "de (manual, via youtube-transcript-api)"

    api = FakeAPI([api_transcript("en", False), api_transcript("pl", True)])
    result, _, _ = run(FakeYDL(extract_error=DownloadError("x")), api, langs=["pl"])
    assert result.captions == "pl (auto, via youtube-transcript-api)"


@pytest.mark.parametrize(
    "error, exc_type, message",
    [
        (yta.TranscriptsDisabled(ID), NoCaptionsError, "no captions"),
        (yta.VideoUnavailable(ID), TranscriptError, "private or unavailable"),
        (yta.AgeRestricted(ID), TranscriptError, "age-restricted"),
        (yta.IpBlocked(ID), TranscriptError, "blocking requests"),
        (yta.RequestBlocked(ID), TranscriptError, "blocking requests"),
        (yta.YouTubeRequestFailed(ID, Exception("x")), TranscriptError, "could not reach YouTube"),
    ],
)
def test_fallback_errors_mapped(error, exc_type, message):
    with pytest.raises(exc_type, match=message):
        run(FakeYDL(extract_error=DownloadError("ERROR: Video unavailable")), FakeAPI(error=error))


def test_fallback_with_no_matching_track_is_no_captions():
    with pytest.raises(NoCaptionsError):
        run(FakeYDL(extract_error=DownloadError("x")), FakeAPI([]))


# --------------------------------------------------------------------------- file output


def result(**overrides):
    base = dict(
        video_id=ID, title=TITLE, channel="Rick Astley", duration=213, captions="en (manual)", text="Hello.\n\nWorld."
    )
    return TranscriptResult(**{**base, **overrides})


def test_format_duration():
    assert format_duration(213) == "3:33"
    assert format_duration(59) == "0:59"
    assert format_duration(3725) == "1:02:05"
    assert format_duration(None) is None


def test_render_transcript_file():
    assert render_transcript_file(result()) == (
        f"Title: {TITLE}\n"
        "Channel: Rick Astley\n"
        f"URL: https://www.youtube.com/watch?v={ID}\n"
        "Duration: 3:33\n"
        "Captions: en (manual)\n"
        "\n"
        "Hello.\n\nWorld.\n"
    )


def test_render_unknown_metadata():
    text = render_transcript_file(result(title=None, channel=None, duration=None))
    assert text.splitlines()[:3] == ["Title: (unknown)", f"URL: https://www.youtube.com/watch?v={ID}", "Captions: en (manual)"]


def test_save_overwrites_and_reports(tmp_path):
    out = tmp_path / "transcriptions"
    path, overwrote = save_transcript(result(), out)
    assert path == out / f"{SLUG}.txt" and not overwrote
    path2, overwrote2 = save_transcript(result(text="New."), out)
    assert path2 == path and overwrote2
    assert path.read_text(encoding="utf-8").endswith("\nNew.\n")


def test_save_uses_video_id_for_empty_slug(tmp_path):
    path, _ = save_transcript(result(title="🚀🔥"), tmp_path)
    assert path.name == f"{ID}.txt"
    path, _ = save_transcript(result(title=None), tmp_path)
    assert path.name == f"{ID}.txt"


# --------------------------------------------------------------------------- CLI


def test_main_rejects_non_youtube(capsys):
    assert T.main(["https://example.com/video"]) == 2
    assert "not a YouTube video" in capsys.readouterr().err


def test_main_success_prints_next_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(T.fetch_transcript.__kwdefaults__, "ydl_factory", FakeYDL(info(subs=["en"])))
    out_dir = tmp_path / "transcriptions"
    assert T.main([f"https://youtu.be/{ID}?si=abc", "--out-dir", str(out_dir)]) == 0
    out = capsys.readouterr().out
    expected = out_dir / f"{SLUG}.txt"
    assert f"Fetching: {TITLE}" in out
    assert "Captions: en (manual)" in out
    assert f"Saved {expected}" in out
    assert f"Next: uv run yt-summarize {expected}" in out
    assert expected.exists()


def test_main_no_captions_exits_nonzero_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(T.fetch_transcript.__kwdefaults__, "ydl_factory", FakeYDL(info()))
    out_dir = tmp_path / "transcriptions"
    assert T.main([ID, "--out-dir", str(out_dir)]) == 1
    captured = capsys.readouterr()
    assert "Error: this video has no captions" in captured.err
    assert not out_dir.exists()


def test_transcript_module_does_not_import_summarizer():
    source = Path(T.__file__).read_text(encoding="utf-8")
    assert "summarize import" not in source and "import yt_summarize.summarize" not in source
