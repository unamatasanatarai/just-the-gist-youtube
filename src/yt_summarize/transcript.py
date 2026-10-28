"""yt-transcript: YouTube URL/ID -> captions -> transcriptions/<slug>.txt.

yt-dlp is the primary source (metadata + json3 captions). youtube-transcript-api
is used only when yt-dlp raises (extract_info or the json3 download fails).
This command never runs the summarizer; it only prints a "Next:" hint.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yt_summarize.captions import json3_to_text, snippets_to_text
from yt_summarize.naming import make_slug, transcript_path
from yt_summarize.video_id import parse_video_id, watch_url

DEFAULT_OUT_DIR = "transcriptions"
UNKNOWN_TITLE = "(unknown)"
NO_CAPTIONS_MESSAGE = (
    "this video has no captions (manual or auto-generated), so there is no transcript to summarize."
)

Log = Callable[[str], None]


class TranscriptError(Exception):
    """A failure with a short, user-facing message."""


class NoCaptionsError(TranscriptError):
    def __init__(self, message: str = NO_CAPTIONS_MESSAGE) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class Track:
    lang: str
    kind: str  # "manual" or "auto"
    translated: bool = False

    @property
    def label(self) -> str:
        kind = "auto-translated" if self.translated else self.kind
        return f"{self.lang} ({kind})"


@dataclass
class TranscriptResult:
    video_id: str
    title: str | None
    channel: str | None
    duration: int | None
    captions: str  # e.g. "en (manual)" or "en (auto, via youtube-transcript-api)"
    text: str


# --------------------------------------------------------------------------- track choice


def _base(code: str | None) -> str | None:
    return code.split("-")[0].lower() if code else None


def _usable(tracks: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Caption tracks that have a json3 format, excluding live chat."""
    usable: dict[str, list[dict[str, Any]]] = {}
    for lang, formats in (tracks or {}).items():
        if lang.startswith("live_chat") or not formats:
            continue
        if any(f.get("ext") == "json3" and f.get("url") for f in formats):
            usable[lang] = formats
    return usable


def _manual_matches(manual: dict[str, Any], lang: str) -> list[str]:
    """Manual track keys for a language: exact code first, then regional variants (en-US, en-GB)."""
    lang = lang.lower()
    exact = [k for k in manual if k.lower() == lang]
    variants = sorted(k for k in manual if k.lower().startswith(lang + "-") and k not in exact)
    return exact + variants


def _auto_track(auto: dict[str, Any], key: str, video_lang: str | None) -> Track:
    translated = (
        not key.endswith("-orig") and video_lang is not None and _base(key) != _base(video_lang)
    )
    return Track(key, "auto", translated)


def choose_track(info: dict[str, Any], langs: Sequence[str] | None = None) -> Track | None:
    """Pick the caption track to use, or None when there is no usable track.

    Default priority (English first):
      1. manual English (en, en-*)
      2. auto English speech recognition (en-orig, or en when the video is English)
      3. manual track in the video's original language
      4. auto original-language track (<lang>-orig or <lang>)
      5. any other manual track
      6. YouTube's machine-translated English (auto 'en' on a non-English video)

    With ``langs``, the user's order replaces this list: for each language,
    manual tracks first, then auto-generated ones.
    """
    manual = _usable(info.get("subtitles"))
    auto = _usable(info.get("automatic_captions"))
    video_lang = info.get("language")
    video_base = _base(video_lang)

    if langs:
        for lang in langs:
            keys = _manual_matches(manual, lang)
            if keys:
                return Track(keys[0], "manual")
            for key in (f"{lang}-orig", lang):
                if key in auto:
                    return _auto_track(auto, key, video_lang)
        return None

    # 1. manual English
    keys = _manual_matches(manual, "en")
    if keys:
        return Track(keys[0], "manual")
    # 2. auto English ASR
    if "en-orig" in auto:
        return Track("en-orig", "auto")
    if "en" in auto and video_base == "en":
        return Track("en", "auto")
    if video_base and video_base != "en":
        # 3. manual original language
        keys = _manual_matches(manual, video_lang) or _manual_matches(manual, video_base)
        if keys:
            return Track(keys[0], "manual")
        # 4. auto original language
        for key in dict.fromkeys((f"{video_lang}-orig", f"{video_base}-orig", video_lang, video_base)):
            if key in auto:
                return Track(key, "auto")
    # 5. any other manual track
    if manual:
        return Track(sorted(manual)[0], "manual")
    # (language unknown) any auto speech-recognition track
    orig = sorted(k for k in auto if k.endswith("-orig"))
    if orig:
        return Track(orig[0], "auto")
    # 6. machine-translated English, last resort
    if "en" in auto:
        return _auto_track(auto, "en", video_lang)
    return None


def _json3_url(formats: list[dict[str, Any]]) -> str:
    for fmt in formats:
        if fmt.get("ext") == "json3" and fmt.get("url"):
            return fmt["url"]
    raise TranscriptError("no json3 caption format")  # unreachable after _usable()


# --------------------------------------------------------------------------- yt-dlp path


class _SilentLogger:
    def debug(self, msg: str) -> None:
        pass

    info = warning = error = debug


def _ytdlp_options(verbose: bool) -> dict[str, Any]:
    opts: dict[str, Any] = {"skip_download": True, "noplaylist": True}
    if not verbose:
        opts.update(quiet=True, no_warnings=True, logger=_SilentLogger())
    return opts


def _default_ydl_factory(opts: dict[str, Any]) -> Any:
    import yt_dlp

    return yt_dlp.YoutubeDL(opts)


def _default_api_factory() -> Any:
    from youtube_transcript_api import YouTubeTranscriptApi

    return YouTubeTranscriptApi()


def _short_error(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return msg.removeprefix("ERROR: ")


# --------------------------------------------------------------------------- fallback path


def _pick_api_transcript(transcript_list: Any, langs: Sequence[str] | None) -> Any | None:
    transcripts = list(transcript_list)

    def matching(lang: str, generated: bool) -> list[Any]:
        lang = lang.lower()
        found = [
            t
            for t in transcripts
            if t.is_generated == generated
            and (t.language_code.lower() == lang or t.language_code.lower().startswith(lang + "-"))
        ]
        return sorted(found, key=lambda t: t.language_code.lower() != lang)

    if langs:
        for lang in langs:
            for generated in (False, True):
                found = matching(lang, generated)
                if found:
                    return found[0]
        return None

    for found in (matching("en", False), matching("en", True)):
        if found:
            return found[0]
    for generated in (False, True):
        for t in transcripts:
            if t.is_generated == generated:
                return t
    return None


def _fetch_with_api(
    video_id: str,
    langs: Sequence[str] | None,
    api_factory: Callable[[], Any],
    ytdlp_error: BaseException,
) -> tuple[str, str]:
    """Return (captions label, text) using youtube-transcript-api, mapping its errors."""
    import youtube_transcript_api as yta

    detail = f"(yt-dlp: {_short_error(ytdlp_error)})"
    try:
        api = api_factory()
        transcript = _pick_api_transcript(api.list(video_id), langs)
        if transcript is None:
            raise NoCaptionsError()
        fetched = transcript.fetch()
    except (yta.TranscriptsDisabled, yta.NoTranscriptFound) as exc:
        raise NoCaptionsError() from exc
    except (yta.VideoUnavailable, yta.VideoUnplayable, yta.InvalidVideoId) as exc:
        raise TranscriptError(f"the video is private or unavailable {detail}") from exc
    except yta.AgeRestricted as exc:
        raise TranscriptError(f"the video is age-restricted and needs a signed-in account {detail}") from exc
    except yta.RequestBlocked as exc:
        raise TranscriptError(
            f"YouTube is blocking requests from this IP address; try again later {detail}"
        ) from exc
    except yta.YouTubeTranscriptApiException as exc:
        raise TranscriptError(
            f"could not reach YouTube {detail} (youtube-transcript-api: {type(exc).__name__})"
        ) from exc

    snippets = getattr(fetched, "snippets", fetched)
    text = snippets_to_text(snippets)
    if not text:
        raise NoCaptionsError()
    kind = "auto" if transcript.is_generated else "manual"
    return f"{transcript.language_code} ({kind}, via youtube-transcript-api)", text


# --------------------------------------------------------------------------- orchestration


def fetch_transcript(
    video_id: str,
    langs: Sequence[str] | None = None,
    *,
    ydl_factory: Callable[[dict[str, Any]], Any] = _default_ydl_factory,
    api_factory: Callable[[], Any] = _default_api_factory,
    log: Log = print,
    warn: Log | None = None,
    verbose: bool = False,
) -> TranscriptResult:
    warn = warn or (lambda m: print(m, file=sys.stderr))
    info: dict[str, Any] | None = None
    try:
        with ydl_factory(_ytdlp_options(verbose)) as ydl:
            info = ydl.extract_info(watch_url(video_id), download=False)
            log(f"Fetching: {info.get('title') or UNKNOWN_TITLE}")
            track = choose_track(info, langs)
            if track is None:
                raise NoCaptionsError()
            source = info["subtitles"] if track.kind == "manual" else info["automatic_captions"]
            raw = ydl.urlopen(_json3_url(source[track.lang])).read()
            text = json3_to_text(json.loads(raw))
            if not text:
                raise ValueError("the json3 caption file contained no text")
            return TranscriptResult(
                video_id=info.get("id") or video_id,
                title=info.get("title"),
                channel=info.get("channel") or info.get("uploader"),
                duration=info.get("duration"),
                captions=track.label,
                text=text,
            )
    except NoCaptionsError:
        raise
    except Exception as exc:  # yt-dlp raised (extract or json3 fetch): fall back
        ytdlp_error = exc

    warn(f"yt-dlp failed ({_short_error(ytdlp_error)}); trying youtube-transcript-api...")
    fallback_id = (info or {}).get("id") or video_id
    captions, text = _fetch_with_api(fallback_id, langs, api_factory, ytdlp_error)
    if info is None:
        warn("Warning: video title unknown (yt-dlp failed); naming the file by video ID.")
    return TranscriptResult(
        video_id=fallback_id,
        title=(info or {}).get("title"),
        channel=(info or {}).get("channel") or (info or {}).get("uploader"),
        duration=(info or {}).get("duration"),
        captions=captions,
        text=text,
    )


def format_duration(seconds: int | float | None) -> str | None:
    if seconds is None:
        return None
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def render_transcript_file(result: TranscriptResult) -> str:
    lines = [f"Title: {result.title or UNKNOWN_TITLE}"]
    if result.channel:
        lines.append(f"Channel: {result.channel}")
    lines.append(f"URL: {watch_url(result.video_id)}")
    duration = format_duration(result.duration)
    if duration:
        lines.append(f"Duration: {duration}")
    lines.append(f"Captions: {result.captions}")
    return "\n".join(lines) + "\n\n" + result.text.strip() + "\n"


def save_transcript(result: TranscriptResult, out_dir: Path | str) -> tuple[Path, bool]:
    """Write transcriptions/<slug>.txt, always overwriting. Returns (path, overwrote)."""
    path = transcript_path(out_dir, make_slug(result.title, result.video_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    path.write_text(render_transcript_file(result), encoding="utf-8")
    return path, existed


# --------------------------------------------------------------------------- CLI


def _parse_langs(value: str) -> list[str]:
    langs = [part.strip() for part in value.split(",") if part.strip()]
    if not langs:
        raise argparse.ArgumentTypeError("expected a comma-separated list such as 'pl,en'")
    return langs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-transcript",
        description="Fetch a YouTube video's captions and save them to transcriptions/<slug>.txt.",
    )
    parser.add_argument("video", help="YouTube URL (watch, youtu.be, shorts, embed, live) or 11-character video ID")
    parser.add_argument(
        "--lang",
        type=_parse_langs,
        metavar="CODES",
        help="caption languages in priority order, e.g. 'pl,en' (default: English first, then the original language)",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"directory for transcript files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show yt-dlp's own output and warnings")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    video_id = parse_video_id(args.video)
    if video_id is None:
        print(f"Error: not a YouTube video URL or ID: {args.video}", file=sys.stderr)
        return 2

    try:
        result = fetch_transcript(video_id, args.lang, verbose=args.verbose)
    except TranscriptError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Captions: {result.captions}")
    path, overwrote = save_transcript(result, args.out_dir)
    print(f"{'Overwrote' if overwrote else 'Saved'} {path}")
    print(f"Next: uv run yt-summarize {shlex.quote(str(path))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
