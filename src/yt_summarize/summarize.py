"""yt-summarize: transcript .txt -> Gemini (model rotation) -> stdout + summaries/<slug>.md.

Works on any transcript file in the yt-transcript format: "Key: value" header
lines, a blank line, then the transcript text. The API key is GEMINI_API_KEY,
read from the environment or from .env in the current directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from yt_summarize.naming import summary_path

DEFAULT_OUT_DIR = "summaries"
DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",  # shutdown 2027-05-07; drop then
]
REQUEST_TIMEOUT_MS = 180_000
MISSING_KEY_MESSAGE = (
    "GEMINI_API_KEY is not set. Add it to .env or export it. "
    "Get a key at https://aistudio.google.com/apikey"
)
UNKNOWN_TITLE = "(unknown)"

SYSTEM_INSTRUCTION = """\
You summarize YouTube video transcripts. The input starts with metadata header \
lines (Title, Channel, URL, Duration, Captions) followed by the transcript, \
which may be auto-generated, unpunctuated, or in any language.

Always write your answer in English, whatever language the transcript is in.
Return JSON with these fields:
- tldr: one sentence capturing the main point of the video.
- summary: 2-4 paragraphs (separate paragraphs with a blank line) explaining what the video covers.
- key_points: 5-10 concise, self-contained key points or takeaways, in the order they appear.
- notable_quotes: short verbatim quotes worth remembering (translated to English if needed); empty list if none stand out.
- topics: 3-8 short topic tags (one to three words each).
Base everything only on the transcript; do not invent facts. Ignore caption \
artifacts such as [Music] or repeated filler words."""


class VideoSummary(BaseModel):
    tldr: str = Field(description="One sentence.")
    summary: str = Field(description="2-4 paragraphs.")
    key_points: list[str] = Field(description="5-10 items.")
    notable_quotes: list[str] = Field(description="May be empty.")
    topics: list[str] = Field(description="Short tags.")


# --------------------------------------------------------------------------- transcript file


def parse_header(text: str) -> dict[str, str]:
    """Read the leading "Key: value" lines up to the first blank line."""
    header: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if not sep or not key.strip() or " " in key.strip():
            break  # not a header line: the file has no (more) header
        header[key.strip().lower()] = value.strip()
    return header


# --------------------------------------------------------------------------- models


def resolve_models(flag: str | None, env: Mapping[str, str]) -> list[str]:
    """--model wins over GEMINI_MODELS, which wins over DEFAULT_MODELS."""
    for raw in (flag, env.get("GEMINI_MODELS")):
        if raw:
            models = [m.strip() for m in raw.split(",") if m.strip()]
            if models:
                return models
    return list(DEFAULT_MODELS)


@dataclass
class Attempt:
    model: str
    error: str


class SummarizeError(Exception):
    """Base for failures with a user-facing message."""


class FatalModelError(SummarizeError):
    """An error another model would not fix (400/401/403)."""


class AllModelsFailed(SummarizeError):
    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        lines = ["every model in the rotation failed:"]
        lines += [f"  {a.model}: {a.error}" for a in attempts]
        super().__init__("\n".join(lines))


def _api_error_text(exc: Any) -> str:
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = getattr(exc, "message", None) or str(exc)
    prefix = " ".join(str(p) for p in (code, status) if p)
    return f"{prefix}: {message}" if prefix else str(message)


def _short_reason(exc: Any) -> str:
    parts = [str(p) for p in (getattr(exc, "code", None), getattr(exc, "status", None)) if p]
    return " ".join(parts) or type(exc).__name__


def summarize(
    transcript_text: str,
    models: Sequence[str],
    client: Any,
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[VideoSummary, str]:
    """Try each model once, in order. Returns (summary, model that answered).

    Rotation rules: 5xx, 429, 404, invalid JSON and network errors move on to
    the next model; any other 4xx (400/401/403) stops immediately.
    """
    import httpx
    from google.genai import errors, types

    log = log or (lambda m: print(m, file=sys.stderr))
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_json_schema=VideoSummary.model_json_schema(),
    )
    attempts: list[Attempt] = []
    for index, model in enumerate(models):
        if index == 0:
            log(f"Summarizing with {model}...")
        try:
            response = client.models.generate_content(model=model, contents=transcript_text, config=config)
            summary = VideoSummary.model_validate_json(response.text or "")
            return summary, model
        except errors.ServerError as exc:
            attempts.append(Attempt(model, _api_error_text(exc)))
            reason = _short_reason(exc)
        except errors.ClientError as exc:
            if exc.code not in (404, 429):
                raise FatalModelError(f"{model}: {_api_error_text(exc)}") from exc
            attempts.append(Attempt(model, _api_error_text(exc)))
            reason = _short_reason(exc)
        except ValidationError as exc:
            attempts.append(Attempt(model, f"invalid JSON reply ({exc.error_count()} validation errors)"))
            reason = "invalid JSON reply"
        except httpx.TransportError as exc:
            attempts.append(Attempt(model, f"network error: {type(exc).__name__}: {exc}"))
            reason = f"network error: {type(exc).__name__}"
        if index + 1 < len(models):
            log(f"{model} unavailable ({reason}), trying {models[index + 1]}...")
        else:
            log(f"{model} unavailable ({reason}).")
    raise AllModelsFailed(attempts)


# --------------------------------------------------------------------------- rendering


def render_markdown(
    summary: VideoSummary,
    header: Mapping[str, str],
    *,
    model: str,
    source: Path | str,
    date: dt.date,
) -> str:
    title = header.get("title")
    if not title or title == UNKNOWN_TITLE:
        title = Path(source).stem
    out = [f"# {title}", ""]

    meta = [header[k] for k in ("channel", "duration", "url") if header.get(k)]
    if meta:
        out += [" · ".join(meta), ""]

    out += [f"> **TL;DR** {summary.tldr.strip()}", ""]
    out += ["## Summary", "", summary.summary.strip(), ""]
    out += ["## Key points", ""]
    out += [f"- {point.strip()}" for point in summary.key_points]
    out.append("")

    quotes = [q.strip().strip('"“”') for q in summary.notable_quotes if q.strip()]
    if quotes:
        out += ["## Notable quotes", ""]
        for quote in quotes:
            out += [f'> "{quote}"', ""]

    if summary.topics:
        out += ["## Topics", "", " · ".join(t.strip() for t in summary.topics), ""]

    out += ["---", f"Generated by {model} from {source} on {date.isoformat()}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- CLI


def make_client(api_key: str) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-summarize",
        description="Summarize a transcript file with Gemini; print Markdown and save summaries/<slug>.md.",
    )
    parser.add_argument("transcript", type=Path, help="path to a transcript .txt (e.g. from yt-transcript)")
    parser.add_argument(
        "--model",
        metavar="MODELS",
        help="comma-separated Gemini models to try in order (overrides GEMINI_MODELS; "
        f"default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"directory for summary files (default: {DEFAULT_OUT_DIR})",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, client_factory: Callable[[str], Any] = make_client) -> int:
    from dotenv import find_dotenv, load_dotenv

    args = build_parser().parse_args(argv)
    path: Path = args.transcript

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    if not text.strip():
        print(f"Error: {path} is empty", file=sys.stderr)
        return 2

    load_dotenv(find_dotenv(usecwd=True), override=False)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(MISSING_KEY_MESSAGE, file=sys.stderr)
        return 2

    models = resolve_models(args.model, os.environ)
    try:
        summary, model = summarize(text, models, client_factory(api_key))
    except SummarizeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    markdown = render_markdown(summary, parse_header(text), model=model, source=path, date=dt.date.today())
    print()
    print(markdown, end="")

    out_path = summary_path(args.out_dir, path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Saved {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
