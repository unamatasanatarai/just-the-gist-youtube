"""Caption data -> plain text paragraphs (pure).

Two inputs are supported:

* yt-dlp ``json3`` caption files (``{"events": [{"tStartMs", "dDurationMs", "segs": [{"utf8"}]}]}``)
* youtube-transcript-api snippets (objects or dicts with ``text``, ``start``, ``duration`` in seconds)

Both are reduced to ``(start_ms, end_ms, text)`` cues and then grouped into
paragraphs: a new paragraph starts after a pause in speech, or once a
paragraph grows long and reaches the end of a sentence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

PAUSE_MS = 2000  # silence that starts a new paragraph
SOFT_PARAGRAPH_CHARS = 600  # break at the next sentence end after this
HARD_PARAGRAPH_CHARS = 1200  # break at the next cue after this regardless

_WS_RE = re.compile(r"\s+")
_SENTENCE_END_RE = re.compile(r"[.!?…][\"')\]]*$")

Cue = tuple[int, int, str]


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def cues_to_paragraphs(cues: Iterable[Cue]) -> str:
    """Group cues into paragraphs separated by blank lines."""
    paragraphs: list[str] = []
    current: list[str] = []
    current_len = 0
    prev_end: int | None = None

    def flush() -> None:
        nonlocal current, current_len
        if current:
            paragraphs.append(" ".join(current))
        current = []
        current_len = 0

    for start, end, raw in cues:
        text = _clean(raw)
        if not text:
            continue
        if current:
            paused = prev_end is not None and start - prev_end >= PAUSE_MS
            last = current[-1]
            long_at_sentence_end = current_len >= SOFT_PARAGRAPH_CHARS and _SENTENCE_END_RE.search(last)
            too_long = current_len >= HARD_PARAGRAPH_CHARS
            if paused or long_at_sentence_end or too_long:
                flush()
        current.append(text)
        current_len += len(text) + 1
        prev_end = end if prev_end is None else max(prev_end, end)

    flush()
    return "\n\n".join(paragraphs)


def json3_cues(data: dict[str, Any]) -> list[Cue]:
    """Turn json3 events into cues. Newline-only ``aAppend`` events become spaces."""
    cues: list[Cue] = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(seg.get("utf8", "") for seg in segs)
        if not text.strip():
            continue
        start = int(event.get("tStartMs", 0))
        end = start + int(event.get("dDurationMs", 0))
        cues.append((start, end, text))
    return cues


def json3_to_text(data: dict[str, Any]) -> str:
    return cues_to_paragraphs(json3_cues(data))


def _get(snippet: Any, name: str, default: Any = None) -> Any:
    if isinstance(snippet, dict):
        return snippet.get(name, default)
    return getattr(snippet, name, default)


def snippets_to_text(snippets: Iterable[Any]) -> str:
    """Turn youtube-transcript-api snippets (``text``, ``start``, ``duration`` seconds) into paragraphs."""
    cues: list[Cue] = []
    for snippet in snippets:
        start = int(round(float(_get(snippet, "start", 0.0)) * 1000))
        duration = int(round(float(_get(snippet, "duration", 0.0)) * 1000))
        cues.append((start, start + duration, str(_get(snippet, "text", ""))))
    return cues_to_paragraphs(cues)
