"""Title -> slug -> file path (pure)."""

from __future__ import annotations

from pathlib import Path

from slugify import slugify

MAX_SLUG_LENGTH = 80


def make_slug(title: str | None, video_id: str) -> str:
    """ASCII lowercase slug of the title, capped at 80 chars on a word boundary.

    Falls back to the video ID when the title is unknown or slugifies to an
    empty string (e.g. an all-emoji title).
    """
    slug = slugify(title or "", max_length=MAX_SLUG_LENGTH, word_boundary=True)
    return slug or video_id


def transcript_path(directory: Path | str, slug: str) -> Path:
    return Path(directory) / f"{slug}.txt"


def summary_path(directory: Path | str, slug: str) -> Path:
    return Path(directory) / f"{slug}.md"
