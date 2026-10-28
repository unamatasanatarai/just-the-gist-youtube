"""Parse a YouTube URL or bare video ID into the 11-character video ID (pure)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

_ID_RE = re.compile(r"^[0-9A-Za-z_-]{11}$")

_YOUTUBE_HOSTS = {
    "youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "youtubekids.com",
    "m.youtubekids.com",
}
_PATH_PREFIXES = ("shorts", "embed", "live", "v", "e")


def _is_id(value: str) -> bool:
    return bool(_ID_RE.match(value))


def parse_video_id(arg: str) -> str | None:
    """Return the video ID for a YouTube URL or bare ID, or None if not recognised.

    Handles ``watch?v=``, ``youtu.be/``, ``shorts/``, ``embed/``, ``live/``,
    ``v/``, ``e/``, ``m.``/``music.``/``www.`` hosts, youtube-nocookie and
    youtubekids, scheme-less URLs, and bare 11-character IDs. Extra query
    parameters (``t=``, ``list=``, ``si=``) are ignored.
    """
    arg = arg.strip()
    if not arg:
        return None
    if _is_id(arg):
        return arg

    url = arg if "://" in arg else f"https://{arg}"
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None

    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    parts = [p for p in parsed.path.split("/") if p]

    if host == "youtu.be":
        candidate = parts[0] if parts else ""
        return candidate if _is_id(candidate) else None

    if host not in _YOUTUBE_HOSTS:
        return None

    if parts and parts[0] == "watch":
        values = parse_qs(parsed.query).get("v", [])
        candidate = values[0] if values else ""
        return candidate if _is_id(candidate) else None

    if len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
        candidate = parts[1]
        return candidate if _is_id(candidate) else None

    return None


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"
