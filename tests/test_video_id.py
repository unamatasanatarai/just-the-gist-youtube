import pytest

from yt_summarize.video_id import parse_video_id, watch_url

ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "arg",
    [
        ID,
        f"  {ID}  ",
        f"https://www.youtube.com/watch?v={ID}",
        f"https://www.youtube.com/watch?v={ID}&t=10&list=PL123&si=abc",
        f"https://www.youtube.com/watch?feature=share&v={ID}",
        f"youtube.com/watch?v={ID}",
        f"http://m.youtube.com/watch?v={ID}",
        f"https://music.youtube.com/watch?v={ID}",
        f"https://youtu.be/{ID}",
        f"https://youtu.be/{ID}?si=abc&t=10",
        f"youtu.be/{ID}",
        f"https://www.youtube.com/shorts/{ID}",
        f"https://www.youtube.com/embed/{ID}?start=5",
        f"https://www.youtube-nocookie.com/embed/{ID}",
        f"https://www.youtube.com/live/{ID}?feature=share",
        f"https://www.youtube.com/v/{ID}",
        f"https://www.youtubekids.com/watch?v={ID}",
    ],
)
def test_parses_known_forms(arg):
    assert parse_video_id(arg) == ID


def test_id_with_leading_dash():
    assert parse_video_id("-abcdefghij") == "-abcdefghij"


@pytest.mark.parametrize(
    "arg",
    [
        "",
        "hello",
        f"{ID}x",  # bare ID with trailing text
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://vimeo.com/123456",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/@RickAstleyYT",
        "https://www.youtube.com/playlist?list=PL123",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/",
    ],
)
def test_rejects_non_videos(arg):
    assert parse_video_id(arg) is None


def test_watch_url():
    assert watch_url(ID) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
