import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from yt_summarize import summarize as S
from yt_summarize.summarize import (
    DEFAULT_MODELS,
    MISSING_KEY_MESSAGE,
    AllModelsFailed,
    FatalModelError,
    VideoSummary,
    parse_header,
    render_markdown,
    resolve_models,
    summarize,
)

TRANSCRIPT = """\
Title: Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)
Channel: Rick Astley
URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Duration: 3:33
Captions: en (manual)

We're no strangers to love. You know the rules and so do I.
"""

CANNED = {
    "tldr": "A man promises never to give you up.",
    "summary": "Paragraph one.\n\nParagraph two.",
    "key_points": ["He will not give you up.", "He will not let you down."],
    "notable_quotes": ["Never gonna give you up"],
    "topics": ["music", "1980s pop"],
}


def server_error(code=503, status="UNAVAILABLE", message="The model is overloaded."):
    return errors.ServerError(code, {"error": {"code": code, "status": status, "message": message}})


def client_error(code, status, message="nope"):
    return errors.ClientError(code, {"error": {"code": code, "status": status, "message": message}})


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        outcome = self.outcomes[model]
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(text=outcome)


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


OK = json.dumps(CANNED)


def run(outcomes, models=None):
    models = models or list(outcomes)
    client = FakeClient(outcomes)
    logs = []
    try:
        return summarize(TRANSCRIPT, models, client, log=logs.append), client, logs
    except Exception as exc:
        exc.client, exc.logs = client, logs
        raise


# --------------------------------------------------------------------------- header / models


def test_parse_header():
    assert parse_header(TRANSCRIPT) == {
        "title": "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)",
        "channel": "Rick Astley",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "duration": "3:33",
        "captions": "en (manual)",
    }


def test_parse_header_without_header():
    assert parse_header("Just some text: with a colon.\nMore text.") == {}
    assert parse_header("") == {}


def test_resolve_models_precedence():
    assert resolve_models(None, {}) == DEFAULT_MODELS
    assert resolve_models(None, {"GEMINI_MODELS": "a, b"}) == ["a", "b"]
    assert resolve_models("x", {"GEMINI_MODELS": "a,b"}) == ["x"]
    assert resolve_models(" , ", {"GEMINI_MODELS": ""}) == DEFAULT_MODELS


def test_default_models():
    assert DEFAULT_MODELS == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


# --------------------------------------------------------------------------- rotation


def test_first_model_success_sends_transcript_and_schema():
    (summary, model), client, logs = run({"m1": OK, "m2": OK})
    assert model == "m1"
    assert summary == VideoSummary(**CANNED)
    assert len(client.models.calls) == 1
    call = client.models.calls[0]
    assert call.contents == TRANSCRIPT
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_json_schema == VideoSummary.model_json_schema()
    assert "English" in call.config.system_instruction
    assert logs == ["Summarizing with m1..."]


@pytest.mark.parametrize(
    "error",
    [
        server_error(503),
        server_error(500, "INTERNAL"),
        server_error(504, "DEADLINE_EXCEEDED"),
        client_error(429, "RESOURCE_EXHAUSTED"),
        client_error(404, "NOT_FOUND"),
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),
    ],
)
def test_rotates_on_retryable_errors(error):
    (summary, model), client, logs = run({"m1": error, "m2": OK})
    assert model == "m2"
    assert [c.model for c in client.models.calls] == ["m1", "m2"]
    assert logs[1].startswith("m1 unavailable (") and logs[1].endswith("trying m2...")


def test_rotation_log_shows_code_and_status():
    _, _, logs = run({"m1": server_error(503), "m2": OK})
    assert logs[1] == "m1 unavailable (503 UNAVAILABLE), trying m2..."


@pytest.mark.parametrize("reply", ["not json", json.dumps({"tldr": "only"}), None, ""])
def test_rotates_on_invalid_json(reply):
    (summary, model), _, _ = run({"m1": reply, "m2": OK})
    assert model == "m2"


@pytest.mark.parametrize(
    "error", [client_error(400, "INVALID_ARGUMENT"), client_error(401, "UNAUTHENTICATED"), client_error(403, "PERMISSION_DENIED")]
)
def test_stops_on_fatal_client_errors(error):
    with pytest.raises(FatalModelError) as info:
        run({"m1": error, "m2": OK})
    assert [c.model for c in info.value.client.models.calls] == ["m1"]
    assert str(error.code) in str(info.value)


def test_all_models_failed_lists_each():
    with pytest.raises(AllModelsFailed) as info:
        run({"m1": server_error(503), "m2": client_error(429, "RESOURCE_EXHAUSTED", "quota"), "m3": "bad"})
    message = str(info.value)
    assert [a.model for a in info.value.attempts] == ["m1", "m2", "m3"]
    assert "m1: 503 UNAVAILABLE: The model is overloaded." in message
    assert "m2: 429 RESOURCE_EXHAUSTED: quota" in message
    assert "m3: invalid JSON reply" in message


def test_single_model_no_rotation():
    with pytest.raises(AllModelsFailed):
        run({"only": server_error()})


# --------------------------------------------------------------------------- rendering


def test_render_markdown():
    md = render_markdown(
        VideoSummary(**CANNED),
        parse_header(TRANSCRIPT),
        model="gemini-3.6-flash",
        source=Path("transcriptions/rick.txt"),
        date=dt.date(2026, 10, 28),
    )
    assert md == (
        "# Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)\n"
        "\n"
        "Rick Astley · 3:33 · https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
        "\n"
        "> **TL;DR** A man promises never to give you up.\n"
        "\n"
        "## Summary\n"
        "\n"
        "Paragraph one.\n\nParagraph two.\n"
        "\n"
        "## Key points\n"
        "\n"
        "- He will not give you up.\n"
        "- He will not let you down.\n"
        "\n"
        "## Notable quotes\n"
        "\n"
        '> "Never gonna give you up"\n'
        "\n"
        "## Topics\n"
        "\n"
        "music · 1980s pop\n"
        "\n"
        "---\n"
        "Generated by gemini-3.6-flash from transcriptions/rick.txt on 2026-10-28\n"
    )


def test_render_omits_empty_quotes_and_unknown_title():
    data = {**CANNED, "notable_quotes": [], "key_points": ["only one"]}
    md = render_markdown(
        VideoSummary(**data),
        {"title": "(unknown)", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        model="m",
        source="transcriptions/dQw4w9WgXcQ.txt",
        date=dt.date(2026, 10, 28),
    )
    assert md.startswith("# dQw4w9WgXcQ\n\nhttps://www.youtube.com/watch?v=dQw4w9WgXcQ\n")
    assert "Notable quotes" not in md
    assert "- only one\n" in md


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    (tmp_path / "transcriptions").mkdir()
    path = tmp_path / "transcriptions" / "rick.txt"
    path.write_text(TRANSCRIPT, encoding="utf-8")
    return tmp_path


def no_client(key):
    raise AssertionError("client must not be created")


def test_main_missing_key_exits_2(workdir, capsys):
    assert S.main(["transcriptions/rick.txt"], client_factory=no_client) == 2
    assert MISSING_KEY_MESSAGE in capsys.readouterr().err
    assert not (workdir / "summaries").exists()


def test_main_missing_file_exits_2(workdir, capsys):
    assert S.main(["transcriptions/nope.txt"], client_factory=no_client) == 2
    assert "cannot read" in capsys.readouterr().err


def test_main_reads_key_from_dotenv_and_saves(workdir, monkeypatch, capsys):
    (workdir / ".env").write_text("GEMINI_API_KEY=from-dotenv\nGEMINI_MODELS=m1,m2\n", encoding="utf-8")
    seen = {}

    def factory(key):
        seen["key"] = key
        return FakeClient({"m1": server_error(), "m2": OK})

    assert S.main(["transcriptions/rick.txt"], client_factory=factory) == 0
    assert seen["key"] == "from-dotenv"
    captured = capsys.readouterr()
    assert "# Rick Astley - Never Gonna Give You Up" in captured.out
    assert "Generated by m2 from transcriptions/rick.txt" in captured.out
    assert "m1 unavailable (503 UNAVAILABLE), trying m2..." in captured.err
    assert "Saved summaries/rick.md" in captured.err
    saved = (workdir / "summaries" / "rick.md").read_text(encoding="utf-8")
    assert saved in captured.out


def test_main_env_wins_over_dotenv_and_flag_over_env(workdir, monkeypatch):
    (workdir / ".env").write_text("GEMINI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "exported")
    monkeypatch.setenv("GEMINI_MODELS", "env-model")
    seen = {}

    def factory(key):
        seen["key"] = key
        client = FakeClient({"flag-model": OK})
        seen["client"] = client
        return client

    assert S.main(["transcriptions/rick.txt", "--model", "flag-model"], client_factory=factory) == 0
    assert seen["key"] == "exported"
    assert [c.model for c in seen["client"].models.calls] == ["flag-model"]


def test_main_overwrites_existing_summary(workdir, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    (workdir / "summaries").mkdir()
    (workdir / "summaries" / "rick.md").write_text("old", encoding="utf-8")
    assert S.main(["transcriptions/rick.txt", "--model", "m"], client_factory=lambda k: FakeClient({"m": OK})) == 0
    assert (workdir / "summaries" / "rick.md").read_text(encoding="utf-8").startswith("# Rick Astley")


def test_main_all_failed_exits_1(workdir, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    factory = lambda k: FakeClient({"a": server_error(), "b": client_error(404, "NOT_FOUND")})
    assert S.main(["transcriptions/rick.txt", "--model", "a,b"], client_factory=factory) == 1
    err = capsys.readouterr().err
    assert "every model in the rotation failed" in err and "a: 503" in err and "b: 404" in err
    assert not (workdir / "summaries").exists()


def test_make_client_passes_key_and_disables_retries(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "stray")
    client = S.make_client("explicit-key")
    assert client._api_client.api_key == "explicit-key"
    assert client._api_client._http_options.retry_options.attempts == 1
    assert client._api_client._http_options.timeout == S.REQUEST_TIMEOUT_MS


def test_summarize_module_does_not_import_transcript():
    source = Path(S.__file__).read_text(encoding="utf-8")
    assert "transcript import" not in source and "import yt_summarize.transcript" not in source
