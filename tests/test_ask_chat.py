"""第1回: 実際のSDKで履歴を組み立て、API送信部分だけを置き換える。"""
from copy import deepcopy

import pytest
from google.genai import errors, models, types

from app import ask


def response(text="覚えました。", usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text=text)]), finish_reason="STOP")],
        usage_metadata=(types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=4) if usage else None),
    )


@pytest.fixture
def requests(monkeypatch):
    sent = []
    monkeypatch.setattr(ask, "get_api_key", lambda: "offline-test-placeholder")

    def generate(self, *, model, contents, **kwargs):
        sent.append(deepcopy(contents))
        return response()

    monkeypatch.setattr(models.Models, "generate_content", generate)
    return sent


def test_chat_sends_user_and_model_history(monkeypatch, requests, capsys):
    inputs = iter(["", "   ", "合言葉は何ですか？", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    ask.chat("合言葉は青いりんごです。")
    assert len(requests) == 2
    assert [c.role for c in requests[1]] == ["user", "model", "user"]
    assert [c.parts[0].text for c in requests[1]] == [
        "合言葉は青いりんごです。", "覚えました。", "合言葉は何ですか？"]
    output = capsys.readouterr().out
    assert "[input tokens] 12" in output
    assert "[output tokens] 4" in output


@pytest.mark.parametrize("ending", ["/exit", "/quit", EOFError, KeyboardInterrupt])
def test_chat_exit(monkeypatch, requests, ending):
    def read(_):
        if isinstance(ending, type):
            raise ending()
        return ending
    monkeypatch.setattr("builtins.input", read)
    ask.chat("こんにちは")
    assert len(requests) == 1


def test_single_question_stays_single(requests):
    assert ask.ask("LoRAとは？") == "覚えました。"
    assert requests == ["LoRAとは？"]


@pytest.mark.parametrize("usage, expected", [
    (None, "[input tokens] 不明\n[output tokens] 不明"),
    (types.GenerateContentResponseUsageMetadata(prompt_token_count=0), "[input tokens] 0\n[output tokens] 不明"),
])
def test_missing_usage(usage, expected, capsys):
    ask.print_usage(types.GenerateContentResponse(usage_metadata=usage))
    assert expected in capsys.readouterr().out


def test_api_error_is_reported(monkeypatch, requests, capsys):
    def fail(*args, **kwargs):
        raise errors.ClientError(429, {"error": {"message": "test rate limit"}})
    monkeypatch.setattr(models.Models, "generate_content", fail)
    monkeypatch.setattr("sys.argv", ["ask", "--chat", "こんにちは"])
    with pytest.raises(SystemExit) as exc:
        ask.main()
    assert exc.value.code == 1
    assert "レート制限" in capsys.readouterr().out
