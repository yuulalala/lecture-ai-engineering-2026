from app.ask import ask


class _DummyUsageMetadata:
    prompt_token_count = 123
    candidates_token_count = 45


class _DummyResponse:
    text = "AIの返答"
    usage_metadata = _DummyUsageMetadata()


class _DummyModels:
    @staticmethod
    def generate_content(*args, **kwargs):
        return _DummyResponse()


class _DummyClient:
    models = _DummyModels()


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    monkeypatch.setattr("app.ask.genai.Client", lambda api_key: _DummyClient())
    monkeypatch.setattr("app.ask.get_api_key", lambda: "test-key")

    result = ask("LoRAとは何ですか？")

    captured = capsys.readouterr()
    assert result == "AIの返答"
    assert "[input tokens] 123" in captured.out
    assert "[output tokens] 45" in captured.out
