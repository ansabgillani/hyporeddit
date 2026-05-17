"""Unit tests for translation/translator.py.

All tests inject _client directly so no real network call is ever made.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _make_anthropic_mock(response_text: str = "We waited 14 months for the building permit.") -> MagicMock:
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_message
    return mock_client


def _make_openai_mock(response_text: str = "We waited 14 months for the building permit.") -> MagicMock:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    return mock_client


@pytest.fixture
def mock_anthropic(monkeypatch):
    import hyporeddit.translation.translator as _t_mod
    monkeypatch.setattr(_t_mod.settings, "llm_provider", "anthropic")
    return _make_anthropic_mock()


# ---------------------------------------------------------------------------
# Single-item translation — Anthropic provider
# ---------------------------------------------------------------------------

class TestTranslatorBasic:
    def test_translate_returns_english_text(self, mock_anthropic):
        from hyporeddit.translation.translator import translate_de_to_en

        result = translate_de_to_en("Wir haben 14 Monate auf den Bauantrag gewartet.", _client=mock_anthropic)
        assert result == "We waited 14 months for the building permit."

    def test_translate_calls_messages_create(self, mock_anthropic):
        from hyporeddit.translation.translator import translate_de_to_en

        translate_de_to_en("Irgendein Text", _client=mock_anthropic)
        mock_anthropic.messages.create.assert_called_once()

    def test_translate_uses_haiku_model(self, mock_anthropic):
        from hyporeddit.translation.translator import translate_de_to_en

        translate_de_to_en("Text", _client=mock_anthropic)
        call_kwargs = mock_anthropic.messages.create.call_args
        assert "haiku" in call_kwargs.kwargs.get("model", "").lower()

    def test_translate_includes_source_text_in_prompt(self, mock_anthropic):
        from hyporeddit.translation.translator import translate_de_to_en

        german = "Wir haben lange gewartet."
        translate_de_to_en(german, _client=mock_anthropic)
        call_kwargs = mock_anthropic.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages", [])
        prompt_text = " ".join(
            m["content"] for m in messages if isinstance(m.get("content"), str)
        )
        assert german in prompt_text

    def test_translate_uses_prompt_file(self, mock_anthropic):
        from hyporeddit.translation.translator import translate_de_to_en

        translate_de_to_en("Text", _client=mock_anthropic)
        call_kwargs = mock_anthropic.messages.create.call_args
        system_prompt = call_kwargs.kwargs.get("system", "")
        prompt_path = Path("prompts/translation_v1.txt")
        if prompt_path.exists():
            expected = prompt_path.read_text().strip()
            assert expected in system_prompt


class TestTranslatorRetry:
    def test_translate_retries_on_api_error(self, monkeypatch):
        import anthropic as _anthropic
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "anthropic")

        call_count = 0
        mock_client = MagicMock()

        def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _anthropic.APIError(
                    message="server error", request=MagicMock(), body=None
                )
            msg = MagicMock()
            msg.content = [MagicMock(text="Success")]
            return msg

        mock_client.messages.create.side_effect = flaky_create

        from hyporeddit.translation.translator import translate_de_to_en

        result = translate_de_to_en("Text", _client=mock_client)

        assert result == "Success"
        assert call_count == 3

    def test_translate_raises_after_max_retries(self, monkeypatch):
        import anthropic as _anthropic
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "anthropic")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _anthropic.APIError(
            message="always fails", request=MagicMock(), body=None
        )

        from hyporeddit.translation.translator import translate_de_to_en

        with pytest.raises(Exception):
            translate_de_to_en("Text", _client=mock_client)

        assert mock_client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# Single-item translation — OpenAI provider
# ---------------------------------------------------------------------------

class TestTranslatorOpenAI:
    def test_translate_uses_chat_completions_api_when_provider_is_openai(self, monkeypatch):
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "openai")

        mock_client = _make_openai_mock("Building permit approved.")
        from hyporeddit.translation.translator import translate_de_to_en

        result = translate_de_to_en("Baugenehmigung erteilt.", _client=mock_client)

        mock_client.chat.completions.create.assert_called_once()
        assert result == "Building permit approved."

    def test_translate_does_not_call_anthropic_messages_when_provider_is_openai(self, monkeypatch):
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "openai")

        mock_client = _make_openai_mock()
        from hyporeddit.translation.translator import translate_de_to_en

        translate_de_to_en("Text", _client=mock_client)

        mock_client.messages.create.assert_not_called()

    def test_translate_sends_source_text_in_openai_user_message(self, monkeypatch):
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "openai")

        mock_client = _make_openai_mock()
        from hyporeddit.translation.translator import translate_de_to_en

        german = "Wir haben lange gewartet."
        translate_de_to_en(german, _client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert german in user_content

    def test_translate_uses_openai_classification_model(self, monkeypatch):
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "openai")

        mock_client = _make_openai_mock()
        from hyporeddit.translation.translator import translate_de_to_en

        translate_de_to_en("Text", _client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == _t_mod.settings.openai_classification_model


class TestTranslatorOpenAIRetry:
    def test_translate_retries_on_openai_api_error(self, monkeypatch):
        import openai as _openai
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "openai")

        call_count = 0
        mock_client = MagicMock()

        def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _openai.APIError(
                    message="server error", request=MagicMock(), body=None
                )
            mock_choice = MagicMock()
            mock_choice.message.content = "Success"
            return MagicMock(choices=[mock_choice])

        mock_client.chat.completions.create.side_effect = flaky_create

        from hyporeddit.translation.translator import translate_de_to_en

        result = translate_de_to_en("Text", _client=mock_client)

        assert result == "Success"
        assert call_count == 3

    def test_translate_raises_after_max_openai_retries(self, monkeypatch):
        import openai as _openai
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "openai")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _openai.APIError(
            message="always fails", request=MagicMock(), body=None
        )

        from hyporeddit.translation.translator import translate_de_to_en

        with pytest.raises(Exception):
            translate_de_to_en("Text", _client=mock_client)

        assert mock_client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# Batch translation — Anthropic provider
# ---------------------------------------------------------------------------

def _make_batch_mock(response_json: str) -> MagicMock:
    mock_client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_json)]
    mock_client.messages.create.return_value = msg
    return mock_client


class TestTranslateBatch:
    @pytest.fixture(autouse=True)
    def pin_anthropic(self, monkeypatch):
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "anthropic")

    def test_batch_returns_one_translation_per_input(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        # _call_batch_anthropic prepends "[" to the response (assistant prefill),
        # so the mock must return the continuation after "[".
        client = _make_batch_mock('"English A", "English B", "English C"]')
        results = translate_batch_de_to_en(["Text A", "Text B", "Text C"], _client=client)

        assert results == ["English A", "English B", "English C"]

    def test_batch_makes_one_api_call_for_three_texts(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        client = _make_batch_mock('"T1", "T2", "T3"]')
        translate_batch_de_to_en(["A", "B", "C"], _client=client)

        assert client.messages.create.call_count == 1

    def test_batch_sends_all_texts_in_prompt(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        client = _make_batch_mock('"T1", "T2"]')
        translate_batch_de_to_en(["Hallo", "Welt"], _client=client)

        call_kwargs = client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "Hallo" in user_content
        assert "Welt" in user_content

    def test_batch_empty_list_returns_empty(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        client = MagicMock()
        results = translate_batch_de_to_en([], _client=client)

        assert results == []
        client.messages.create.assert_not_called()

    def test_batch_falls_back_to_per_item_on_bad_json(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            msg = MagicMock()
            if call_count == 1:
                msg.content = [MagicMock(text="not json at all")]
            else:
                msg.content = [MagicMock(text=f"Item {call_count - 1}")]
            return msg

        client = MagicMock()
        client.messages.create.side_effect = side_effect

        results = translate_batch_de_to_en(["A", "B"], _client=client)

        assert len(results) == 2
        assert client.messages.create.call_count == 3  # 1 batch attempt + 2 per-item fallbacks

    def test_batch_falls_back_on_wrong_item_count(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            msg = MagicMock()
            if call_count == 1:
                msg.content = [MagicMock(text='["only one"]')]
            else:
                msg.content = [MagicMock(text=f"fallback {call_count}")]
            return msg

        client = MagicMock()
        client.messages.create.side_effect = side_effect

        results = translate_batch_de_to_en(["A", "B"], _client=client)

        assert len(results) == 2
        assert client.messages.create.call_count == 3  # 1 batch + 2 per-item

    def test_batch_splits_into_sub_batches_at_size_limit(self, monkeypatch):
        from hyporeddit.translation.translator import translate_batch_de_to_en
        import hyporeddit.translation.translator as _t_mod

        monkeypatch.setattr(_t_mod.settings, "translation_batch_size", 2)

        # Responses are continuations after the "[" assistant prefill
        responses = ['"T1", "T2"]', '"T3"]']
        call_idx = 0

        def side_effect(**kwargs):
            nonlocal call_idx
            msg = MagicMock()
            msg.content = [MagicMock(text=responses[call_idx])]
            call_idx += 1
            return msg

        client = MagicMock()
        client.messages.create.side_effect = side_effect

        results = translate_batch_de_to_en(["A", "B", "C"], _client=client)

        assert results == ["T1", "T2", "T3"]
        assert client.messages.create.call_count == 2

    def test_batch_parses_valid_json_continuation(self):
        """Batch translation correctly parses a valid JSON continuation after the '[' prefill."""
        from hyporeddit.translation.translator import translate_batch_de_to_en

        # The implementation sends assistant prefill "[" and prepends "[" to response,
        # so provide the continuation without the leading "["
        client = _make_batch_mock('"Good"]')
        results = translate_batch_de_to_en(["Text"], _client=client)

        assert results == ["Good"]


# ---------------------------------------------------------------------------
# Batch translation — OpenAI provider
# ---------------------------------------------------------------------------

class TestTranslateBatchOpenAI:
    @pytest.fixture(autouse=True)
    def pin_openai(self, monkeypatch):
        import hyporeddit.translation.translator as _t_mod
        monkeypatch.setattr(_t_mod.settings, "llm_provider", "openai")

    def test_batch_uses_chat_completions_when_provider_is_openai(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        client = _make_openai_mock('["T1", "T2"]')
        translate_batch_de_to_en(["A", "B"], _client=client)

        client.chat.completions.create.assert_called_once()
        client.messages.create.assert_not_called()

    def test_batch_returns_translations_from_openai(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        client = _make_openai_mock('["English A", "English B"]')
        results = translate_batch_de_to_en(["Text A", "Text B"], _client=client)

        assert results == ["English A", "English B"]

    def test_batch_openai_sends_all_texts_in_user_message(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        client = _make_openai_mock('["T1", "T2"]')
        translate_batch_de_to_en(["Hallo", "Welt"], _client=client)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "Hallo" in user_content
        assert "Welt" in user_content

    def test_batch_openai_falls_back_per_item_on_bad_json(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_choice = MagicMock()
            if call_count == 1:
                mock_choice.message.content = "not json at all"
            else:
                mock_choice.message.content = f"Item {call_count - 1}"
            return MagicMock(choices=[mock_choice])

        client = MagicMock()
        client.chat.completions.create.side_effect = side_effect

        results = translate_batch_de_to_en(["A", "B"], _client=client)

        assert len(results) == 2
        assert client.chat.completions.create.call_count == 3  # 1 batch + 2 per-item fallbacks

    def test_batch_openai_strips_markdown_fences(self):
        from hyporeddit.translation.translator import translate_batch_de_to_en

        client = _make_openai_mock('```json\n["Good"]\n```')
        results = translate_batch_de_to_en(["Text"], _client=client)

        assert results == ["Good"]
