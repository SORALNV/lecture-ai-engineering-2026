"""第1回の履歴・トークン表示・CLIを、APIキーも通信も使わずに確認する。"""

from unittest.mock import MagicMock

import pytest
from google.genai import errors, types

from app import ask as ask_module


def make_response(text="応答", prompt_tokens=12, output_tokens=4):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)])
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
        ),
    )


def sent_messages(call):
    return [
        (content.role, "".join(part.text or "" for part in content.parts))
        for content in call.kwargs["contents"]
    ]


@pytest.fixture
def client_factory(monkeypatch):
    client = MagicMock()
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    # 本物のAPIキーの環境変数を読む関数は呼ばない。
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-only-not-a-real-key")
    monkeypatch.setattr(ask_module.genai, "Client", factory)
    return factory, client


def test_single_question_keeps_existing_call_shape_and_closes_client(
    client_factory, capsys
):
    factory, client = client_factory
    client.models.generate_content.return_value = make_response("LoRAの説明")

    assert ask_module.ask("LoRAとは？") == "LoRAの説明"

    client.models.generate_content.assert_called_once_with(
        model=ask_module.MODEL, contents="LoRAとは？"
    )
    assert factory.return_value.__exit__.call_count == 1
    assert "入力: 12 / 出力: 4" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, "入力: 取得不可 / 出力: 取得不可"),
        (
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=0, candidates_token_count=0
            ),
            "入力: 0 / 出力: 0",
        ),
        (
            types.GenerateContentResponseUsageMetadata(prompt_token_count=9),
            "入力: 9 / 出力: 取得不可",
        ),
    ],
)
def test_token_display_distinguishes_missing_metadata_from_zero(usage, expected, capsys):
    response = types.GenerateContentResponse(usage_metadata=usage)
    ask_module.print_usage(response)
    assert expected in capsys.readouterr().out


def test_second_question_receives_user_and_model_history():
    client = MagicMock()
    client.models.generate_content.side_effect = [
        make_response("最初の回答"),
        make_response("続きの回答"),
    ]
    conversation = ask_module.Conversation(client, model="test-model")

    assert conversation.ask("最初の質問") == "最初の回答"
    assert conversation.ask("それを詳しく") == "続きの回答"

    first, second = client.models.generate_content.call_args_list
    assert first.kwargs["model"] == second.kwargs["model"] == "test-model"
    assert sent_messages(first) == [("user", "最初の質問")]
    assert sent_messages(second) == [
        ("user", "最初の質問"),
        ("model", "最初の回答"),
        ("user", "それを詳しく"),
    ]


def test_reset_removes_history_from_the_next_request():
    client = MagicMock()
    client.models.generate_content.return_value = make_response()
    conversation = ask_module.Conversation(client)
    conversation.ask("前の話題")
    conversation.reset()
    conversation.ask("新しい話題")

    assert sent_messages(client.models.generate_content.call_args) == [
        ("user", "新しい話題")
    ]


def test_api_failure_does_not_append_failed_question_to_history():
    client = MagicMock()
    failure = errors.APIError(429, {"error": {"message": "rate limited"}})
    client.models.generate_content.side_effect = [
        make_response("成功した回答"),
        failure,
        make_response("次の回答"),
    ]
    conversation = ask_module.Conversation(client)
    conversation.ask("成功した質問")
    with pytest.raises(errors.APIError):
        conversation.ask("失敗した質問")
    conversation.ask("別の質問")

    assert sent_messages(client.models.generate_content.call_args) == [
        ("user", "成功した質問"),
        ("model", "成功した回答"),
        ("user", "別の質問"),
    ]


@pytest.mark.parametrize("empty_text", ["", "   "])
def test_empty_response_does_not_become_successful_history(empty_text):
    client = MagicMock()
    client.models.generate_content.side_effect = [
        make_response(empty_text),
        make_response("回答"),
    ]
    conversation = ask_module.Conversation(client)
    with pytest.raises(ask_module.EmptyResponseError):
        conversation.ask("回答が空の質問")
    conversation.ask("別の質問")

    assert sent_messages(client.models.generate_content.call_args) == [
        ("user", "別の質問")
    ]


def test_blocked_response_without_candidates_is_reported():
    with pytest.raises(ask_module.EmptyResponseError, match="回答本文"):
        ask_module.response_text(types.GenerateContentResponse())


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [([], ask_module.DEFAULT_QUESTION), (["単発の質問"], "単発の質問")],
)
def test_cli_single_question_with_or_without_positional_argument(
    client_factory, monkeypatch, capsys, arguments, expected
):
    _, client = client_factory
    client.models.generate_content.return_value = make_response("単発の回答")
    monkeypatch.setattr("sys.argv", ["app.ask", *arguments])

    ask_module.main()

    assert client.models.generate_content.call_args.kwargs["contents"] == expected
    assert "単発の回答" in capsys.readouterr().out


def test_cli_chat_preserves_history_and_skips_blank_input(
    client_factory, monkeypatch, capsys
):
    _, client = client_factory
    client.models.generate_content.side_effect = [
        make_response("回答1"),
        make_response("回答2"),
    ]
    inputs = iter(["", "質問1", "続けて質問", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat"])

    ask_module.main()

    assert client.models.generate_content.call_count == 2
    assert sent_messages(client.models.generate_content.call_args) == [
        ("user", "質問1"),
        ("model", "回答1"),
        ("user", "続けて質問"),
    ]
    assert "回答2" in capsys.readouterr().out


def test_cli_chat_supports_initial_question_and_reset(client_factory, monkeypatch):
    _, client = client_factory
    client.models.generate_content.return_value = make_response()
    inputs = iter(["/reset", "新しい質問", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat", "初回の質問"])

    ask_module.main()

    assert client.models.generate_content.call_count == 2
    assert sent_messages(client.models.generate_content.call_args) == [
        ("user", "新しい質問")
    ]


def test_cli_chat_can_continue_after_rate_limit(client_factory, monkeypatch, capsys):
    _, client = client_factory
    client.models.generate_content.side_effect = [
        errors.APIError(429, {"error": {"message": "rate limited"}}),
        make_response("再入力後の回答"),
    ]
    inputs = iter(["質問1", "質問2", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat"])

    ask_module.main()

    assert sent_messages(client.models.generate_content.call_args) == [("user", "質問2")]
    output = capsys.readouterr().out
    assert "レート制限" in output
    assert "再入力後の回答" in output


@pytest.mark.parametrize("end_input", [EOFError, KeyboardInterrupt])
def test_cli_chat_closes_client_on_eof_or_keyboard_interrupt(
    client_factory, monkeypatch, end_input
):
    factory, client = client_factory

    def stop_input(_):
        raise end_input

    monkeypatch.setattr("builtins.input", stop_input)
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat"])

    ask_module.main()

    client.models.generate_content.assert_not_called()
    assert factory.return_value.__exit__.call_count == 1


@pytest.mark.parametrize(
    "response_or_error",
    [
        make_response(""),
        errors.APIError(503, {"error": {"message": "unavailable"}}),
    ],
)
def test_cli_single_failure_returns_nonzero_exit_code(
    client_factory, monkeypatch, response_or_error
):
    _, client = client_factory
    if isinstance(response_or_error, Exception):
        client.models.generate_content.side_effect = response_or_error
    else:
        client.models.generate_content.return_value = response_or_error
    monkeypatch.setattr("sys.argv", ["app.ask", "質問"])

    with pytest.raises(SystemExit) as exit_info:
        ask_module.main()
    assert exit_info.value.code == 1
