"""第1回: 素のLLMへの単発質問と、履歴を引き継ぐ対話。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 続けて質問する。/quit で終了、/reset で履歴消去

検索ツールや講座資料を持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


class EmptyResponseError(ValueError):
    """安全性によるブロックなどで、回答本文が得られなかった場合。"""


def print_usage(response: types.GenerateContentResponse) -> None:
    """API が返したトークン数を表示する。未取得を 0 と取り違えない。"""
    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    prompt_label = prompt_tokens if prompt_tokens is not None else "取得不可"
    output_label = output_tokens if output_tokens is not None else "取得不可"
    print(f"[tokens] 入力: {prompt_label} / 出力: {output_label}")


def response_text(response: types.GenerateContentResponse) -> str:
    """本文がない応答は成功扱いにせず、会話履歴にも追加しない。"""
    text = response.text
    if text is None or not text.strip():
        raise EmptyResponseError(
            "回答本文を取得できませんでした。安全性設定や質問内容を確認してください。"
        )
    return text


def ask(question: str) -> str:
    """既存の単発質問用インターフェースを維持する。"""
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response)
    return response_text(response)


class Conversation:
    """成功した user / model の組だけを、次のリクエストに渡す。"""

    def __init__(self, client: genai.Client, model: str = MODEL) -> None:
        self.client = client
        self.model = model
        self._history: list[types.Content] = []

    def reset(self) -> None:
        self._history.clear()

    def ask(self, question: str) -> str:
        # 失敗した質問を履歴に残さないよう、送信用のリストを別に作る。
        contents = self._history + [
            types.Content(role="user", parts=[types.Part(text=question)])
        ]
        response = self.client.models.generate_content(
            model=self.model, contents=contents
        )
        print_usage(response)
        answer = response_text(response)
        self._history = contents + [
            types.Content(role="model", parts=[types.Part(text=answer)])
        ]
        return answer


def run_chat(initial_question: str | None = None) -> None:
    """対話の履歴はこのプロセスのメモリだけに保持する。"""
    print("[対話] /quit または /exit で終了、/reset で履歴を消去します。")
    with genai.Client(api_key=get_api_key()) as client:
        conversation = Conversation(client)
        question = initial_question
        while True:
            if question is None:
                try:
                    question = input("あなた> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            question = question.strip()
            if question in {"/quit", "/exit"}:
                break
            if question == "/reset":
                conversation.reset()
                print("[対話] 履歴を消去しました。")
            elif question:
                try:
                    print(conversation.ask(question))
                except errors.APIError as error:
                    # 自動再送はせず、利用者が次の入力を判断する。
                    print(api_error_message(error.code, error.message))
                except EmptyResponseError as error:
                    print(str(error))
            question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMへの単発質問・履歴付き対話")
    parser.add_argument("question", nargs="?", help="質問（省略時は講座固有の質問）")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        run_chat(args.question)
        return

    question = args.question if args.question is not None else DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except EmptyResponseError as e:
        print(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
