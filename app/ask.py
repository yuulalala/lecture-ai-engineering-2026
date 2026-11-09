"""第1回: LLMに質問する。--chat で会話履歴を保って続けられる。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat "LoRAとは何ですか？"

通常は履歴なしで1回だけ質問する。--chat の履歴は終了時に破棄する。
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


def print_usage(response: types.GenerateContentResponse) -> None:
    """未取得の値と実際の0トークンを区別して表示する。"""
    usage = response.usage_metadata
    incoming = usage.prompt_token_count if usage else None
    outgoing = usage.candidates_token_count if usage else None
    print(f"[input tokens] {incoming if incoming is not None else '不明'}")
    print(f"[output tokens] {outgoing if outgoing is not None else '不明'}")


def ask(question: str) -> str:
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
        print_usage(response)
        return response.text or "（テキスト応答がありません）"


def chat(question: str) -> None:
    """同じSDKチャットを再利用し、質問・回答の履歴を次の送信に含める。"""
    with genai.Client(api_key=get_api_key()) as client:
        session = client.chats.create(model=MODEL)
        print("続けて質問できます。/exit または /quit で終了します。")
        while True:
            response = session.send_message(question)
            print_usage(response)
            print(response.text or "（テキスト応答がありません）")
            while True:
                try:
                    question = input("\nあなた> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n会話を終了しました。")
                    return
                if question.lower() in {"/exit", "/quit"}:
                    print("会話を終了しました。")
                    return
                if question:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        if args.chat:
            chat(args.question)
        else:
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n中断しました。")


if __name__ == "__main__":
    main()
