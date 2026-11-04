"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
from datetime import date, datetime
from functools import cache
from pathlib import Path
import random

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3
LEARNING_LOG_DIR = DATA_DIR.parent / "learning_log"


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回の用語から1つ選び、学習後の確認問題を作る。

    「第2回の内容から確認問題を出して」のように、その回の用語を
    復習したいときに使う。問題と答えをJSON形式で返す。
    """
    terms = [term for term in load_glossary() if term["lecture"] == lecture]
    if not terms:
        return f"第{lecture}回の用語が用語集に見つかりませんでした。"
    term = random.choice(terms)
    return json.dumps(
        {"question": f"「{term['term']}」の意味は？", "answer": term["definition"]},
        ensure_ascii=False,
    )


def save_learning_log(topic: str, learning: str) -> str:
    """学習内容を日付付きMarkdownに追記する（呼び出し前に利用者承認が必要）。

    利用者が「今日の学習記録を保存して」など、明示的に記録を求めたときに使う。
    この関数は学習記録をローカルの learning_log/ に追記するため、エージェント側で
    `requires_approval=True` を指定し、保存前に利用者へ確認すること。
    """
    clean_topic = " ".join(topic.split())
    clean_learning = learning.strip()
    if not clean_topic:
        return "保存できませんでした。トピックが空です。"
    if not clean_learning:
        return "保存できませんでした。学んだ内容が空です。"
    if len(clean_topic) > 200 or len(clean_learning) > 4000:
        return "保存できませんでした。トピックは200文字、学習内容は4000文字以内にしてください。"

    today = date.today().isoformat()
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S %Z")
    path = LEARNING_LOG_DIR / f"{today}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(f"\n## {timestamp} — {clean_topic}\n\n{clean_learning}\n")
    return f"学習記録を learning_log/{today}.md に追記しました。"
