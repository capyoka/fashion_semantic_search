from openai import OpenAI
import os, json, re

# =========================================================
# ✳️ 外部で定義できるプロンプト群
# =========================================================

SYSTEM_PROMPT_BASE = """
You are a helpful assistant. Answer concisely in Japanese.
"""

PROMPT_REWRITE =  """
あなたはファッション商品の検索支援アシスタントです。
ユーザが入力した自然文の検索クエリを、検索システムで扱いやすい2種類の英語クエリに変換してください。

目的：
1. ベクトル検索（semantic search）に適した自然な英文フレーズ（semantic_query）
2. 単語検索（BM25）に適した主要キーワードの列（bm25_query）

ルール：
- 出力は英語のJSON形式で行う。
- semantic_query は1文または1フレーズで、自然言語に近いが簡潔な検索文にする。
- bm25_query はスペース区切りの主要キーワード群にする。
- ユーザが明示していない属性（素材、色、用途など）を推測して追加しない。
- 主観的な表現（かわいい、写真映え、人気など）は、見た目や用途に基づく中立的な表現へ置き換える。
  例：「かわいい服」→「stylish outfit」, 「写真映え」→「bright and eye-catching」
- 検索語として有効な要素（カテゴリ、用途、季節、色、素材、スタイル、オケージョンなど）を中心に構成する。
- 出力形式は必ず以下のJSON形式に従う：
{
  "semantic_query": "...",
  "bm25_query": "..."
}
"""

PROMPT_RERANK = """
以下はユーザの検索意図と、候補の商品説明リストです。
ユーザ質問: {user_query}

候補一覧（[]内はインデックス）:
{enumerated}

指示:
- ユーザが最も欲しい順に並べ替えてください。
- 出力は JSON 配列で、インデックス番号のみ（例: [2,0,1]）。
- 余計な文字は出さず、JSONだけを返す。
"""

# =========================================================
# ✳️ クラス定義
# =========================================================

class Generator:
    def __init__(self, model: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=key)
        self.model = model

    # --- 基本生成 ---
    def generate(self, query: str, context: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        r = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return r.choices[0].message.content.strip()

    # --- クエリ書き換え ---
    def rewrite_query(self, user_query: str, prompt_template: str = PROMPT_REWRITE) -> str:
        prompt = prompt_template.format(user_query=user_query)
        msgs = [
            {"role": "system", "content": "Japanese concise query rewriter."},
            {"role": "user", "content": prompt},
        ]
        r = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return r.choices[0].message.parsed

    # --- LLMリランク ---
    def rerank(
        self,
        user_query: str,
        items: list[str],
        prompt_template: str = PROMPT_RERANK,
    ) -> list[int]:
        enumerated = "\n".join([f"[{i}] {t}" for i, t in enumerate(items)])
        prompt = prompt_template.format(user_query=user_query, enumerated=enumerated)
        msgs = [
            {"role": "system", "content": "You are a ranking model. Return only a JSON array of integers."},
            {"role": "user", "content": prompt},
        ]
        r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        text = r.choices[0].message.content.strip()

        # JSONパース（壊れていたらフォールバック）
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and all(isinstance(x, int) for x in arr):
                return arr
        except Exception:
            pass
        return [int(m) for m in re.findall(r"\d+", text)]
