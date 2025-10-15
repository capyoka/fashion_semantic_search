from typing import List
import os, json, re
from openai import AsyncOpenAI

# =========================================================
# ✳️ 外部で定義できるプロンプト群
# =========================================================

SYSTEM_PROMPT_BASE = "You are a helpful assistant. Answer concisely in Japanese."

PROMPT_REWRITE = """以下のユーザ質問を検索に適した短いクエリへ書き換えてください。
- ノイズ除去・具体属性は保持
- 日本語・30字以内・1行のみ
質問: {user_query}"""

PROMPT_KEYWORDS = """以下のユーザ質問からBM25向け重要語を抽出。
- 日本語
- {k_min}〜{k_max}語
- 出力はカンマ区切りのみ（例: 速乾, 夏, ワンピース）
質問: {user_query}"""

PROMPT_RERANK = """ユーザ質問: {user_query}
候補（[]はインデックス）:
{enumerated}

指示:
- ユーザが欲しい順に並べ替え。
- 出力は整数インデックスのJSON配列のみ（例: [2,0,1]）。他の文字は禁止。"""

# =========================================================
# ✳️ クラス定義
# =========================================================

class AsyncGenerator:
    def __init__(self, model: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = AsyncOpenAI(api_key=key)
        self.model = model

    # --- クエリ書き換え ---
    async def rewrite_query(self, user_query: str, prompt_template: str = PROMPT_REWRITE) -> str:
        prompt = prompt_template.format(user_query=user_query)
        msgs = [
            {"role": "system", "content": "Japanese concise query rewriter."},
            {"role": "user", "content": prompt},
        ]
        r = await self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        return r.choices[0].message.content.strip()

    # --- BM25用キーワード抽出 ---
    async def extract_bm25_keywords(
        self,
        user_query: str,
        k_min: int = 3,
        k_max: int = 10,
        prompt_template: str = PROMPT_KEYWORDS,
    ) -> list[str]:
        prompt = prompt_template.format(user_query=user_query, k_min=k_min, k_max=k_max)
        msgs = [
            {"role": "system", "content": "Japanese keyword extractor."},
            {"role": "user", "content": prompt},
        ]
        r = await self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        line = r.choices[0].message.content.strip()
        return [w.strip() for w in re.split(r"[,\u3001]", line) if w.strip()][:k_max]

    # --- LLMリランク ---
    async def rerank(
        self,
        user_query: str,
        items: List[str],
        prompt_template: str = PROMPT_RERANK,
    ) -> list[int]:
        enumerated = "\n".join([f"[{i}] {t}" for i, t in enumerate(items)])
        prompt = prompt_template.format(user_query=user_query, enumerated=enumerated)
        msgs = [
            {"role": "system", "content": "You are a ranking model. Return only a JSON array of integers."},
            {"role": "user", "content": prompt},
        ]
        r = await self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        text = r.choices[0].message.content.strip()

        # JSONパース（壊れていたらフォールバック）
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and all(isinstance(x, int) for x in arr):
                return arr
        except Exception:
            pass

        # 壊れていたらデフォルト順
        return list(range(len(items)))
