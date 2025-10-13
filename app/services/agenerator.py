from typing import List
import os, json, re
from openai import AsyncOpenAI

SYSTEM = "You are a helpful assistant. Answer concisely in Japanese."

class AsyncGenerator:
    def __init__(self, model: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = AsyncOpenAI(api_key=key)
        self.model = model

    async def rewrite_query(self, user_query: str) -> str:
        prompt = f"""以下のユーザ質問を検索に適した短いクエリへ書き換えてください。
- ノイズ除去・具体属性は保持
- 日本語・30字以内・1行のみ
質問: {user_query}"""
        msgs = [
            {"role": "system", "content": "Japanese concise query rewriter."},
            {"role": "user", "content": prompt},
        ]
        r = await self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        return r.choices[0].message.content.strip()

    async def extract_bm25_keywords(self, user_query: str, k_min: int = 3, k_max: int = 10) -> list[str]:
        prompt = f"""以下のユーザ質問からBM25向け重要語を抽出。
- 日本語
- {k_min}〜{k_max}語
- 出力はカンマ区切りのみ（例: 速乾, 夏, ワンピース）
質問: {user_query}"""
        msgs = [
            {"role": "system", "content": "Japanese keyword extractor."},
            {"role": "user", "content": prompt},
        ]
        r = await self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        line = r.choices[0].message.content.strip()
        return [w.strip() for w in re.split(r"[,\u3001]", line) if w.strip()][:k_max]

    async def rerank(self, user_query: str, items: List[str]) -> list[int]:
        enumerated = "\n".join([f"[{i}] {t}" for i, t in enumerate(items)])
        prompt = f"""ユーザ質問: {user_query}
候補（[]はインデックス）:
{enumerated}

指示:
- ユーザが欲しい順に並べ替え。
- 出力は整数インデックスのJSON配列のみ（例: [2,0,1]）。他の文字は禁止。"""
        msgs = [
            {"role": "system", "content": "You are a ranking model. Return only a JSON array of integers."},
            {"role": "user", "content": prompt},
        ]
        r = await self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        text = r.choices[0].message.content.strip()
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and all(isinstance(x, int) for x in arr):
                return arr
        except Exception:
            pass
        # 壊れていたらデフォルト順
        return list(range(len(items)))
