from openai import OpenAI
import os, json, re

SYSTEM = "You are a helpful assistant. Answer concisely in Japanese."

class Generator:
    def __init__(self, model: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def generate(self, query: str, context: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.2)
        return r.choices[0].message.content.strip()

    # --- 追加: クエリ書き換え ---
    def rewrite_query(self, user_query: str) -> str:
        prompt = f"""以下のユーザ質問を、検索に適したクエリに短く書き換えてください。
- ノイズを除去
- 具体的な属性（用途・季節・素材・価格帯等）があれば保持
- 日本語で、30字以内、改行なし、1行で出力
質問: {user_query}"""
        msgs = [{"role":"system","content":"Japanese concise query rewriter."},
                {"role":"user","content":prompt}]
        r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        return r.choices[0].message.content.strip()

    # --- 追加: BM25用キーワード抽出 ---
    def extract_bm25_keywords(self, user_query: str, k_min:int=3, k_max:int=10) -> list[str]:
        prompt = f"""以下のユーザ質問から、BM25向けの重要キーワードを抽出してください。
- 日本語
- {k_min}〜{k_max}語
- 出力は「カンマ区切り」のみ（例: 速乾, 夏, ワンピース）
質問: {user_query}"""
        msgs = [{"role":"system","content":"Japanese keyword extractor."},
                {"role":"user","content":prompt}]
        r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        line = r.choices[0].message.content.strip()
        # 逗点/カンマ両対応・空白除去
        parts = [w.strip() for w in re.split(r"[,\u3001]", line) if w.strip()]
        return parts[:k_max]

    # --- 追加: LLMリランク（indexの配列をJSONで返す） ---
    def rerank(self, user_query: str, items: list[str]) -> list[int]:
        """
        items: 検索結果のテキスト配列。呼び出し側で enumerate して番号→本文の形に整形して渡すと精度安定。
        返り値: 並べ替えた index のリスト
        """
        # インデックス付きの箇条書きにしてプロンプト化
        enumerated = "\n".join([f"[{i}] {t}" for i, t in enumerate(items)])
        prompt = f"""以下はユーザの検索意図と、候補の商品説明リストです。
ユーザ質問: {user_query}

候補一覧（[]内はインデックス）:
{enumerated}

指示:
- ユーザが最も欲しい順に並べ替えてください。
- 出力は JSON 配列で、インデックス番号のみ（例: [2,0,1]）。
- 余計な文字は出さず、JSONだけを返す。
"""
        msgs = [{"role":"system","content":"You are a ranking model. Return only a JSON array of integers."},
                {"role":"user","content":prompt}]
        r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.0)
        text = r.choices[0].message.content.strip()
        # パース（壊れていたら数字だけ拾うフォールバック）
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and all(isinstance(x,int) for x in arr):
                return arr
        except Exception:
            pass
        nums = [int(m) for m in re.findall(r"\d+", text)]
        return nums
