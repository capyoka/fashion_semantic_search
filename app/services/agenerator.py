import logging
from typing import List
import os, json, re, ast
from openai import AsyncOpenAI

# Logger設定
logger = logging.getLogger(__name__)

# =========================================================
# ✳️ 外部で定義できるプロンプト群
# =========================================================

SYSTEM_PROMPT_BASE = "You are a helpful assistant. Answer concisely in Japanese."

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
- 出力形式は必ず以下の要素を持つJSON形式に従う：

  semantic_query: ...,
  bm25_query: ...


ユーザクエリ: {user_query}
出力:
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

class AsyncGenerator:
    def __init__(self, model: str, fast_model: str = None):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = AsyncOpenAI(api_key=key)
        self.model = model
        self.fast_model = fast_model

    # --- クエリ書き換え ---
    async def rewrite_query(self, user_query: str, prompt_template: str = PROMPT_REWRITE) -> str:
        logger.info(f"🔄 Starting query rewrite for: '{user_query}'")
        logger.debug(f"Using model: {self.fast_model if self.fast_model else self.model}")
        
        prompt = prompt_template.format(user_query=user_query)
        msgs = [
            {"role": "system", "content": "Japanese concise query rewriter."},
            {"role": "user", "content": prompt},
        ]
        
        try:
            r = await self.client.chat.completions.create(
                model=self.fast_model if self.fast_model else self.model,
                messages=msgs,
                response_format={"type": "json_object"},
            )
            res = r.choices[0].message.content
            logger.debug(f"Raw LLM response: {res}")
            
            result = ast.literal_eval(res)
            semantic_query = result["semantic_query"]
            bm25_query = result["bm25_query"]
            
            logger.info(f"✅ Query rewrite successful:")
            logger.info(f"   Original: '{user_query}'")
            logger.info(f"   Semantic: '{semantic_query}'")
            logger.info(f"   BM25: '{bm25_query}'")
            
            return semantic_query, bm25_query
        except Exception as e:
            logger.error(f"❌ Query rewrite failed: {e}")
            logger.error(f"   Input query: '{user_query}'")
            logger.error(f"   Model: {self.fast_model if self.fast_model else self.model}")
            raise

    # --- LLMリランク ---
    async def rerank(
        self,
        user_query: str,
        items: List[str],
        prompt_template: str = PROMPT_RERANK,
    ) -> list[int]:
        logger.debug(f"Reranking {len(items)} items for query: '{user_query}'")
        enumerated = "\n".join([f"[{i}] {t}" for i, t in enumerate(items)])
        prompt = prompt_template.format(user_query=user_query, enumerated=enumerated)
        msgs = [
            {"role": "system", "content": "You are a ranking model. Return only a JSON array of integers."},
            {"role": "user", "content": prompt},
        ]
        try:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                reasoning_effort="minimal",
                verbosity="low",
            )
            text = r.choices[0].message.content.strip()

            # JSONパース（壊れていたらフォールバック）
            try:
                arr = json.loads(text)
                if isinstance(arr, list) and all(isinstance(x, int) for x in arr):
                    logger.debug(f"Reranking successful: {len(arr)} items ranked")
                    return arr
            except Exception as e:
                logger.warning(f"Failed to parse rerank result: {e}")

            # 壊れていたらデフォルト順
            logger.warning("Rerank failed, using default order")
            return list(range(len(items)))
        except Exception as e:
            logger.error(f"Rerank request failed: {e}")
            return list(range(len(items)))
