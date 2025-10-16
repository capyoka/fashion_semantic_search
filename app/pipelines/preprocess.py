"""
pipelines/hybrid_indexer.py

Qdrantに対して Dense（OpenAI Embedding）+ Sparse（BM25）を登録するハイブリッドインデクサ。
BM25の語彙情報をコレクションmetadataとして保存し、後で検索側で完全復元できる。
"""

import os
import re
import json
import time
import uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from tqdm import tqdm
from qdrant_client import QdrantClient, models
from openai import OpenAI
from openai import RateLimitError, APIError

from app.core.config import settings
from app.services.bm25_sparse import BM25Sparse


# ============================================================
# 設定
# ============================================================
CHAT_MODEL = settings.OPENAI_CHAT_MODEL
EMBED_MODEL = settings.OPENAI_EMBED_MODEL
QDRANT_PATH = os.path.abspath(settings.QDRANT_PATH)
COLLECTION_NAME = settings.QDRANT_COLLECTION
OPENAI_API_KEY = settings.OPENAI_API_KEY

CAPTION_MAX_WORKERS = 5
EMBED_BATCH_SIZE = 64
QDRANT_BATCH_SIZE = 256


# ============================================================
# ユーティリティ
# ============================================================
def iter_json(path: str):
    """JSONL または JSON配列を読み込む"""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for item in json.load(f):
                yield item
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def stable_uuid(val: str) -> str:
    """文字列から安定的なUUIDを生成"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, val))


def extract_image_refs(product: Dict) -> List[str]:
    """画像URLを抽出"""
    imgs = product.get("images", [])
    refs = []
    for img in imgs:
        if isinstance(img, dict) and "large" in img:
            refs.append(img["large"])
        elif isinstance(img, str):
            refs.append(img)
    return refs[:1]


# ============================================================
# キャプション生成
# ============================================================

CAPTION_SYSTEM_PROMPT = """
You are an assistant that generates factual captions for e-commerce product images.  
Write one objective caption in English, consisting of 3–5 sentences.

## Rules
- Describe only what is visible in the image or what is explicitly stated in the provided context or user intent.  
- If the image and text conflict, always prioritize the image.  
- Output only the caption text.

## Key aspects to consider
Focus on the following attributes when writing the caption.  
You do not need to include every element.  
Do not include information such as purpose, scene, or other attributes if they cannot be clearly identified from the image or context.

- Brand name, material, color, pattern, and structural details (sleeves, length, collar, waist, buttons, pockets, etc.)  
- Silhouette and fit (e.g., loose, tight, cropped)  
- Texture and fabric appearance (e.g., sheer, opaque, glossy, draped, stretchy)  
- Sets or bundled items, and accessory-specific features (e.g., lens shape, hardware, fastener type)  
- Target (men’s / women’s / unisex / kids)  
- Purpose or scene (casual, office, travel, beach, formal, party, etc.)  
- Age group (adult / teen / child)  
- Seasonal impression (spring / summer / fall / winter)  
- Visual characteristics (e.g., lightweight-looking, warm-textured — objective, appearance-based expressions only)

Do not describe brand, performance, or functional claims (e.g., waterproof, quick-drying, UV protection)  
unless they are explicitly visible in the image or stated in the text.  
"""

CAPTION_USER_PROMPT = """
Input:
- Image
- Context:
  - Title: {title}
  - Store: {store}
  - Features: {features}

Task:
Generate one factual, English caption (3–5 sentences) based on the visible attributes in the image,  
integrating any explicitly confirmed information from the context.  
Output only the caption text.
"""

# CAPTION_SYSTEM_PROMPT = """
# あなたはEC商品の画像キャプションを作成するアシスタントです。
# 英語で、客観的なキャプションを1つ（3〜5文）出力します。

# ## ルール
# - 画像に見える内容、または提供コンテキスト/ユーザ意図に明示された事実のみを書くこと。
# - 画像とテキストが矛盾する場合は、画像を優先すること。
# - 出力はキャプション本文のみ。

# ## キャプションに含める観点
# 以下の要素に注目してキャプションを作成してください。
# すべての要素を必ずしも含む必要はありません、用途やシーンなどは画像から判断できない場合は含めないでください。
# - ブランド名、素材、色、パターン、構造的ディテール（袖・丈・襟・ウエスト・ボタン・ポケットなど）
# - シルエットやフィット（loose/tight、croppedなど）
# - テクスチャ・質感（sheer/opaque/glossy/drape/stretch など）
# - セット/同梱やアクセサリー特有の要素（例：レンズ形状、金具、留め具など）
# - 対象（men’s / women’s / unisex / kids）
# - 用途・シーン（casual, office, travel, beach, formal, party など）
# - 年齢層（adult / teen / child など）
# - 季節感（spring / summer / fall / winter）
# - 印象的特徴（lightweight-looking, warm-textured など、見た目に基づく客観的な表現）

# ブランドや性能、機能（例：防水・吸汗速乾・UVカットなど）の解釈の余地がない要素は画像やテキストに明記されない限り絶対に記述しないでください。
# """

# CAPTION_USER_PROMPT = """
# 入力：
# - 画像
# - コンテキスト:
#   - Title: {title}
#   - Store: {store}
#   - Features: {features}

# キャプション:
# """


def caption_images(image_urls: List[str], title: str, store: str, features: str) -> Optional[str]:
    """1商品分のキャプションを生成（堅牢なリトライ付き）"""
    if not image_urls:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_prompt = CAPTION_USER_PROMPT.format(
        title=title or "",
        store=store or "",
        features=features or "None",
    )

    # 画像は1枚ずつ送る想定
    messages = [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_urls[0]}},
            ],
            "reasoning_effort": "minimal",
            "verbosity": "low",
        },
    ]

    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                timeout=60,  # 念のためタイムアウト指定
                reasoning_effort="minimal",
                verbosity="low",
            )
            caption = r.choices[0].message.content.strip()
            return caption

        except RateLimitError as e:
            # ✅ レート制限時：指数バックオフ＋ランダムスリープ
            wait = min(30, 2 ** attempt + random.uniform(0, 3))
            print(f"RateLimitError: retrying in {wait:.1f}s ({attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
        except APIError as e:
            # ✅ APIエラー（5xx系）：指数バックオフで再試行
            wait = min(20, 2 ** attempt)
            print(f"APIError: {e}. retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            # その他エラーは1回だけ待って続行
            print(f"Caption generation failed ({attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(3)
    print("Caption generation failed after maximum retries.")
    return None


# ============================================================
# Embedding
# ============================================================
def embed_parallel(texts: List[str], model_name: str = EMBED_MODEL) -> List[List[float]]:
    """OpenAI埋め込みをバッチ生成"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    vectors = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="embedding"):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        try:
            r = client.embeddings.create(model=model_name, input=batch)
            vectors.extend([d.embedding for d in r.data])
        except Exception as e:
            print(f"Embedding failed at batch {i}: {e}")
            time.sleep(5)
    return vectors


# ============================================================
# メイン処理
# ============================================================
# ------------------------------------------------------------
# メイン処理：JSON → Qdrant (dense + sparse + payload + BM25 meta)
def run(input_path: str, qdrant_path: str = QDRANT_PATH):
    os.makedirs(qdrant_path, exist_ok=True)
    qdrant = QdrantClient(path=qdrant_path)

    products = list(iter_json(input_path))
    all_texts = []
    payloads = []

    # キャプション生成（並列化）
    def process_prod(prod):
        pid = prod.get("asin") or stable_uuid(json.dumps(prod))
        title = prod.get("title", "")
        store = prod.get("store", "")
        features = "; ".join(prod.get("features", []))
        images = extract_image_refs(prod)
        caption = caption_images(images, title, store, features)
        search_text = " ".join([title, features, caption or ""]).strip()
        payload = {**prod, "_id": pid, "_caption": caption, "_search_text": search_text}
        return payload, search_text

    print(f"Generating captions in parallel ({CAPTION_MAX_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=CAPTION_MAX_WORKERS) as executor:
        futures = [executor.submit(process_prod, p) for p in products]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="caption"):
            payload, text = fut.result()
            payloads.append(payload)
            all_texts.append(text)

    # Dense embedding
    dense = embed_parallel(all_texts, EMBED_MODEL)

    # BM25 モデル作成
    bm25 = BM25Sparse(all_texts)

    # Qdrant コレクション作成（dense + sparse）
    dim = len(dense[0])
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text-dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams()
        }
    )

    # BM25 メタ保存（ダミー Point）
    bm25.save_to_qdrant(qdrant, COLLECTION_NAME)

    # 各商品 upsert
    points = []
    for vec, text, payload in zip(dense, all_texts, payloads):
        idx, vals = bm25.transform_doc(text)
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(payload["_id"])))
        points.append(
            models.PointStruct(
                id=uid,
                vector={
                    "text-dense": vec,
                    "text-sparse": models.SparseVector(indices=idx, values=vals),
                },
                payload=payload,
            )
        )
        if len(points) >= QDRANT_BATCH_SIZE:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            points.clear()
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"✅ Hybrid index built: {len(payloads)} points (BM25 meta included)")