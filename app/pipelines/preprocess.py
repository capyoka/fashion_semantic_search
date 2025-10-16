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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from tqdm import tqdm
from qdrant_client import QdrantClient, models
from openai import OpenAI

from app.core.config import settings
from app.services.bm25_sparse import BM25Sparse


# ============================================================
# 設定
# ============================================================
CHAT_MODEL = settings.OPENAI_CHAT_MODEL
EMBED_MODEL = settings.OPENAI_EMBED_MODEL or "text-embedding-3-large"
QDRANT_PATH = os.path.abspath(settings.QDRANT_PATH)
COLLECTION_NAME = settings.QDRANT_COLLECTION
OPENAI_API_KEY = settings.OPENAI_API_KEY

CAPTION_MAX_WORKERS = 6
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
You are a factual assistant for e-commerce product images.
Always write in English.
Produce concise, objective captions (3–5 sentences).
Describe only what is visible in the image or explicitly provided in the user prompt.
Avoid speculation or marketing tone.
"""

CAPTION_USER_PROMPT = """
Write one concise factual caption (3–5 sentences) for this product image using both the visual content and the context below.
Context:
Title: {title}
Store: {store}
Features: {features}
"""


def caption_images(image_urls: List[str], title: str, store: str, features: str) -> Optional[str]:
    """1商品分のキャプションを生成"""
    if not image_urls:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_prompt = CAPTION_USER_PROMPT.format(
        title=title or "",
        store=store or "",
        features=features or "None",
    )

    messages = [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                *[{"type": "image_url", "image_url": {"url": url}} for url in image_urls],
            ],
        },
    ]

    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Caption generation failed ({attempt+1}/3): {e}")
            time.sleep(3)
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
            print(f"❌ Embedding failed at batch {i}: {e}")
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

    print(f"🧠 Generating captions in parallel ({CAPTION_MAX_WORKERS} workers)...")
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