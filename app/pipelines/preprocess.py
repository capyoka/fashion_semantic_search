import os
import json
import time
import uuid
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ============================================================
# 設定
# ============================================================
COLLECTION_NAME = "fashion_hybrid"
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CAPTION_SYSTEM_PROMPT = """
You are a factual assistant for e-commerce product images.
Always write in English.
Produce concise, objective captions (3–5 sentences).
Describe only what is visible in the image or explicitly provided in the user prompt.
Do not speculate about attributes that are not clearly visible or provided.
For Brand, include it only if it is verifiable from the image (logo/label) or explicitly given in the prompt; otherwise, omit any brand claim.
Avoid marketing language, opinions, and unverifiable claims.
Prefer concrete attributes: category, color, material, silhouette/fit, notable details.
Mention season/occasion/style only if clearly supported by visible cues (coverage, fabric weight, sparkle, etc.).
"""

CAPTION_USER_PROMPT = """
Write one concise factual caption (3–5 sentences) for this product image using both the visual content and the context below.
Do NOT contradict what is visible in the image. If the context mentions attributes that are not visible, include them only if they do not contradict the image.
Prefer concrete attributes: category, color, material, silhouette/fit, notable details.
Mention season/occasion/style only if supported by visible cues or explicitly provided.
For Brand, include it only if it is visible in the image or explicitly present in the context (Title/Store); otherwise, omit any brand claim.
Keep a neutral, non-marketing tone.

Context:
Title: {title}
Store: {store}
Features: {features}
"""

CAPTION_MAX_IMAGES_PER_ITEM = 2
CAPTION_MAX_WORKERS = 6
EMBED_BATCH_SIZE = 64
EMBED_MAX_WORKERS = 4
QDRANT_BATCH_SIZE = 256


# ============================================================
# ユーティリティ
# ============================================================
def iter_json(path: str):
    """JSONL or JSON配列を読み込むジェネレータ"""
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
    """画像URLを抽出（最大N件）"""
    imgs = product.get("images", [])
    refs = []
    if isinstance(imgs, list):
        for img in imgs:
            if isinstance(img, dict) and "hi_res" in img:
                refs.append(img["hi_res"])
            elif isinstance(img, str):
                refs.append(img)
    return refs[:CAPTION_MAX_IMAGES_PER_ITEM]


# ============================================================
# 画像キャプション生成
# ============================================================
def _caption_worker(image_url: str, title: str, store: str, features: str) -> Optional[str]:
    """1画像＋文脈でキャプション生成"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    user_prompt = CAPTION_USER_PROMPT.format(
        title=title or "",
        store=store or "",
        features=features or "None"
    )

    try:
        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        r = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.2
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return None


def caption_images_parallel(image_urls: List[str], title: str, store: str, features: str) -> List[str]:
    """画像＋文脈キャプションを並列生成"""
    if not image_urls:
        return []
    caps = []
    with ThreadPoolExecutor(max_workers=CAPTION_MAX_WORKERS) as ex:
        futs = {
            ex.submit(_caption_worker, url, title, store, features): url
            for url in image_urls[:CAPTION_MAX_IMAGES_PER_ITEM]
        }
        for fut in as_completed(futs):
            res = fut.result()
            if res:
                caps.append(res)
    return caps


# ============================================================
# BM25 Sparse
# ============================================================
class BM25Sparse:
    def __init__(self, corpus: List[str]):
        self.tokenized = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized)

    def transform_doc(self, text: str):
        tokens = text.split()
        scores = {t: self.bm25.idf.get(t, 0.0) for t in tokens}
        idx = list(range(len(scores)))
        vals = list(scores.values())
        return idx, vals


# ============================================================
# Embedding
# ============================================================
def embed_parallel(texts: List[str], model_name: str = EMBED_MODEL) -> List[List[float]]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    if not texts:
        return []
    vectors = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="embedding"):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        r = client.embeddings.create(model=model_name, input=batch)
        vectors.extend([d.embedding for d in r.data])
    return vectors


# ============================================================
# メイン処理：JSON → Qdrant (dense + sparse + raw data)
# ============================================================
def run(input_path: str, qdrant_path: str = "./qdrant_hybrid"):
    os.makedirs(qdrant_path, exist_ok=True)
    qdrant = QdrantClient(path=qdrant_path)

    all_texts = []   # 検索用テキスト
    payloads = []    # Qdrantに格納する全情報

    # --- Step 1: 元データ処理 + キャプション生成 ---
    for prod in tqdm(iter_json(input_path), desc="process"):
        pid = prod.get("parent_asin") or prod.get("asin") or stable_uuid(json.dumps(prod))
        title = prod.get("title", "")
        store = prod.get("store", "")
        features = "; ".join(prod.get("features", []))
        images = extract_image_refs(prod)

        captions = caption_images_parallel(images, title, store, features)
        caption_text = " ".join(captions)

        # 🔹 検索用テキスト（title + features + captions）
        search_text = " ".join([
            title or "",
            features or "",
            caption_text or ""
        ]).strip()

        # 🔹 payloadには「元データ + 拡張情報」をすべて格納
        payload = {
            **prod,                      # 元データ構造をそのまま保持
            "_id": pid,                  # 安定UUID
            "_captions": captions,       # LLM生成キャプション
            "_search_text": search_text  # 検索対象テキスト
        }
        payloads.append(payload)
        all_texts.append(search_text)

    # --- Step 2: Dense Embedding ---
    dense = embed_parallel(all_texts)

    # --- Step 3: Sparse (BM25) ---
    bm25 = BM25Sparse(all_texts)

    # --- Step 4: Qdrant コレクション作成 ---
    dim = len(dense[0])
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text-dense": models.VectorParams(size=dim, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams()
        }
    )


    # --- Step 5: Upsert (dense + sparse + payload) ---
    points = []
    for i, (vec, text, payload) in enumerate(zip(dense, all_texts, payloads)):
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

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Hybrid Qdrant built with {len(points)} items.")
