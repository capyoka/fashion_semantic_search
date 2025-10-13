import os
import io
import json
import time
import base64
import hashlib
from typing import Iterable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.services.generator import Generator            # OpenAI Chat (text)
from app.services.embedder import Embedder              # OpenAI Embedding

# ====== 設定 ======
COLLECTION_NAME = "fashion"

# 画像キャプション
VISION_MODEL = "gpt-4o-mini"
CAPTION_MAX_IMAGES_PER_ITEM = 2      # 1商品の上限
CAPTION_MAX_WORKERS = 10             # ★ 並列キャプションのスレッド数
CAPTION_MAX_RETRIES = 4
CAPTION_RETRY_BASE = 1.6

# 埋め込み
EMBED_BATCH_SIZE = 64                # 1リクエストのバッチ
EMBED_MAX_WORKERS = 4                # ★ バッチ処理を並列で投げる数（レートと相談）
EMBED_MAX_RETRIES = 4
EMBED_RETRY_BASE = 1.6

# Qdrant
QDRANT_BATCH_SIZE = 256

# LLMテキスト整形
TEXT_CLEAN_PROMPT = """以下はECサイトの商品説明（title/description/features）です。
冗長/販促/重複を削除し、事実ベースの特徴だけを日本語で150字以内に要約してください。
---
{content}
"""

CAPTION_PROMPT = "この商品の画像を短く（1〜2文）日本語で説明してください。"

# ========== ユーティリティ ==========
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: str, rows: List[Dict], append: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stable_id(prod: Dict) -> str:
    for key in ["parent_asin", "asin", "id", "sku"]:
        if key in prod and prod[key]:
            return str(prod[key])
    h = hashlib.sha256(json.dumps(prod, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return h[:16]

# ====== テキスト整形（LLM要約） ======
def build_clean_text(gen: Generator, product: Dict) -> str:
    title = product.get("title", "") or ""
    desc  = product.get("description", [])
    if isinstance(desc, list):
        desc = " ".join(desc)
    desc = str(desc or "")
    feats = product.get("features", [])
    if isinstance(feats, list):
        feats = " ".join(feats)
    feats = str(feats or "")
    content = f"title: {title}\ndescription: {desc}\nfeatures: {feats}"
    return gen.generate(query=TEXT_CLEAN_PROMPT.format(content=content), context="")

# ====== 画像パス/URL 取り出し ======
def extract_image_refs(product: Dict) -> List[str]:
    refs: List[str] = []
    imgs = product.get("images")
    if not imgs:
        return refs

    def _push(v):
        if isinstance(v, str) and v.strip():
            refs.append(v.strip())

    if isinstance(imgs, dict):
        for _, arr in imgs.items():
            if isinstance(arr, list):
                for v in arr:
                    _push(v)
            else:
                _push(arr)
    elif isinstance(imgs, list):
        for v in imgs:
            _push(v)
    elif isinstance(imgs, str):
        _push(imgs)

    # 重複排除
    seen = set()
    out = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out

def _file_to_data_url(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path.lower())[1].lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

# ====== 画像→キャプション（並列） ======
def _caption_worker(image_ref: str, model: str) -> Optional[str]:
    """
    1画像を説明するワーカー。各スレッドで OpenAI クライアントを都度生成（スレッド安全性を担保）。
    """
    from openai import OpenAI
    import os, time
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    url = image_ref
    if not image_ref.startswith("http"):
        data_url = _file_to_data_url(image_ref)
        if not data_url:
            return None
        url = data_url

    for attempt in range(CAPTION_MAX_RETRIES):
        try:
            messages = [
                {"role": "system", "content": "Describe the image in concise Japanese."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                },
            ]
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            time.sleep((CAPTION_RETRY_BASE ** attempt) + 0.25 * attempt)
    return None

def caption_images_parallel(image_refs: List[str], max_images: int, max_workers: int) -> List[str]:
    """
    image_refs から max_images 件を並列キャプション。
    """
    refs = image_refs[:max_images]
    if not refs:
        return []
    captions: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_caption_worker, ref, VISION_MODEL): ref for ref in refs}
        for fut in as_completed(futs):
            cap = fut.result()
            if cap:
                captions.append(cap)
    return captions

# ====== テキスト結合 ======
def merge_text(clean_text: str, captions: List[str]) -> str:
    return clean_text if not captions else clean_text + " " + " ".join(captions)

# ====== 埋め込み（並列バッチ投げ） ======
def _embed_batch_worker(batch_texts: List[str], model_name: str) -> List[List[float]]:
    """
    バッチ単位で埋め込みを投げるワーカー。各スレッドでクライアントを生成。
    """
    from openai import OpenAI
    import os, time
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            # OpenAI embeddings API: まとめて投げる
            res = client.embeddings.create(model=model_name, input=batch_texts)
            return [d.embedding for d in res.data]
        except Exception:
            time.sleep((EMBED_RETRY_BASE ** attempt) + 0.25 * attempt)
    # 失敗時はゼロ配列返却（後で次元合わせ）
    return [[0.0] for _ in batch_texts]

def embed_parallel(texts: List[str], batch_size: int, max_workers: int, model_name: str) -> List[List[float]]:
    """
    テキストを batch_size ごとに分割し、max_workers 並列で埋め込み。
    """
    if not texts:
        return []
    batches: List[List[str]] = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    vectors: List[List[float]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_embed_batch_worker, b, model_name) for b in batches]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="embedding(parallel)"):
            vectors.extend(fut.result())
    return vectors

# ====== Qdrant バッチUpsert ======
def upsert_qdrant_in_batches(qdr: QdrantClient, collection: str, vectors: List[List[float]], payloads: List[Dict], batch_size: int):
    for i in tqdm(range(0, len(vectors), batch_size), desc="qdrant_upsert"):
        vs = vectors[i:i+batch_size]
        pls = payloads[i:i+batch_size]
        ids = [str(i0) for i0 in range(i, i + len(vs))]
        qdr.upsert(collection_name=collection, points=models.Batch(ids=ids, vectors=vs, payloads=pls))

# ====== メイン実行 ======
def run(src_jsonl: str, out_jsonl: str, qdrant_path: str) -> None:
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    os.makedirs(qdrant_path, exist_ok=True)

    # 生成系はシングルトン1つでOK（要約は1件ずつ同期で十分）
    gen = Generator(settings.OPENAI_CHAT_MODEL)
    qdr = QdrantClient(path=qdrant_path)

    processed: List[Dict] = []
    texts: List[str] = []
    payloads: List[Dict] = []

    # 1) 要約 + 画像キャプション（キャプションだけ並列）
    for idx, prod in enumerate(tqdm(iter_jsonl(src_jsonl), desc="preprocess(jsonl)")):
        pid = stable_id(prod)
        clean = build_clean_text(gen, prod)

        img_refs = extract_image_refs(prod)
        captions = caption_images_parallel(img_refs, CAPTION_MAX_IMAGES_PER_ITEM, CAPTION_MAX_WORKERS)

        merged = merge_text(clean, captions)

        rec = {
            "id": pid,
            "idx": idx,
            "text": merged,
            "captions": captions,
            "image_refs": img_refs[:CAPTION_MAX_IMAGES_PER_ITEM],
        }
        processed.append(rec)
        texts.append(merged)
        payloads.append({"id": pid, "idx": idx, "text": merged})

    # 2) 保存（BM25コーパス兼ねる）
    write_jsonl(out_jsonl, processed, append=False)

    # 3) 埋め込み（並列）
    embed_model = settings.OPENAI_EMBED_MODEL
    vectors = embed_parallel(texts, EMBED_BATCH_SIZE, EMBED_MAX_WORKERS, embed_model)

    # 次元あわせ
    dim = max((len(v) for v in vectors if len(v) > 0), default=0)
    if dim == 0:
        raise RuntimeError("No valid embeddings produced.")
    vectors = [v if len(v) == dim else ([0.0] * dim) for v in vectors]

    # 4) Qdrant へ再作成→登録
    qdr.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    upsert_qdrant_in_batches(qdr, COLLECTION_NAME, vectors, payloads, QDRANT_BATCH_SIZE)

    print("Preprocess (parallel captions & parallel embeddings) completed.")
