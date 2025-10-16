from fastapi import APIRouter, Response
from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse
from app.core.config import settings
from app.services.aembedder import AsyncEmbedder
from app.services.agenerator import AsyncGenerator
from app.services.retriever import HybridRetriever
from app.services.bm25_sparse import BM25Sparse
from qdrant_client import QdrantClient
import asyncio

router = APIRouter(tags=["rag"])

# ============================================================
# 初期化：クライアント・Retriever 準備
# ============================================================

# 非同期 LLM クライアント
_gen = AsyncGenerator(settings.OPENAI_CHAT_MODEL, settings.OPENAI_CHAT_FAST_MODEL)
_embed = AsyncEmbedder(settings.OPENAI_EMBED_MODEL)

# Qdrant（ローカル or Remote）
_qdr = QdrantClient(path=settings.QDRANT_PATH)

# BM25 メタデータ読み込み
_bm25 = BM25Sparse.load_from_qdrant(_qdr, settings.QDRANT_COLLECTION)

# Retriever 準備
_ret = HybridRetriever(_qdr, settings.QDRANT_COLLECTION, bm25_indexer=_bm25)

# ============================================================
# 検索エンドポイント
# ============================================================
@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, resp: Response):
    """
    ハイブリッド検索 (dense + sparse RRF融合) + LLMリランク
    """
    user_q = (req.query or "").strip()
    top_k = req.top_k or 30

    # 1️⃣ クエリ書き換え（rewrite のみ実行）
    rewritten, keywords = await _gen.rewrite_query(user_q)
    rewritten = rewritten.strip() if rewritten else user_q
    keywords = keywords.strip() if keywords else user_q

    # 2️⃣ 埋め込み生成（非同期）
    qvec = (await _embed.encode([rewritten]))[0]

    # 3️⃣ ハイブリッド検索（同期Qdrant → スレッド実行）
    docs = await asyncio.to_thread(_ret.search, keywords, qvec, top_k)

    # BM25メタデータPoint（__bm25_meta__）を除外
    docs = [d for d in docs if not d.get("_bm25_meta")]

    if not docs:
        return {"results": []}

    # 4️⃣ LLM リランク（非同期・安全化）
    try:
        order = await _gen.rerank(user_q, docs)
        valid_order = [i for i in order if 0 <= i < len(docs)]
        order = valid_order or list(range(len(docs)))
        docs_r = [docs[i] for i in order]
    except Exception as e:
        print(f"[Warn] Rerank failed: {e}")
        docs_r = docs

    # 5️⃣ 上位 top_k 結果を返す
    return {"results": docs_r[:top_k]}
