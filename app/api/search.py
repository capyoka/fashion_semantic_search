from fastapi import APIRouter, Response
from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse
from app.core.config import settings
from app.services.aembedder import AsyncEmbedder
from app.services.agenerator import AsyncGenerator
from app.services.retriever import HybridRetriever
from qdrant_client import QdrantClient
import json, os, asyncio

router = APIRouter(tags=["rag"])

# 非同期LLMクライアント
_gen   = AsyncGenerator(settings.OPENAI_CHAT_MODEL)
_embed = AsyncEmbedder(settings.OPENAI_EMBED_MODEL)

# Qdrantは同期クライアントなので、検索時はスレッドに逃す
_qdr   = QdrantClient(path=settings.QDRANT_PATH)

_texts = []
if os.path.exists(settings.PROCESSED_JSONL):
    with open(settings.PROCESSED_JSONL, "r", encoding="utf-8") as f:
        _texts = [json.loads(l)["text"] for l in f]

_ret   = HybridRetriever(_qdr, "fashion", _texts)

@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, resp: Response):
    user_q = req.query

    # 1) クエリ前処理を並列に（書き換え / キーワード抽出）
    rewritten, bm25_kw = await asyncio.gather(
        _gen.rewrite_query(user_q),
        _gen.extract_bm25_keywords(user_q),
    )

    # 2) 埋め込み（書き換え後クエリ）
    qvec = (await _embed.encode([rewritten]))[0]

    # 3) ベクトル検索 + BM25融合（同期→別スレッド実行）
    docs, scores = await asyncio.to_thread(
        _ret.search, rewritten, qvec, req.alpha, req.top_k, bm25_kw
    )

    # 4) LLMリランク（非同期）
    order = await _gen.rerank(user_q, docs)
    order = [i for i in order if 0 <= i < len(docs)] or list(range(len(docs)))
    docs_r   = [docs[i] for i in order]
    scores_r = [scores[i] for i in order]

    # 観測用にヘッダで返す（任意）
    resp.headers["X-Orig-Query"] = user_q
    resp.headers["X-Rewritten-Query"] = rewritten

    return {"context": docs_r[:req.top_k], "scores": scores_r[:req.top_k]}
