import logging
import time
import asyncio
from fastapi import APIRouter, Response
from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse
from app.core.config import settings
from app.services.aembedder import AsyncEmbedder
from app.services.agenerator import AsyncGenerator
from app.services.retriever import HybridRetriever
from app.services.bm25_sparse import BM25Sparse
from qdrant_client import QdrantClient

# Logger configuration
logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])

# ============================================================
# Initialization: Client and Retriever setup
# ============================================================

# Async LLM client
_gen = AsyncGenerator(settings.OPENAI_CHAT_MODEL, settings.OPENAI_CHAT_FAST_MODEL)
_embed = AsyncEmbedder(settings.OPENAI_EMBED_MODEL)

# Qdrant (local or remote)
_qdr = QdrantClient(path=settings.QDRANT_PATH)

# Load BM25 metadata
_bm25 = BM25Sparse.load_from_qdrant(_qdr, settings.QDRANT_COLLECTION)

# Prepare retriever
_ret = HybridRetriever(_qdr, settings.QDRANT_COLLECTION, bm25_indexer=_bm25)

# ============================================================
# Search endpoint
# ============================================================
@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, resp: Response):
    """
    Hybrid search (dense + sparse RRF fusion) + LLM reranking
    """
    start_time = time.time()
    user_q = (req.query or "").strip()
    top_k = req.top_k or 30
    
    logger.info(f"🔍 Search request started - Query: '{user_q}', Top-K: {top_k}")
    logger.debug(f"Request details - User-Agent: {req.__dict__}")

    # 1. Query rewriting (rewrite only)
    query_start = time.time()
    logger.info("📝 Step 1: Starting query rewrite...")
    try:
        rewritten, keywords = await _gen.rewrite_query(user_q)
        rewritten = rewritten.strip() if rewritten else user_q
        keywords = keywords.strip() if keywords else user_q
        query_time = time.time() - query_start
        logger.info(f"✅ Query rewrite completed in {query_time:.3f}s")
        logger.info(f"   Original: '{user_q}'")
        logger.info(f"   Semantic: '{rewritten}'")
        logger.info(f"   Keywords: '{keywords}'")
    except Exception as e:
        logger.error(f"❌ Query rewrite failed: {e}")
        rewritten, keywords = user_q, user_q

    # 2. Embedding generation (async)
    embed_start = time.time()
    logger.info("🧠 Step 2: Generating embeddings...")
    try:
        qvec = (await _embed.encode([rewritten]))[0]
        embed_time = time.time() - embed_start
        logger.info(f"✅ Embedding generated in {embed_time:.3f}s (dim: {len(qvec)})")
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise

    # 3. Hybrid search (sync Qdrant → thread execution)
    search_start = time.time()
    logger.info("🔎 Step 3: Performing hybrid search...")
    try:
        docs = await asyncio.to_thread(_ret.search, keywords, qvec, top_k)
        search_time = time.time() - search_start
        logger.info(f"✅ Hybrid search completed in {search_time:.3f}s")
    except Exception as e:
        logger.error(f"❌ Hybrid search failed: {e}")
        raise

    # Exclude BM25 metadata points (__bm25_meta__)
    original_count = len(docs)
    docs = [d for d in docs if not d.get("_bm25_meta")]
    filtered_count = len(docs)
    if original_count != filtered_count:
        logger.debug(f"Filtered out {original_count - filtered_count} metadata points")

    logger.info(f"📊 Retrieved {len(docs)} documents from search")

    if not docs:
        total_time = time.time() - start_time
        logger.warning(f"⚠️ No documents found for query. Total time: {total_time:.3f}s")
        return {"results": []}

    # 4. LLM reranking (async, safe)
    rerank_start = time.time()
    logger.info("🔄 Step 4: Starting LLM reranking...")
    try:
        order = await _gen.rerank(user_q, docs)
        valid_order = [i for i in order if 0 <= i < len(docs)]
        order = valid_order or list(range(len(docs)))
        docs_r = [docs[i] for i in order]
        rerank_time = time.time() - rerank_start
        logger.info(f"✅ Reranking completed in {rerank_time:.3f}s")
        logger.debug(f"Rerank order: {order[:10]}{'...' if len(order) > 10 else ''}")
    except Exception as e:
        rerank_time = time.time() - rerank_start
        logger.warning(f"⚠️ Rerank failed after {rerank_time:.3f}s: {e}")
        logger.info("Using original search order as fallback")
        docs_r = docs

    # 5. Return top_k results
    final_results = docs_r[:top_k]
    total_time = time.time() - start_time
    
    logger.info(f"🎯 Step 5: Returning {len(final_results)} final results")
    logger.info(f"⏱️ Total search time: {total_time:.3f}s")
    
    # Detailed result logging (debug level)
    if logger.isEnabledFor(logging.DEBUG):
        for i, doc in enumerate(final_results[:3]):  # Top 3 only
            logger.debug(f"Result {i+1}: ID={doc.get('id', 'N/A')}, Score={doc.get('score', 'N/A'):.4f}")
            logger.debug(f"  Title: {doc.get('title', 'N/A')[:100]}...")
    
    return {"results": final_results}
