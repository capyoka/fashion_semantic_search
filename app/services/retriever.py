import logging
import re
import time
from qdrant_client import QdrantClient, models
from openai import OpenAI
from app.services.bm25_sparse import BM25Sparse  # 同じモジュール
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from app.core.config import settings

# Logger設定
logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, qdrant: QdrantClient, collection: str, bm25_indexer: BM25Sparse = None, auto_load: bool = True):
        self.qdrant = qdrant
        self.collection = collection
        self.bm25 = bm25_indexer
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY)

        if self.bm25 is None and auto_load:
            self.bm25 = BM25Sparse.load_from_qdrant(qdrant, collection)
            if self.bm25:
                logger.info("Loaded BM25 metadata")
            else:
                logger.warning("BM25 metadata not found; dense-only")

    def embed_query(self, query: str):
        clean = self._preprocess(query)
        logger.debug(f"Embedding query: '{query}' -> '{clean}'")
        for attempt in range(3):
            try:
                r = self.openai.embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=clean)
                logger.debug("Query embedding successful")
                return clean, r.data[0].embedding
            except Exception as e:
                logger.warning(f"Embedding failed ({attempt+1}): {e}")
                time.sleep(2)
        logger.error("Embedding failed after 3 attempts")
        raise RuntimeError("Embedding failed")

    def _preprocess(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()
        tokens = word_tokenize(text)
        lem = WordNetLemmatizer()
        stp = set(stopwords.words("english"))
        clean = [lem.lemmatize(t) for t in tokens if t not in stp and len(t) > 1]
        return " ".join(clean)

    def search(self, query: str, qvec: list[float], top_k: int = 5, prefetch_k: int = 50):
        logger.info(f"🔍 Starting hybrid search")
        logger.info(f"   Query: '{query}'")
        logger.info(f"   Top-K: {top_k}, Prefetch-K: {prefetch_k}")
        logger.info(f"   Dense vector dim: {len(qvec)}")
        
        # BM25 sparse vector
        sparse_pref = None
        if self.bm25:
            q_idx, q_vals = self.bm25.transform_query(query)
            sparse_pref = models.Prefetch(
                query=models.SparseVector(indices=q_idx, values=q_vals),
                using="text-sparse",
                limit=prefetch_k
            )
            logger.info(f"✅ BM25 sparse vector created: {len(q_idx)} terms")
            logger.debug(f"BM25 terms: {q_idx[:10]}{'...' if len(q_idx) > 10 else ''}")
        else:
            logger.warning("⚠️ BM25 not available, using dense-only search")

        dense_pref = models.Prefetch(
            query=qvec,
            using="text-dense",
            limit=prefetch_k
        )

        prefetchs = [dense_pref]
        if sparse_pref:
            prefetchs.insert(0, sparse_pref)
            logger.info("🔀 Using hybrid search (dense + sparse)")
        else:
            logger.info("🔀 Using dense-only search")

        try:
            res = self.qdrant.query_points(
                collection_name=self.collection,
                prefetch=prefetchs,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                limit=top_k + 1  # 余裕を持たせてメタ除外を見越す
            )
            logger.debug(f"Qdrant returned {len(res.points)} points")
        except Exception as e:
            logger.error(f"❌ Qdrant query failed: {e}")
            raise

        # 結果整形／メタ除外
        docs = []
        metadata_count = 0
        for p in res.points:
            if p.id == "__bm25_meta__":
                metadata_count += 1
                continue
            docs.append({
                "id": p.id,
                "score": p.score,
                **p.payload
            })
        
        if metadata_count > 0:
            logger.debug(f"Filtered out {metadata_count} metadata points")
        
        logger.info(f"✅ Hybrid search completed: {len(docs)} documents found")
        
        # スコア分布のログ（デバッグレベル）
        if logger.isEnabledFor(logging.DEBUG) and docs:
            scores = [doc.get('score', 0) for doc in docs]
            logger.debug(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
            logger.debug(f"Top 3 scores: {sorted(scores, reverse=True)[:3]}")
        
        return docs