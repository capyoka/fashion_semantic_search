from qdrant_client import QdrantClient, models
from openai import OpenAI
from app.services.bm25_sparse import BM25Sparse  # 同じモジュール
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import time
from app.core.config import settings


class HybridRetriever:
    def __init__(self, qdrant: QdrantClient, collection: str, bm25_indexer: BM25Sparse = None, auto_load: bool = True):
        self.qdrant = qdrant
        self.collection = collection
        self.bm25 = bm25_indexer
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY)

        if self.bm25 is None and auto_load:
            self.bm25 = BM25Sparse.load_from_qdrant(qdrant, collection)
            if self.bm25:
                print("[HybridRetriever] Loaded BM25 metadata")
            else:
                print("[HybridRetriever] Warning: BM25 metadata not found; dense-only")

    def embed_query(self, query: str):
        clean = self._preprocess(query)
        for attempt in range(3):
            try:
                r = self.openai.embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=clean)
                return clean, r.data[0].embedding
            except Exception as e:
                print(f"Embedding failed ({attempt+1}): {e}")
                time.sleep(2)
        raise RuntimeError("Embedding failed")

    def _preprocess(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()
        tokens = word_tokenize(text)
        lem = WordNetLemmatizer()
        stp = set(stopwords.words("english"))
        clean = [lem.lemmatize(t) for t in tokens if t not in stp and len(t) > 1]
        return " ".join(clean)

    def search(self, query: str, qvec: list[float], top_k: int = 5, prefetch_k: int = 50):
        # BM25 sparse vector
        sparse_pref = None
        if self.bm25:
            q_idx, q_vals = self.bm25.transform_query(query)
            sparse_pref = models.Prefetch(
                query=models.SparseVector(indices=q_idx, values=q_vals),
                using="text-sparse",
                limit=prefetch_k
            )

        dense_pref = models.Prefetch(
            query=qvec,
            using="text-dense",
            limit=prefetch_k
        )

        prefetchs = [dense_pref]
        if sparse_pref:
            prefetchs.insert(0, sparse_pref)

        res = self.qdrant.query_points(
            collection_name=self.collection,
            prefetch=prefetchs,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=top_k + 1  # 余裕を持たせてメタ除外を見越す
        )

        # 結果整形／メタ除外
        docs = []
        for p in res.points:
            if p.id == "__bm25_meta__":
                continue
            docs.append({
                "id": p.id,
                "score": p.score,
                **p.payload
            })
        return docs