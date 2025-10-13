from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import numpy as np

def _norm(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0: return a
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12: return np.zeros_like(a)
    return (a - mn) / (mx - mn + 1e-12)

class HybridRetriever:
    def __init__(self, qdrant: QdrantClient, collection: str, bm25_texts: list[str] | None):
        self.qdrant = qdrant
        self.collection = collection
        self.bm25 = BM25Okapi([t.split() for t in bm25_texts]) if bm25_texts else None

    def search(
        self,
        query: str,
        qvec: list[float],
        alpha: float = 0.6,
        top_k: int = 5,
        bm25_tokens: list[str] | None = None,   # ★ 追加：BM25用トークンを外部指定可
    ):
        # Qdrant（ベクトル）
        hits = self.qdrant.search(collection_name=self.collection, query_vector=qvec, limit=top_k)
        cand_texts = [h.payload.get("text", "") for h in hits]
        vec_scores = [1 - h.score for h in hits]  # cosine 距離→類似度

        # BM25（候補上で簡易再計算）※全文コーパスBM25でもOKだが手軽さ優先
        if self.bm25 and cand_texts:
            bm25_local = BM25Okapi([t.split() for t in cand_texts])
            tokens = bm25_tokens if bm25_tokens else query.split()
            bm25_scores = bm25_local.get_scores(tokens)
        else:
            bm25_scores = [0.0] * len(cand_texts)

        v, b = _norm(vec_scores), _norm(bm25_scores)
        hybrid = (alpha * b + (1 - alpha) * v).tolist()
        ranked = sorted(zip(cand_texts, hybrid), key=lambda x: -x[1])[:top_k]
        docs = [d for d, _ in ranked]
        scores = [s for _, s in ranked]
        return docs, scores
