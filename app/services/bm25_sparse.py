"""
bm25_sparse.py

BM25スコアリングのロジックと、
Qdrantへの永続化（save/load）機能を統合したモジュール。
"""

from collections import Counter
import math
from qdrant_client import QdrantClient, models


class BM25Sparse:
    """
    軽量BM25実装（tokenize→df→idf→sparse vector変換）
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self.df = {}       # term -> document frequency
        self.N = 0         # total number of documents
        self.avgdl = 0.0   # average document length
        self.tokenizer_version = "basic_v1"

    # ------------------------
    # Fitting
    # ------------------------
    def fit(self, docs: list[str]):
        """全文書からdf, N, avgdlを計算"""
        self.N = len(docs)
        df_counter = Counter()
        total_len = 0

        for doc in docs:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            for t in set(tokens):
                df_counter[t] += 1

        self.df = dict(df_counter)
        self.avgdl = total_len / self.N if self.N else 0

    # ------------------------
    # Query transform
    # ------------------------
    def transform_query(self, query: str):
        """クエリ文字列→(indices, values)"""
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        indices, values = [], []

        for term, f in tf.items():
            if term not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))
            score = idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b))
            indices.append(hash(term) % (2**31))  # ← termのhashをint index化
            values.append(score)
        return indices, values

    def transform_doc(self, doc: str):
        """文書→(indices, values)"""
        tokens = self._tokenize(doc)
        tf = Counter(tokens)
        indices, values = [], []

        for term, f in tf.items():
            if term not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5))
            score = idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b))
            indices.append(hash(term) % (2**31))
            values.append(score)
        return indices, values

    # ------------------------
    # Qdrant persistence
    # ------------------------
    def save_to_qdrant(self, qdrant: QdrantClient, collection: str):
        """BM25メタをQdrantに保存"""
        meta = {
            "_bm25_meta": True,
            "N": self.N,
            "avgdl": self.avgdl,
            "df": self.df,
            "tokenizer": self.tokenizer_version,
        }
        qdrant.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id="__bm25_meta__", payload=meta)
            ],
        )
        print(f"[BM25] Saved to Qdrant: N={self.N}, avgdl={self.avgdl:.2f}, vocab={len(self.df)}")

    @classmethod
    def load_from_qdrant(cls, qdrant: QdrantClient, collection: str):
        """QdrantからBM25メタをロード"""
        pts = qdrant.retrieve(collection_name=collection, ids=["__bm25_meta__"])
        if not pts or not pts[0].payload.get("_bm25_meta"):
            raise RuntimeError(f"[BM25] meta not found in collection '{collection}'")

        meta = pts[0].payload
        bm25 = cls(k1=0.9, b=0.4)
        bm25.N = meta["N"]
        bm25.avgdl = meta["avgdl"]
        bm25.df = {k: int(v) for k, v in meta["df"].items()}
        bm25.tokenizer_version = meta.get("tokenizer", "basic_v1")
        print(f"[BM25] Loaded from Qdrant: N={bm25.N}, avgdl={bm25.avgdl:.2f}, vocab={len(bm25.df)}")
        return bm25

    # ------------------------
    # Utilities
    # ------------------------
    @staticmethod
    def _tokenize(text: str):
        """単純なトークナイザ（英語・ローマ字前提）"""
        return [t.lower() for t in text.split() if t.strip()]
