from qdrant_client import QdrantClient, models
import numpy as np


class HybridRetriever:
    """
    QdrantのRRF融合（dense + BM25 sparse）を使うHybrid Retriever。
    """

    def __init__(
        self,
        qdrant: QdrantClient,
        collection: str,
        bm25_indexer=None,
        auto_load_bm25: bool = True,
    ):
        """
        Args:
            qdrant: QdrantClient (例: QdrantClient("http://qdrant:6333"))
            collection: コレクション名
            bm25_indexer: 既存のBM25Sparseインスタンス（指定しない場合はQdrantからロード）
            auto_load_bm25: Trueなら自動でQdrantからBM25メタをロード
        """
        self.qdrant = qdrant
        self.collection = collection
        self.bm25 = bm25_indexer

        if self.bm25 is None and auto_load_bm25:
            print(f"[Init] Loading BM25 metadata for collection '{collection}' ...")
            try:
                self.bm25 = load_bm25_from_qdrant(collection)
                print("[Init] BM25 loaded successfully.")
            except Exception as e:
                print(f"[Warn] Failed to load BM25 from Qdrant: {e}")
                self.bm25 = None  # dense-only fallback可能に

    # ---------------------------
    # 検索メイン関数
    # ---------------------------
    def search(
        self,
        query: str,
        qvec: list[float],
        top_k: int = 5,
        prefetch_k: int = 50,
    ):
        """
        Dense + Sparse のハイブリッド検索（Qdrant内RRF融合）

        Args:
            query: ユーザクエリ
            qvec: OpenAI埋め込みベクトル（例: text-embedding-3-large）
            top_k: 取得件数
            prefetch_k: dense/sparseそれぞれがfusion前に取得する候補数
        """
        # --- Sparse（BM25） ---
        if self.bm25:
            q_idx, q_vals = self.bm25.transform_query(query)
            sparse_prefetch = models.Prefetch(
                query=models.SparseVector(indices=q_idx, values=q_vals),
                using="text-sparse",
                limit=prefetch_k,
            )
        else:
            sparse_prefetch = None
            print("[Warn] BM25 indexer unavailable. Using dense-only search.")

        # --- Dense ---
        dense_prefetch = models.Prefetch(
            query=qvec,
            using="text-dense",
            limit=prefetch_k,
        )

        # --- Qdrant内RRF融合検索 ---
        prefetch_list = [dense_prefetch]
        if sparse_prefetch:
            prefetch_list.insert(0, sparse_prefetch)

        res = self.qdrant.query_points(
            collection_name=self.collection,
            prefetch=prefetch_list,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=top_k,
        )

        return res.points
