from qdrant_client import QdrantClient, models
import numpy as np

class HybridRetriever:
    """
    QdrantのRRF融合（dense + BM25 sparse）を使うHybrid Retriever。
    あなたの旧HybridRetrieverと同じ使い方ができます。
    """

    def __init__(self, qdrant: QdrantClient, collection: str, bm25_indexer=None):
        """
        Args:
            qdrant: QdrantClient (例: QdrantClient(":memory:"))
            collection: コレクション名
            bm25_indexer: BM25Sparse インスタンス（idf情報を持つもの）
        """
        self.qdrant = qdrant
        self.collection = collection
        self.bm25 = bm25_indexer  # 検索時のquery→sparse変換に使う

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
            query: ユーザクエリ（英語想定）
            qvec: OpenAI埋め込みベクトル（text-embedding-3-largeなど）
            top_k: 取得件数
            prefetch_k: fusion前にdense/sparseそれぞれが取得する候補数
        Returns:
            docs: [payload辞書のリスト]
        """

        if self.bm25 is None:
            raise ValueError("bm25_indexer (BM25Sparse) を指定してください。")

        # --- BM25 sparseベクトル作成 ---
        q_idx, q_vals = self.bm25.transform_query(query)

        # --- Qdrant内で dense + sparse をRRF融合 ---
        res = self.qdrant.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=q_idx, values=q_vals),
                    using="text-sparse",
                    limit=prefetch_k,
                ),
                models.Prefetch(
                    query=qvec,
                    using="text-dense",
                    limit=prefetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=top_k,
        )

        docs = [
            {
                "id": p.id,
                "title": p.payload.get("title"),
                "caption": p.payload.get("_caption"),
                "image_url": p.payload.get("_image_url"),
                "combined_text": p.payload.get("_combined_text"),
                "score": p.score,
            }
            for p in res.points
        ]
        return docs
