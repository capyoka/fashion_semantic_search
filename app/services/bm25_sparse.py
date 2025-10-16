"""
services/bm25_sparse.py

BM25Sparse — インデクサと検索で共通利用できるBM25ユーティリティ。
・英語前処理（正規化 + トークン化 + 原形化 + stopword除去）
・idf辞書をQdrantのcollection metadataに保存/復元
"""

import re
from typing import List, Dict
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
import uuid

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pathlib

# --- NLTK データパス --- 
NLTK_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "nltk_data"
nltk.data.path.append(str(NLTK_PATH))

class BM25Sparse:
    """BM25 Sparseベクトル化 + Qdrant メタ保存対応（ダミー Point 方式）"""

    def __init__(self, corpus: List[str] = None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.bm25 = None
        self.vocab: List[str] = []
        self.token_to_index: dict[str, int] = {}

        if corpus:
            self.fit(corpus)

    # —————————————————————
    # 前処理／トークナイズ
    # —————————————————————
    def _preprocess(self, text: str) -> List[str]:
        """英語テキストの正規化 + トークン化 + 原形化 + stopword 除去"""
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()
        tokens = word_tokenize(text)
        clean_tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 1
        ]
        return clean_tokens

    def fit(self, corpus: List[str]):
        """コーパス全体を使って BM25 モデルを構築"""
        tokenized = [self._preprocess(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        # BM25Okapi の idf を語彙とともに保持
        self.vocab = list(self.bm25.idf.keys())
        self.token_to_index = {t: i for i, t in enumerate(self.vocab)}
        return self

    # —————————————————————
    # 文書／クエリ → 疎ベクトル変換
    # —————————————————————
    def transform_doc(self, text: str):
        """文書→BM25疎ベクトル（重複インデックス統合済み）"""
        tokens = self._preprocess(text)
        scores = {}

        for t in tokens:
            if t in self.token_to_index:
                idx = self.token_to_index[t]
                val = self.bm25.idf.get(t, 0.0)
                scores[idx] = scores.get(idx, 0.0) + val  # ← ここで統合

        indices = list(scores.keys())
        values = list(scores.values())
        return indices, values

    def transform_query(self, text: str):
        """クエリ→BM25疎ベクトル（同様に統合）"""
        return self.transform_doc(text)


    # —————————————————————
    # Qdrant 永続化：ダミー Point 保存方式
    # —————————————————————
    def save_to_qdrant(self, qdrant: QdrantClient, collection_name: str):
        """BM25 の vocab + idf を Qdrant に保存（ダミー UUID Point）"""
        if self.bm25 is None:
            raise ValueError("BM25 model is not fitted; cannot save metadata.")

        payload = {
            "_bm25_meta": True,
            "vocab": self.vocab,
            "idf": self.bm25.idf,
        }

        # ✅ UUID形式の固定IDを使用（安定的で有効なUUID）
        bm25_meta_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "bm25_meta"))

        # ✅ ダミーゼロベクトルを追加（pydantic ValidationError 回避）
        dummy_vector = {"text-dense": [0.0] * 3072}

        qdrant.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=bm25_meta_id,
                    vector=dummy_vector,
                    payload=payload,
                )
            ],
        )
        print(f"[BM25] Saved metadata point as UUID={bm25_meta_id} (vocab size={len(self.vocab)})")

    @classmethod
    def load_from_qdrant(cls, qdrant: QdrantClient, collection_name: str):
            """Qdrant からダミー Point を読み込み、BM25 モデルを再構築"""
            # ✅ 保存時と同じ安定UUIDを使用
            bm25_meta_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "bm25_meta"))

            pts = qdrant.retrieve(collection_name=collection_name, ids=[bm25_meta_id])
            if not pts or not pts[0].payload.get("_bm25_meta"):
                raise RuntimeError(f"[BM25] metadata not found in collection '{collection_name}'")

            meta = pts[0].payload

            inst = cls()
            inst.vocab = meta["vocab"]
            inst.token_to_index = {t: i for i, t in enumerate(inst.vocab)}

            # ✅ 空BM25を作ってidf上書き
            inst.bm25 = BM25Okapi([["dummy"]])
            inst.bm25.idf = meta["idf"]

            print(f"[BM25] Loaded metadata from dummy point (UUID={bm25_meta_id}); vocab size={len(inst.vocab)}")
            return inst