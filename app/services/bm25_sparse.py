"""
services/bm25_sparse.py

BM25Sparse — BM25 utility for common use in indexing and search.
- English preprocessing (normalization + tokenization + lemmatization + stopword removal)
- Save/restore IDF dictionary to/from Qdrant collection metadata
"""

import logging
import re
import uuid
import pathlib
from typing import List, Dict
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Logger configuration
logger = logging.getLogger(__name__)

# --- NLTK data path --- 
NLTK_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "nltk_data"
nltk.data.path.append(str(NLTK_PATH))

class BM25Sparse:
    """BM25 Sparse vectorization + Qdrant metadata save support (dummy Point method)"""

    def __init__(self, corpus: List[str] = None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.bm25 = None
        self.vocab: List[str] = []
        self.token_to_index: dict[str, int] = {}

        if corpus:
            self.fit(corpus)

    # —————————————————————
    # Preprocessing / Tokenization
    # —————————————————————
    def _preprocess(self, text: str) -> List[str]:
        """English text normalization + tokenization + lemmatization + stopword removal"""
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()
        tokens = word_tokenize(text)
        clean_tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 1
        ]
        return clean_tokens

    def fit(self, corpus: List[str]):
        """Build BM25 model using entire corpus"""
        logger.info(f"Fitting BM25 model with {len(corpus)} documents")
        tokenized = [self._preprocess(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        # Keep BM25Okapi's IDF with vocabulary
        self.vocab = list(self.bm25.idf.keys())
        self.token_to_index = {t: i for i, t in enumerate(self.vocab)}
        logger.info(f"BM25 model fitted: vocabulary size = {len(self.vocab)}")
        return self

    # —————————————————————
    # Document / Query → Sparse vector conversion
    # —————————————————————
    def transform_doc(self, text: str):
        """Document → BM25 sparse vector (duplicate index integration completed)"""
        tokens = self._preprocess(text)
        scores = {}

        for t in tokens:
            if t in self.token_to_index:
                idx = self.token_to_index[t]
                val = self.bm25.idf.get(t, 0.0)
                scores[idx] = scores.get(idx, 0.0) + val  # Integration here

        indices = list(scores.keys())
        values = list(scores.values())
        return indices, values

    def transform_query(self, text: str):
        """Query → BM25 sparse vector (similarly integrated)"""
        return self.transform_doc(text)


    # —————————————————————
    # Qdrant persistence: Dummy Point save method
    # —————————————————————
    def save_to_qdrant(self, qdrant: QdrantClient, collection_name: str):
        """Save BM25 vocab + IDF to Qdrant (dummy UUID Point)"""
        if self.bm25 is None:
            raise ValueError("BM25 model is not fitted; cannot save metadata.")

        payload = {
            "_bm25_meta": True,
            "vocab": self.vocab,
            "idf": self.bm25.idf,
        }

        # Use fixed UUID format ID (stable and valid UUID)
        bm25_meta_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "bm25_meta"))

        # Add dummy zero vector (avoid pydantic ValidationError)
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
        logger.info(f"Saved BM25 metadata point as UUID={bm25_meta_id} (vocab size={len(self.vocab)})")

    @classmethod
    def load_from_qdrant(cls, qdrant: QdrantClient, collection_name: str):
            """Load dummy Point from Qdrant and reconstruct BM25 model"""
            # Use same stable UUID as when saving
            bm25_meta_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "bm25_meta"))

            pts = qdrant.retrieve(collection_name=collection_name, ids=[bm25_meta_id])
            if not pts or not pts[0].payload.get("_bm25_meta"):
                logger.error(f"BM25 metadata not found in collection '{collection_name}'")
                raise RuntimeError(f"BM25 metadata not found in collection '{collection_name}'")

            meta = pts[0].payload

            inst = cls()
            inst.vocab = meta["vocab"]
            inst.token_to_index = {t: i for i, t in enumerate(inst.vocab)}

            # Create empty BM25 and overwrite IDF
            inst.bm25 = BM25Okapi([["dummy"]])
            inst.bm25.idf = meta["idf"]

            logger.info(f"Loaded BM25 metadata from dummy point (UUID={bm25_meta_id}); vocab size={len(inst.vocab)}")
            return inst