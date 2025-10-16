import logging
from typing import List
import os
from openai import AsyncOpenAI

# Logger configuration
logger = logging.getLogger(__name__)

class AsyncEmbedder:
    def __init__(self, model_name: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = AsyncOpenAI(api_key=key)
        self.model = model_name
        logger.info(f"AsyncEmbedder initialized with model: {model_name}")

    async def encode(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"🧠 Starting embedding generation")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   Text count: {len(texts)}")
        
        # Detailed text logging (debug level)
        if logger.isEnabledFor(logging.DEBUG):
            for i, text in enumerate(texts[:3]):  # First 3 only
                logger.debug(f"Text {i+1}: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        try:
            res = await self.client.embeddings.create(model=self.model, input=texts)
            vectors = [d.embedding for d in res.data]
            
            logger.info(f"✅ Embedding successful: {len(vectors)} vectors generated")
            logger.debug(f"Vector dimension: {len(vectors[0]) if vectors else 'N/A'}")
            
            # Vector statistics (debug level)
            if logger.isEnabledFor(logging.DEBUG) and vectors:
                import numpy as np
                all_values = np.concatenate(vectors)
                logger.debug(f"Vector stats - Mean: {np.mean(all_values):.4f}, Std: {np.std(all_values):.4f}")
                logger.debug(f"Vector range: {np.min(all_values):.4f} - {np.max(all_values):.4f}")
            
            return vectors
        except Exception as e:
            logger.error(f"❌ Embedding failed: {e}")
            logger.error(f"   Model: {self.model}")
            logger.error(f"   Text count: {len(texts)}")
            logger.error(f"   First text: '{texts[0][:100] if texts else 'N/A'}...'")
            raise
