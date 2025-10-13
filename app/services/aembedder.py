from typing import List
import os
from openai import AsyncOpenAI

class AsyncEmbedder:
    def __init__(self, model_name: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = AsyncOpenAI(api_key=key)
        self.model = model_name

    async def encode(self, texts: List[str]) -> List[List[float]]:
        res = await self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in res.data]
