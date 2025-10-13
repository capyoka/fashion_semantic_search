from openai import OpenAI
import os

class Embedder:
    def __init__(self, model_name: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=key)
        self.model = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in res.data]
