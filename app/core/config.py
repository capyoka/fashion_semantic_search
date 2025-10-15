import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/vectorstore/qdrant")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fashion_collection")
    PROCESSED_JSONL = os.getenv("PROCESSED_JSONL", "./data/processed/processed_texts.jsonl")
    API_TITLE = os.getenv("API_TITLE", "Local RAG API")

settings = Settings()
