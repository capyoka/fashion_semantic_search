from app.pipelines.preprocess import run
from app.core.config import settings

if __name__ == "__main__":
    run(
        input_path=settings.PROCESSED_JSONL,
        qdrant_path=settings.QDRANT_PATH,
    )
