from app.pipelines.preprocess_from_csv import run
from app.core.config import settings

if __name__ == "__main__":
    run('data/raw/products.csv', settings.PROCESSED_JSONL, settings.QDRANT_PATH, use_image_caption=False)
    print("Preprocess (CSV) done.")
