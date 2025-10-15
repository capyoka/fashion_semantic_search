# Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースをコピー
COPY . .

# 既定値（秘密は置かない）
ENV EMBEDDING_MODEL=text-embedding-3-small \
    ALPHA=0.6 \
    TOP_K=5 \
    DATA_PROFILE=small \ 
    INGEST_JSONL=/app/data/${DATA_PROFILE}/embeddings.jsonl

EXPOSE 8000
# 実行コマンドは compose 側で指定（前処理→API起動）
