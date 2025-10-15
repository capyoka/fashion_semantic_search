# Local RAG API (FastAPI + Qdrant + BM25 + OpenAI)

自然言語のファッション検索用ミニサービス。  
クエリを埋め込み化して **Qdrant(HNSW)** でベクトル検索、**BM25** と重み付きでスコア融合します。

## Requirements
- Docker / Docker Compose
- （任意）OpenAI API Key（埋め込みを使う場合）

## Quick Start (Docker)
```bash
cp .env.sample .env             # OPENAI_API_KEY を自分の値に（未設定ならBM25のみで動作）
docker compose up --build
# → API: http://localhost:8000/docs


## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env  # set OPENAI_API_KEY
```

## Preprocess (CSV -> clean -> embed -> Qdrant)
```bash
python -m scripts.run_preprocess
```

## API
```bash
uvicorn app.main:app --reload
# POST /api/search {"query":"夏にビーチで着る服","alpha":0.6,"top_k":5}
```
