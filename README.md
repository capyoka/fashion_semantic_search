# Local RAG API (FastAPI + Qdrant + BM25 + OpenAI)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env  # set OPENAI_API_KEY
```

## Preprocess (CSV -> clean -> embed -> Qdrant)
```bash
python -m scripts.run_preprocess_csv
```

## API
```bash
uvicorn app.main:app --reload
# POST /api/search {"query":"夏にビーチで着る服","alpha":0.6,"top_k":5}
```
