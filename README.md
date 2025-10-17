# Semantic Fashion Recommender

## Overview

This project demonstrates the concept of  
**semantic search–based recommendation** within the fashion category of an e-commerce site.

Instead of traditional keyword searches (e.g., “t-shirt”),  
the system understands and processes natural language queries such as  
“casual clothes for a spring date” based on their semantic meaning.

## Setup Guide

### Requirements
- Docker / Docker Compose **or** Python 3.8+
- OpenAI API Key (for embeddings and caption generation)

### Method 1: Docker (Recommended and Easiest)
```bash
# 1. Move to the project directory
cd semantic_fashion_search

# 2. Set environment variables
cp .env.sample .env
# Set your OPENAI_API_KEY in the .env file

# 3. Build and launch with Docker Compose
docker compose up --build
# → API: http://localhost:8000/docs
```

### Method 2: Local Development
```bash
# 1. Move to the project directory
cd semantic_fashion_search

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.sample .env  # Configure your OPENAI_API_KEY

# 5. Start the API server
uvicorn app.main:app --reload
# → Access Swagger UI at http://localhost:8000/docs
```

> **Note:** Use **either** Method 1 (Docker) **or** Method 2 (Local). Do not mix both.

#### Using the Full Dataset

The provided zip archive already includes a 1,000-item sample dataset
(`sample/sample_1000.json`).  
To use the full dataset, follow these steps:

```bash
# 1. Place the full dataset
# Put meta_Amazon_Fashion.jsonl into data/raw/

# 2. Update environment variables
# In the .env file, update the following path:
# PROCESSED_JSONL=./data/raw/meta_Amazon_Fashion.jsonl

# 3. Run preprocessing
python -m scripts.run_preprocess

# 4. Restart the API server
uvicorn app.main:app --reload
```

### Verification

#### 1. Test via Sample UI
After starting the API server, open the following URL to try the prototype app:  
http://127.0.0.1:8000/static/index.html

#### 2. Test via API
```bash
curl -X POST "http://localhost:8000/search"   -H "Content-Type: application/json"   -d '{
    "query": "clothes to wear at the beach in summer",
    "top_k": 30
  }'
```

#### Example
**input:**  
- query: "clothes to wear at the beach in summer"
- top_k: 5

**output:**  
```
{
  "results": [
    {
      "id": "148ebebd-6a28-59f5-993c-4a7e154cc966",
      "score": 0.2,
      "main_category": "AMAZON FASHION",
      "title": "NIMIN High Waisted Shorts for Women Comfy Drawstring Casual Elastic Shorts Summer Beach Lightweight Short Pants with Pockets Blue Medium",
      "average_rating": 3.3,
      "rating_number": 8,
      "features": [
        "Machine Wash"
      ],
      "description": [],
      "price": null,
      "images": [
        {
          "thumb": "https://m.media-amazon.com/images/I/41IUYkIZbSS._AC_SR38,50_.jpg",
          "large": "https://m.media-amazon.com/images/I/41IUYkIZbSS._AC_.jpg",
          "variant": "MAIN",
          "hi_res": "https://m.media-amazon.com/images/I/61Xb8dissbS._AC_UL1280_.jpg"
        },
        {
          "thumb": "https://m.media-amazon.com/images/I/41nBfZNh8qS._AC_SR38,50_.jpg",
          "large": "https://m.media-amazon.com/images/I/41nBfZNh8qS._AC_.jpg",
          "variant": "PT01",
          "hi_res": "https://m.media-amazon.com/images/I/6190XTR+XeS._AC_UL1280_.jpg"
        },
        {
          "thumb": "https://m.media-amazon.com/images/I/410JpZvENPS._AC_SR38,50_.jpg",
          "large": "https://m.media-amazon.com/images/I/410JpZvENPS._AC_.jpg",
          "variant": "PT02",
          "hi_res": "https://m.media-amazon.com/images/I/61oOhxdhe-S._AC_UL1280_.jpg"
        },
        {
          "thumb": "https://m.media-amazon.com/images/I/41RaPIkRLwS._AC_SR38,50_.jpg",
          "large": "https://m.media-amazon.com/images/I/41RaPIkRLwS._AC_.jpg",
          "variant": "PT03",
          "hi_res": "https://m.media-amazon.com/images/I/71HN9CKUEhS._AC_UL1280_.jpg"
        },
        {
          "thumb": "https://m.media-amazon.com/images/I/41ocnctBSkS._AC_SR38,50_.jpg",
          "large": "https://m.media-amazon.com/images/I/41ocnctBSkS._AC_.jpg",
          "variant": "PT04",
          "hi_res": "https://m.media-amazon.com/images/I/61gA6a7EurS._AC_UL1280_.jpg"
        },
        {
          "thumb": "https://m.media-amazon.com/images/I/41HyeDPK17S._AC_SR38,50_.jpg",
          "large": "https://m.media-amazon.com/images/I/41HyeDPK17S._AC_.jpg",
          "variant": "PT05",
          "hi_res": "https://m.media-amazon.com/images/I/61SIMdDHsOS._AC_UL1500_.jpg"
        }
      ],
      "videos": [
        {
          "title": "NIMIN High Waisted Linen Shorts for Women",
          "url": "https://www.amazon.com/vdp/0afe26a4b4b348e1b64590409581185a?ref=dp_vse_rvc_0",
          "user_id": ""
        }
      ],
      "store": "NIMIN",
      "categories": [],
      "details": {
        "Package Dimensions": "11.57 x 8.15 x 1.18 inches; 6.74 Ounces",
        "Item model number": "NMKZ125-BUM",
        "Date First Available": "May 7, 2021"
      },
      "parent_asin": "B093Q3DG8D",
      "bought_together": null
    },
    {
      ...
    },
    ...
  ]
}
```

## System Architecture
See `data/design/high_level_design.png` for the architecture diagram.

### Directory Structure
```
rag_app/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── api/
│   │   └── search.py              # Search API endpoint
│   ├── core/
│   │   └── config.py              # Configuration management
│   ├── pipelines/
│   │   └── preprocess.py          # Data preprocessing pipeline
│   ├── schemas/
│   │   ├── request.py             # Request schema
│   │   └── response.py            # Response schema
│   └── services/
│       ├── bm25_sparse.py         # BM25 class
│       ├── retriever.py           # Hybrid search logic
│       ├── aembedder.py           # Asynchronous embedding
│       └── agenerator.py          # Asynchronous LLM generation
├── data/
│   ├── raw/                       # Raw data
│   ├── sample/                    # Sample data
│   └── vectorstore/
│       └── qdrant/                # Qdrant vector database
├── static/
│   └── index.html/                # Prototype front end
├── notebooks/                     # EDA and experimental notebooks
├── scripts/
│   └── run_preprocess.py          # Preprocessing execution script
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Docker image definition
├── requirements.txt               # Python dependencies
└── README.md
```

The `notebooks/` directory contains Jupyter notebooks used for EDA,
data sampling, and dictionary downloads:

- **01_sampling.ipynb:** Data sampling  
- **02_overview.ipynb:** Dataset overview analysis  
- **03_eda.ipynb:** Exploratory data analysis  
- **04_image_caption.ipynb:** Image caption generation and prompt experiments  
- **05_nltk_download.ipynb:** Dictionary setup for query preprocessing  

## Key Design Decisions and Trade-offs

### 1. Model Selection
**Decision:**  
Models were selected based on the complexity of each processing stage:

- Query transformation: `gpt-5-nano` (reasoning_effort=minimal, verbosity=low)  
- Embedding generation: `text-embedding-3-large`  
- Reranking: `gpt-5` (reasoning_effort=minimal, verbosity=low)

**Trade-off:**  
Using smaller models (nano/mini) improves response time but can slightly
reduce accuracy.  
This system prioritizes **semantic precision and relevance** over minimal
latency, adopting `gpt-5` and large models for higher-quality results.

---

### 2. Caption Generation
**Decision:**  
The caption generation prompt intentionally allows a moderate degree of
interpretation and uses `gpt-5` for high-quality, context-rich outputs
(reasoning_effort=minimal, verbosity=low).

**Rationale:**  
The dataset mainly consists of short titles or phrases that lack detail
about product features, purpose, or context.  
By generating captions from both images and titles, the system supplements
missing semantics, enabling retrieval that aligns with
**purpose- or situation-based queries**.  
Accordingly, the prompt design allows controlled interpretive flexibility
to enhance result relevance.

**Trade-off:**  
Allowing interpretive captions increases recall at the expense of some
precision.  
This design prioritizes **including the items users truly want**, even if a
few irrelevant ones appear.

---

### 3. Search Strategy
**Decision:**  
Adopted a **hybrid search** combining BM25 (keyword-based) and vector search,
followed by reranking with an LLM.

**Rationale:**  
Users typically submit short, keyword-driven queries like  
“clothes for the beach” or “tops from brand X.”  
Pure vector search may surface semantically similar but contextually
irrelevant items, such as:

Query: “clothes for the beach”  
Vector search results:  
- “Ocean Wave Graphic T-Shirt”  
- “Blue Hoodie”

Although semantically related to words like “ocean” or “blue,” these do not
reflect the intended context (e.g., lightweight, summer, resort-style).  
Therefore, BM25 complements keyword matching, and the LLM reranker
re-evaluates contextual coherence between the query and captions.

**Trade-off:**  
Combining hybrid search with reranking improves accuracy but increases
processing time.  
This system values **relevance and user satisfaction** over slight
performance overhead.

---

## Challenges and Future Improvements

### Improving Search Accuracy Using User Attributes and Metadata
**Challenge:**  
Relying solely on the query can lead to mismatches, such as returning
products for different genders or age groups.

**Solution:**  
Use LLMs to infer metadata such as gender, target audience, and category
from titles and descriptions.  
Then, apply pre-filtering based on user profiles or past search history to
refine results.

---

### Leveraging Reviews for Popularity-Based Recommendations
**Challenge:**  
Similarity-based ranking alone may prioritize low-quality or less-reviewed
products.

**Solution:**  
Incorporate review averages and counts to calculate a **popularity score**,
used during reranking to prioritize reliable, well-rated items.

---

### Establishing Evaluation and Continuous Improvement Mechanisms
**Challenge:**  
There is currently no quantitative framework to evaluate search quality.

**Solution:**  
Create an evaluation dataset pairing representative queries with correct
product matches.  
Use ranking metrics such as Recall@K and NDCG for quantitative assessment.  
If labeled data is unavailable, implement user feedback loops or conduct UAT
sessions with business teams to drive iterative improvement.
