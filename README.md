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
cd rag_app

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
cd rag_app

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
curl -X POST "http://localhost:8000/api/search"   -H "Content-Type: application/json"   -d '{
    "query": "clothes to wear at the beach in summer",
    "top_k": 30
  }'
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

--- 

# Semantic Fashion Recommender

## 概要

このプロジェクトは、ECサイトのファッションカテゴリにおける
「セマンティック検索（意味ベースの推薦）」 の概念実証を目的としています。

従来のキーワード検索（例：「t-shirt」）ではなく、
自然言語文（例：「春のデートに着たいカジュアルな服」）を理解して検索を行います。

## セットアップ手順

### Requirements
- Docker / Docker Compose **または** Python 3.8+
- OpenAI API Key（埋め込みとキャプション生成用）

### 方法1: Docker（推奨・簡単）
```bash
# 1. プロジェクトディレクトリに移動
cd rag_app

# 2. 環境変数設定
cp .env.sample .env
# .envファイルでOPENAI_API_KEYを設定

# 3. Docker Composeで起動
docker compose up --build
# → API: http://localhost:8000/docs
```

### 方法2: ローカル開発環境
```bash
# 1. プロジェクトディレクトリに移動
cd rag_app

# 2. 仮想環境作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 依存関係インストール
pip install -r requirements.txt

# 4. 環境変数設定
cp .env.sample .env  # OPENAI_API_KEYを設定

# 5. APIサーバ起動
uvicorn app.main:app --reload
# → http://localhost:8000/docs でSwagger UI
```

> **注意**: 上記の方法1（Docker）と方法2（ローカル）の**どちらか一方**を選択して実行してください。

#### 全データセットを使用する場合

zipファイルにはすでに1000件分のサンプルデータが含まれています。( `sample/sample_1000.json`)  
全データセットを使用したい場合は以下の手順を実行してください：

```bash
# 1. 全データセットを配置
# meta_Amazon_Fashion.jsonl を data/raw/ ディレクトリに配置

# 2. 環境変数を更新
# .envファイルで以下のパスを変更：
# PROCESSED_JSONL=./data/raw/meta_Amazon_Fashion.jsonl

# 3. データ前処理を実行
python -m scripts.run_preprocess

# 4. APIサーバを再起動
uvicorn app.main:app --reload
```

### 動作確認方法

#### 1. サンプルUIでのテスト
APIサーバー起動後、以下のURLにアクセスすることでプロトアプリを利用可能です。
http://127.0.0.1:8000/static/index.html

#### 2. APIでのテスト
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "夏にビーチで着る服",
    "top_k": 30
  }'
```

## システム構成
data/design/high_level_design.pngを参照してください

### ディレクトリ構成
```
rag_app/
├── app/
│   ├── main.py                    # FastAPIエントリポイント
│   ├── api/
│   │   └── search.py              # 検索APIエンドポイント
│   ├── core/
│   │   └── config.py              # 設定管理
│   ├── pipelines/
│   │   └── preprocess.py          # データ前処理パイプライン
│   ├── schemas/
│   │   ├── request.py             # リクエストスキーマ
│   │   └── response.py            # レスポンススキーマ
│   └── services/
│       ├── bm25_sparse            # BM25検索
│       ├── retriever.py           # ハイブリッド検索
│       ├── aembedder.py           # 非同期埋め込み
│       └── agenerator.py          # 非同期LLM生成
├── data/
│   ├── raw/                       # 生データ
│   ├── sample/                    # サンプルデータ
│   └── vectorstore/
│       └── qdrant/                # Qdrantデータベース
├── notebooks/                     # データ探索・実験
├── scripts/
│   └── run_preprocess.py          # 前処理実行スクリプト
├── docker-compose.yml             # Docker Compose設定
├── Dockerfile                     # Dockerイメージ定義
├── requirements.txt               # Python依存関係
└── README.md
```

notebooksには、EDA、データサンプリング、辞書ダウントードなどに利用したnotebookが格納されています。
- 01_sampling.ipynb: データサンプリング
- 02_overview.ipynb: データセット概要分析
- 03_eda.ipynb: EDA用
- 04_image_caption.ipynb: 画像キャプション生成、プロンプト実験
- 05_nltk_download.ipynb: クエリ前処理のための辞書取得

## 主な設計上の決定とトレードオフ
### 1. モデル選定（Model Selection）
**判断:**  
処理内容の難易度に応じて最適なモデルを選定した。
- クエリ変換：gpt-5-nano（reasoning_effort=minimal, verbosity=low）
- ベクトル化：text-embedding-3-large
- リランキング：gpt-5（reasoning_effort=minimal, verbosity=low）

**トレードオフ:**  
nanoやmini モデルを使用することでレスポンス速度は向上するが、検索精度の低下が懸念される。
本システムでは、数秒の応答遅延よりも「通常のキーワード検索より精度の高い結果を返すこと」を優先し、gpt-5 および large モデルを採用した。

### 2. キャプション生成（Caption Generation）
**判断:**  
キャプション生成では、一定の“解釈”を許容するプロンプト設計を採用し、高精度を期待して gpt-5 を使用した（reasoning_effort=minimal, verbosity=low）。

**根拠:**  
本データセットには、タイトルや短いフレーズのみで構成され、商品の特徴・用途・シーンを十分に表現できないデータが多く含まれている。
そのため、画像とタイトルをもとに補完的なキャプションを生成し、「目的・シチュエーションベースのクエリ」に対応できるようにすることが、検索品質の向上に不可欠と判断し、目的やシチュエーションを生成できるよう、ある程度解釈を含むプロンプトを採用した。

**トレードオフ:**  
解釈を許容するプロンプトは、Precision と Recall のトレードオフを伴う。
解釈を広げることでユーザーの目的に合った商品を網羅的に拾いやすくなる一方、誤った推論を行うリスクも増す。
本システムでは、「多少の誤りを許容しても、本当に欲しい商品が検索結果に含まれること」を重視し、解釈を許容する設計を採用した。

### 3. 検索方式（Search Strategy）

**判断:**  
BM25 とベクトル検索を組み合わせた ハイブリッド検索 を採用し、最終段階で LLM によるリランキングを行った。

**根拠:**
ユーザーは「海に行くときの服」や「○○ブランドの服」など、短文かつキーワード主導の検索を行うことが多いと想定される。
またベクトル検索のみでは「意味的には近いが意図と異なる」結果が返りやすく、例えば以下のような問題が起こり得る。

クエリ：「海に行くときの服」
ベクトル検索結果：
“Ocean Wave Graphic T-Shirt”
“Blue Hoodie”

これらは “ocean” や “blue” といった語に近いEmbeddingを持つが、ユーザーが求める「夏向け・軽装・リゾート感のある服」という文脈的意図を反映していない。
そのため、BM25 でキーワード一致を補完し、さらに LLM のリランク処理によってクエリとキャプションの文脈整合性を再評価する構成を採用した。

**トレードオフ:**  
ハイブリッド検索とリランクを組み合わせることで検索精度は向上するが、処理時間が増加する。
ただし本システムでは、多少の遅延よりも「ユーザーが本当に求めている商品を上位に表示すること」を優先した。


## 課題と改善案
### ユーザー属性とメタデータを活用した検索精度向上
**課題:**  
クエリのみを入力としているため、性別・対象年齢の異なる商品が混在し、意図と異なる検索結果が返る。

**対策:**  
商品タイトルや説明文をLLMで解析し、「性別・対象・カテゴリ」などのメタデータを自動生成。  
またユーザー属性や過去の検索履歴を活用して事前フィルタリングを行い、より的確な結果を返す。

### レビュー情報を活用した人気商品の推薦
**課題:**  
類似度のみの評価では、品質の低い商品やレビュー数が少ない商品が上位に出る場合がある。

**対策:**  
レビュー平均スコアや件数をもとに人気度スコアを算出し、検索結果の再ランキング時に加重評価して、信頼性の高い商品を優先的に表示する。

### 精度評価と継続的改善の仕組み構築
**課題:**  
検索結果の品質を客観的に評価する仕組みがなく、改善効果を定量的に把握できない。

**対策:**  
代表的なクエリと正解商品をセットにした評価データセットを作成し、ランキング指標（Recall@K, NDCGなど）で定量評価を実施。データ整備が難しい場合は、フィードバック機能を実装したり、業務部門を巻き込んだUATを行い評価・改善サイクルを確立する。
