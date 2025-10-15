🧠 Semantic Fashion Recommender

自然言語クエリに基づいてファッション商品をレコメンドするマイクロサービスのプロトタイプです。
ユーザーが「夏にビーチに行くときに着たい服が欲しい」と入力すると、
意味的に関連する商品をデータセットから検索して返します。

🚀 概要

このプロジェクトは、ECサイトのファッションカテゴリにおける
「セマンティック検索（意味ベースの推薦）」 の概念実証を目的としています。

従来のキーワード検索（例：「t-shirt」）ではなく、
自然言語文（例：「春のデートに着たいカジュアルな服」）を理解して検索を行います。

🧩 セットアップ手順

## Requirements
- Docker / Docker Compose **または** Python 3.8+
- OpenAI API Key（埋め込みとキャプション生成用）

## 方法1: Docker（推奨・簡単）
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

## 方法2: ローカル開発環境
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

## 全データセットを使用する場合

デフォルトでは `sample/sample_100.json` のサンプルデータが使用されますが、全データセットを使用したい場合は以下の手順を実行してください：

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

> **注意**: 全データセットの処理には時間がかかります（数時間程度）。また、OpenAI APIの使用量が大幅に増加します。

🧑‍💻 実行確認方法

1. `.env` にOpenAI APIキーを設定
2. セットアップ手順を完了
3. `uvicorn app.main:app --reload` でAPI起動
4. `/api/search` エンドポイントでクエリ送信
5. 結果がJSONで出力されれば成功

💡 使用例

## API使用例

### 基本的な検索
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "夏にビーチで着る服",
    "alpha": 0.6,
    "top_k": 5
  }'
```

### レスポンス例
```json
{
  "results": [
    {
      "id": "uuid-string",
      "payload": {
        "title": "Summer Beach Tank Top",
        "store": "Fashion Store",
        "features": ["Lightweight", "Quick-dry", "UV Protection"],
        "_captions": ["A lightweight tank top perfect for beach activities"],
        "_search_text": "Summer Beach Tank Top Lightweight Quick-dry UV Protection A lightweight tank top perfect for beach activities"
      },
      "score": 0.87
    }
  ]
}
```

### パラメータ説明
- `query`: 検索クエリ（自然言語）
- `alpha`: Dense/Sparse融合の重み（0.0-1.0、デフォルト0.6）
- `top_k`: 取得件数（1-50、デフォルト5）

🏗️ システム構成

主な構成要素

データ前処理層

Amazon Fashionデータセット（JSONL）を読み込み
商品画像からLLM（GPT-4o-mini）でキャプション生成
タイトル、特徴、キャプションを統合した検索用テキストを生成

埋め込み生成層（Embeddings）

OpenAI text-embedding-3-smallでベクトル化
BM25スパースベクトルも並行生成

ベクトルデータベース（Qdrant）

HNSWアルゴリズムによる高速ベクトル検索
RRF（Reciprocal Rank Fusion）でDense + Sparse融合
ローカルファイルベースで運用

クエリ処理層

自然言語クエリをLLMで書き換え・キーワード抽出
非同期処理で埋め込み生成とBM25検索を並行実行

結果生成層（LLM統合）

検索結果をLLMでリランキング
自然な文章での結果整形

API層

FastAPIでREST APIエンドポイントを提供
Docker Composeによる開発環境構築

📁 ディレクトリ構成
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
│       ├── embedder.py            # 埋め込み生成
│       ├── generator.py           # LLM生成
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

⚙️ 主な設計上の決定とトレードオフ

| 項目 | 選定理由 / トレードオフ |
|------|------------------------|
| Embeddingsモデル | OpenAIのtext-embedding-3-smallを採用。精度とコストのバランスが良い |
| ベクトルDB | Qdrantを採用。HNSWアルゴリズムで高速検索、RRF融合でDense+Sparse統合 |
| 検索方式 | RRF（Reciprocal Rank Fusion）でDense + BM25 Sparseを融合 |
| APIフレームワーク | FastAPI。軽量かつSwagger UIが自動生成されるためデモに最適 |
| LLM統合 | GPT-4o-miniでクエリ書き換え、キャプション生成、リランキング |
| データ処理 | 画像キャプション生成でマルチモーダル情報を活用 |
| 非同期処理 | 埋め込み生成、BM25検索、LLM処理を並行実行で高速化 |

📊 データ探索

`notebooks/` ディレクトリにて以下の探索を実施：

- `01_sampling.ipynb`: データサンプリングと前処理
- `02_overview.ipynb`: データセット概要分析
- `03_eda.ipynb`: 探索的データ分析
- `04_image_caption.ipynb`: 画像キャプション生成実験
- `05_create_vectordb.ipynb`: ベクトルデータベース構築
- `06_query_preprocess.ipynb`: クエリ前処理実験

🔮 今後の改善案

- 商品画像からのマルチモーダル埋め込み対応（CLIP等）
- ユーザー嗜好履歴を加味したパーソナライズ
- 商品タグ付け自動生成パイプラインの追加
- Embeddingキャッシュ・再利用最適化
- リアルタイム商品更新対応
- 多言語クエリ対応

🧾 ライセンス

MIT License
