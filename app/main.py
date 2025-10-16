from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.search import router as search_router

app = FastAPI(title="Hybrid Fashion Search API")

# CORS（ブラウザからのfetchを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番ではドメイン指定推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ search API ルーターを登録
app.include_router(search_router)

# ✅ static フォルダをマウント（例: ./static/）
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ トップページで簡易UIを返す
@app.get("/")
async def root():
    return {"message": "Hybrid Fashion Search API is running. Go to /static/index.html"}
