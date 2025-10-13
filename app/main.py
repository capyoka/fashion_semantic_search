from fastapi import FastAPI
from app.api.search import router as search_router
from app.core.config import settings

app = FastAPI(title=settings.API_TITLE)
app.include_router(search_router, prefix="/api")

@app.get("/")
def health():
    return {"status": "ok"}
