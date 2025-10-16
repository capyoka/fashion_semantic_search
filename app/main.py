import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.search import router as search_router

# Logger configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Hybrid Fashion Search API")

# CORS (allow browser fetch requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register search API router
app.include_router(search_router)

# Mount static folder (e.g., ./static/)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Return simple UI on top page
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Hybrid Fashion Search API is running. Go to /static/index.html"}
