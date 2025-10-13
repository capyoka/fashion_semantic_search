from pydantic import BaseModel
from typing import List

class SearchResponse(BaseModel):
    context: List[str]   # 上位ドキュメント本文
    scores: List[float]  # そのスコア
