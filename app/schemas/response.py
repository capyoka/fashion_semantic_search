from pydantic import BaseModel
from typing import Any, List

class SearchResponse(BaseModel):
    results: List[Any]  # 商品データの構造が柔軟なので Any にしておく
