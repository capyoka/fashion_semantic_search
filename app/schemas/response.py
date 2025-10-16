from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ProductItem(BaseModel):
    id: str
    score: float

    main_category: Optional[str]
    title: Optional[str]
    average_rating: Optional[float]
    rating_number: Optional[int]
    features: List[str] = []
    description: List[str] = []
    price: Optional[float] = None

    # 👇 ImageItem を Any に変更（構造に柔軟性を持たせる）
    images: List[Any] = []
    videos: List[Any] = []

    store: Optional[str] = None
    categories: List[str] = []
    details: Optional[Dict[str, Any]] = None

    parent_asin: Optional[str] = None
    bought_together: Optional[Any] = None

    _id: Optional[str] = None
    _captions: List[str] = []
    _search_text: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[ProductItem]
