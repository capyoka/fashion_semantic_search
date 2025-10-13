from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str
    alpha: float = Field(0.6, ge=0.0, le=1.0)
    top_k: int = Field(5, ge=1, le=50)
