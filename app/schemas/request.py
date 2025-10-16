from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(30, ge=1, le=50)
