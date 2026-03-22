from typing import Optional
from pydantic import BaseModel


class Deal(BaseModel):
    deal_id: str
    title: str
    description: str
    image: str
    link: str
    deal_price: float
    old_price: float
    currency: str
    savings_pct: float
    rating: float
    ratings_total: int
    deal_type: str
    is_prime: bool
    is_lightning: bool


class SearchRequest(BaseModel):
    query: str
    max_price: Optional[float] = None
    min_discount_pct: Optional[float] = None
    top_k: int = 12


class SearchResponse(BaseModel):
    query: str
    ai_summary: str
    deals: list[Deal]
    total_found: int


class RefreshResponse(BaseModel):
    deals_loaded: int
    message: str
