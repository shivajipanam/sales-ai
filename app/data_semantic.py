"""
Deals data layer — Semantic Search version (demonstration only).

Uses sentence-transformers (all-MiniLM-L6-v2) + numpy cosine similarity
for true semantic search. Understands synonyms and intent, e.g.:
  "cheap headphones" matches "affordable earbuds"

NOT used in production deployment (image size ~7.9 GB due to PyTorch).
The production app uses data.py (TF-IDF, image ~400 MB).

To run locally with this version:
  pip install sentence-transformers
  # then swap the import in server.py:
  #   from app.data_semantic import load_all_deals, search_deals, ...
"""
import ast
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.models import Deal

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DATA_DIR = Path(__file__).parent.parent / "examples" / "data"

# In-memory store
_deals: list[Deal] = []
_embeddings: Optional[np.ndarray] = None  # shape (N, 384)
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBEDDING_MODEL)
    return _model


def _deal_text(d: Deal) -> str:
    """Text representation used for semantic embedding."""
    return f"{d.title}. {d.description}. Price: ${d.deal_price:.2f}. Save {d.savings_pct:.0f}%."


def _parse_jsonl_record(raw: str) -> Optional[dict]:
    try:
        obj = json.loads(raw)
        doc_str = obj.get("doc", "")
        try:
            return json.loads(doc_str)
        except (json.JSONDecodeError, TypeError):
            return ast.literal_eval(doc_str)
    except Exception:
        return None


def _record_to_deal(r: dict) -> Optional[Deal]:
    try:
        list_price = r.get("list_price")
        if isinstance(list_price, dict):
            list_price = list_price.get("value", 0)
        old_price = float(r.get("old_price") or list_price or 0)

        deal_price = float(r.get("deal_price") or 0)
        if deal_price <= 0:
            return None

        savings_pct = ((old_price - deal_price) / old_price * 100) if old_price > deal_price > 0 else 0.0

        return Deal(
            deal_id=str(r.get("deal_id") or r.get("asin") or ""),
            title=str(r.get("title") or r.get("name") or "Unknown"),
            description=str(r.get("description") or r.get("title") or ""),
            image=str(r.get("image") or ""),
            link=str(r.get("link") or ""),
            deal_price=deal_price,
            old_price=old_price,
            currency=str(r.get("currency") or "USD"),
            savings_pct=round(savings_pct, 1),
            rating=float(r.get("rating") or 0),
            ratings_total=int(r.get("ratings_total") or 0),
            deal_type=str(r.get("deal_type") or "DEAL"),
            is_prime=bool(r.get("is_prime", False)),
            is_lightning=bool(r.get("is_lightning_deal", False)),
        )
    except Exception:
        return None


def load_from_jsonl(path: Path) -> list[Deal]:
    deals = []
    if not path.exists():
        return deals
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = _parse_jsonl_record(line)
            if record:
                deal = _record_to_deal(record)
                if deal:
                    deals.append(deal)
    return deals


def build_index(deals: list[Deal]) -> np.ndarray:
    """Encode all deals into dense embedding vectors using sentence-transformers."""
    model = _get_model()
    texts = [_deal_text(d) for d in deals]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def load_all_deals() -> int:
    """Load all deals from seed JSONL files and build semantic embedding index."""
    global _deals, _embeddings

    all_deals: list[Deal] = []
    for fname in ["rainforest_discounts.jsonl", "csv_discounts.jsonl"]:
        all_deals.extend(load_from_jsonl(_DATA_DIR / fname))

    seen = set()
    unique = []
    for d in all_deals:
        if d.deal_id and d.deal_id not in seen:
            seen.add(d.deal_id)
            unique.append(d)
        elif not d.deal_id:
            unique.append(d)

    _deals = unique
    if _deals:
        _embeddings = build_index(_deals)

    return len(_deals)


def refresh_from_rainforest() -> int:
    api_key = os.getenv("RAINFOREST_API_KEY")
    if not api_key:
        raise ValueError("RAINFOREST_API_KEY not set.")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from examples.rainforest.rainforestapi_helper import send_request
    send_request(str(_DATA_DIR), {})
    return load_all_deals()


def search_deals(
    query: str,
    max_price: Optional[float] = None,
    min_discount_pct: Optional[float] = None,
    top_k: int = 12,
) -> list[Deal]:
    """
    Semantic search: embed the query and compute cosine similarity
    against all deal embeddings. Understands synonyms and intent.
    """
    if not _deals or _embeddings is None:
        return []

    model = _get_model()
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = _embeddings @ q_vec  # cosine similarity (both normalised)

    filtered_scores = scores.copy()
    for i, deal in enumerate(_deals):
        if max_price is not None and deal.deal_price > max_price:
            filtered_scores[i] = -1
        if min_discount_pct is not None and deal.savings_pct < min_discount_pct:
            filtered_scores[i] = -1

    top_indices = np.argsort(filtered_scores)[::-1][:top_k]
    return [_deals[i] for i in top_indices if filtered_scores[i] > -1]


def get_all_deals() -> list[Deal]:
    return _deals


def deals_count() -> int:
    return len(_deals)
