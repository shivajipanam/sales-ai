"""
Deals data layer.
- Loads deals from existing JSONL files (seed data — real Amazon deals)
- Optionally refreshes from Rainforest API if RAINFOREST_API_KEY is set
- Builds in-memory TF-IDF index with scikit-learn (lightweight, no PyTorch needed)
"""
import ast
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Deal

_DATA_DIR = Path(__file__).parent.parent / "examples" / "data"

# In-memory store
_deals: list[Deal] = []
_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix = None  # sparse matrix (N, vocab)


def _deal_text(d: Deal) -> str:
    """Text representation used for TF-IDF indexing."""
    return f"{d.title}. {d.description}. Price: ${d.deal_price:.2f}. Save {d.savings_pct:.0f}%."


def _parse_jsonl_record(raw: str) -> Optional[dict]:
    """
    Parse a single JSONL line from either the rainforest or csv data files.
    The 'doc' field contains a Python-dict-like string (single quotes) — use ast.literal_eval.
    """
    try:
        obj = json.loads(raw)
        doc_str = obj.get("doc", "")
        # Try JSON first, fall back to ast for Python-repr dicts
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
            return None  # skip free/invalid items

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


def build_index(deals: list[Deal]) -> None:
    global _vectorizer, _tfidf_matrix
    texts = [_deal_text(d) for d in deals]
    _vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    _tfidf_matrix = _vectorizer.fit_transform(texts)


def load_all_deals() -> int:
    """Load all deals from seed JSONL files and build TF-IDF index."""
    global _deals

    all_deals: list[Deal] = []
    for fname in ["rainforest_discounts.jsonl", "csv_discounts.jsonl", "mock_discounts.jsonl"]:
        all_deals.extend(load_from_jsonl(_DATA_DIR / fname))

    # Deduplicate by deal_id
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
        build_index(_deals)

    return len(_deals)


def refresh_from_rainforest() -> int:
    """Fetch fresh deals from Rainforest API (requires RAINFOREST_API_KEY env var)."""
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
    TF-IDF cosine similarity search against deal index,
    then apply price/discount filters. Returns top_k results.
    """
    if not _deals or _vectorizer is None:
        return []

    q_vec = _vectorizer.transform([query])
    scores = cosine_similarity(q_vec, _tfidf_matrix).flatten()

    # Apply filters (set score to -1 to exclude)
    filtered_scores = scores.copy()
    for i, deal in enumerate(_deals):
        if max_price is not None and deal.deal_price > max_price:
            filtered_scores[i] = -1
        if min_discount_pct is not None and deal.savings_pct < min_discount_pct:
            filtered_scores[i] = -1

    top_indices = np.argsort(filtered_scores)[::-1][:top_k]
    return [_deals[i] for i in top_indices if filtered_scores[i] > 0.01]


def get_all_deals() -> list[Deal]:
    return _deals


def deals_count() -> int:
    return len(_deals)
