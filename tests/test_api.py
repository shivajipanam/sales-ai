"""
Tests for DealSearch AI backend.
Run with: pytest tests/ -v

Embedding model loads on first run (~30s). All Groq calls are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Client with deals loaded from seed JSONL files."""
    from app.main import app
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
# Data layer tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_deals_loaded(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["deals_loaded"] > 0  # seed JSONL data parsed correctly


def test_deals_endpoint_returns_list(client):
    res = client.get("/deals?limit=10")
    assert res.status_code == 200
    deals = res.json()
    assert isinstance(deals, list)
    assert len(deals) <= 10
    # Each deal has required fields
    for d in deals:
        assert "deal_price" in d
        assert "title" in d
        assert "savings_pct" in d


def test_deals_filter_by_price(client):
    res = client.get("/deals?max_price=30")
    assert res.status_code == 200
    for d in res.json():
        assert d["deal_price"] <= 30


def test_deals_filter_by_discount(client):
    res = client.get("/deals?min_discount=20")
    assert res.status_code == 200
    for d in res.json():
        assert d["savings_pct"] >= 20


# ═══════════════════════════════════════════════════════════════════════════════
# Search tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_search_returns_results(client):
    with patch("app.main.search.ai_summary", return_value="Great deals found!"):
        res = client.post("/search", json={"query": "shoes"})
    assert res.status_code == 200
    body = res.json()
    assert "deals" in body
    assert "ai_summary" in body
    assert isinstance(body["deals"], list)
    assert body["ai_summary"] == "Great deals found!"


def test_search_result_structure(client):
    with patch("app.main.search.ai_summary", return_value="Test summary"):
        res = client.post("/search", json={"query": "electronics"})
    assert res.status_code == 200
    body = res.json()
    if body["deals"]:
        d = body["deals"][0]
        assert "deal_price" in d
        assert "old_price" in d
        assert "savings_pct" in d
        assert "title" in d
        assert "link" in d
        assert "rating" in d


def test_search_with_price_filter(client):
    with patch("app.main.search.ai_summary", return_value="Cheap deals"):
        res = client.post("/search", json={"query": "anything", "max_price": 50})
    assert res.status_code == 200
    for d in res.json()["deals"]:
        assert d["deal_price"] <= 50


def test_search_with_discount_filter(client):
    with patch("app.main.search.ai_summary", return_value="Big discounts"):
        res = client.post("/search", json={"query": "anything", "min_discount_pct": 30})
    assert res.status_code == 200
    for d in res.json()["deals"]:
        assert d["savings_pct"] >= 30


def test_search_empty_query_fails(client):
    res = client.post("/search", json={"query": "  "})
    assert res.status_code == 400


def test_search_graceful_without_groq_key(client):
    """AI summary degrades gracefully if GROQ_API_KEY is not set."""
    import os
    original = os.environ.pop("GROQ_API_KEY", None)
    try:
        res = client.post("/search", json={"query": "laptop"})
        assert res.status_code == 200
        assert len(res.json()["ai_summary"]) > 0  # fallback summary still returned
    finally:
        if original:
            os.environ["GROQ_API_KEY"] = original


# ═══════════════════════════════════════════════════════════════════════════════
# Parser / data layer unit tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_record_to_deal_valid():
    from app.data import _record_to_deal
    record = {
        "deal_id": "abc123",
        "title": "Test Shoes",
        "description": "Great shoes",
        "image": "https://example.com/img.jpg",
        "link": "https://amazon.com/deal/abc123",
        "deal_price": 29.99,
        "old_price": 59.99,
        "currency": "USD",
        "rating": 4.5,
        "ratings_total": 1200,
        "deal_type": "DEAL_OF_THE_DAY",
        "is_prime": True,
        "is_lightning_deal": False,
    }
    deal = _record_to_deal(record)
    assert deal is not None
    assert deal.deal_price == 29.99
    assert deal.savings_pct == pytest.approx(50.0, abs=1)
    assert deal.is_prime is True


def test_record_to_deal_skips_zero_price():
    from app.data import _record_to_deal
    deal = _record_to_deal({"title": "Free thing", "deal_price": 0})
    assert deal is None


def test_root_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert b"DealSearch" in res.content
