import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import data, search
from app.models import Deal, RefreshResponse, SearchRequest, SearchResponse

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load deals & build embedding index at startup
    n = data.load_all_deals()
    print(f"[startup] Loaded {n} deals into memory.")
    yield


app = FastAPI(
    title="DealSearch AI — Find the Best Deals Instantly",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


# ── Root ───────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "DealSearch AI running. See /docs for endpoints."}


@app.get("/health")
def health():
    return {"status": "ok", "deals_loaded": data.deals_count()}


# ── Search ─────────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
def search_deals(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if data.deals_count() == 0:
        raise HTTPException(status_code=503, detail="Deal data not loaded yet. Try again shortly.")

    results = data.search_deals(
        query=request.query,
        max_price=request.max_price,
        min_discount_pct=request.min_discount_pct,
        top_k=request.top_k,
    )

    summary = search.ai_summary(request.query, results)

    return SearchResponse(
        query=request.query,
        ai_summary=summary,
        deals=results,
        total_found=len(results),
    )


# ── Browse all deals ───────────────────────────────────────────────────────────

@app.get("/deals", response_model=list[Deal])
def get_deals(
    limit: int = Query(default=50, le=200),
    min_discount: float = Query(default=0),
    max_price: float = Query(default=99999),
):
    all_deals = data.get_all_deals()
    filtered = [
        d for d in all_deals
        if d.savings_pct >= min_discount and d.deal_price <= max_price
    ]
    return filtered[:limit]


# ── Refresh ────────────────────────────────────────────────────────────────────

@app.post("/refresh", response_model=RefreshResponse)
def refresh():
    """
    Refresh deals from Rainforest API (requires RAINFOREST_API_KEY env var).
    Falls back to reloading seed data if no key is set.
    """
    try:
        n = data.refresh_from_rainforest()
        return RefreshResponse(deals_loaded=n, message=f"Refreshed {n} live deals from Amazon.")
    except ValueError:
        n = data.load_all_deals()
        return RefreshResponse(deals_loaded=n, message=f"Reloaded {n} deals from seed data (set RAINFOREST_API_KEY for live deals).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
