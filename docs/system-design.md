# DealSearch AI — High Level System Design & Solution Document

> Personal reference document for interviews. Covers architecture, implementation decisions, challenges, learnings, and improvement paths.

---

## 1. What is the Application?

**DealSearch AI** is a web application that lets users search for Amazon deals in plain English. Instead of browsing category pages or filtering by price, users type natural language queries like:

- "cheap gaming headset under $50"
- "laptop with good reviews"
- "kitchen appliance with big discount"

The app returns ranked product cards with prices, savings %, ratings, Prime badges, and an AI-generated summary of the best picks.

**Live URL:** https://web-production-da46.up.railway.app/

---

## 2. System Architecture

```
User Browser
     │
     ▼
[ Frontend — HTML/CSS/JS ]
     │  HTTP POST /search
     ▼
[ FastAPI Backend ]
     ├── app/data.py       ← Data layer: loads deals, builds TF-IDF index, runs search
     ├── app/search.py     ← AI layer: calls Groq to generate summary
     ├── app/main.py       ← API routes: /search, /deals, /health, /refresh
     ├── app/models.py     ← Pydantic schemas: Deal, SearchRequest, SearchResponse
     └── server.py         ← Entry point: reads PORT from env, starts uvicorn
          │
          ├── TF-IDF Index (scikit-learn, in-memory)
          │       └── Built from JSONL deal data at startup
          │
          └── Groq API (external)
                  └── Llama 3.1 8B Instant — generates AI summary
```

**No database.** All deal data lives in memory. Rebuilt every time the app starts.

---

## 3. Data Flow — Search Request

1. User types "gaming headset" and clicks Search
2. Frontend sends `POST /search` with `{"query": "gaming headset", "max_price": null, "min_discount_pct": null, "top_k": 12}`
3. `app/main.py` calls `data.search_deals(query="gaming headset", ...)`
4. `app/data.py`:
   - Transforms the query into a TF-IDF vector using the fitted `TfidfVectorizer`
   - Computes cosine similarity against all 169 deal vectors
   - Filters out scores ≤ 0.01 (zero-match threshold)
   - Applies optional price/discount filters
   - Returns top-k deals sorted by score
5. `app/search.py` calls Groq API with the top 8 deals as context
6. Groq (Llama 3.1 8B) returns a 2-3 sentence plain English summary
7. FastAPI returns `SearchResponse` with deals list + AI summary
8. Frontend renders product cards + summary banner

---

## 4. Key Components

### 4.1 Data Layer (`app/data.py`)

**Responsibilities:**
- Parse JSONL files into `Deal` objects
- Build and maintain the in-memory TF-IDF search index
- Execute search queries with filtering

**JSONL Parsing:**
The seed data from the original project stored records as Python dict strings (`{'key': 'value'}`) instead of JSON (`{"key": "value"}`). To handle this, the parser tries `json.loads()` first and falls back to `ast.literal_eval()`:

```python
try:
    return json.loads(doc_str)
except (json.JSONDecodeError, TypeError):
    return ast.literal_eval(doc_str)
```

**`_record_to_deal()` — bug I fixed:**
There was a Python operator precedence bug where the savings percentage always returned 0. The original code:

```python
# BROKEN — ternary if/else evaluates before `or`, so old_price always = 0
old_price = float(r.get("old_price") or r.get("list_price", {}) if isinstance(r.get("list_price"), dict) else r.get("list_price") or 0)
```

Fixed by splitting into clear steps:
```python
list_price = r.get("list_price")
if isinstance(list_price, dict):
    list_price = list_price.get("value", 0)
old_price = float(r.get("old_price") or list_price or 0)
```

**TF-IDF Index:**
At startup, all deals are converted to text strings (`title + description + price + savings`) and fed to `TfidfVectorizer(ngram_range=(1,2), max_features=20000)`. This builds a sparse matrix of shape `(169, vocab_size)`. At query time, the query is transformed into the same vector space and cosine similarity is computed with `sklearn.metrics.pairwise.cosine_similarity`.

**Score threshold:**
Results with cosine similarity ≤ 0.01 are excluded. Without this, a query for "gaming" would return all shoe deals with a score of 0.0.

### 4.2 Search Layer (`app/search.py`)

Calls Groq's API with the top 8 deals formatted as readable text. The prompt tells the LLM to act as a shopping assistant and write a 2-3 sentence summary highlighting best value.

**Graceful degradation:** If `GROQ_API_KEY` is not set, a simple template-based summary is returned — the app still works without the API key.

### 4.3 API Layer (`app/main.py`)

Built with FastAPI. Key design decisions:
- **Lifespan context manager** — deals load at startup, not per-request
- **Static file serving** — the frontend `index.html` is served by the same FastAPI process (no separate web server needed)
- **Pydantic validation** — all request/response bodies are type-validated automatically

### 4.4 Frontend (`frontend/index.html`)

Single HTML file with embedded CSS and JS. No framework — vanilla JS. Key features:
- Hero search bar with filter chips (Max Price, Min Discount %)
- Product cards with image, title, price, savings badge, rating stars, Prime badge
- AI summary banner at the top of results
- Responsive grid layout

### 4.5 Entry Point (`server.py`)

```python
port = int(os.environ.get("PORT", 8000))
uvicorn.run("server:app", host="0.0.0.0", port=port)
```

Why this exists: Railway injects `$PORT` as an environment variable. The Dockerfile CMD `["uvicorn", "--port", "$PORT"]` doesn't expand shell variables (exec form vs shell form). Reading PORT in Python avoids this entirely.

---

## 5. Deployment Architecture

```
GitHub (main branch)
       │ push triggers
       ▼
Railway (auto-deploy)
       │ builds
       ▼
Docker Image (~400MB)
  ├── python:3.11-slim base
  ├── scikit-learn, fastapi, uvicorn, groq, numpy
  └── app code + JSONL data files
       │ runs
       ▼
Container
  └── python server.py → uvicorn on $PORT
```

**Why Docker over Nixpacks:** Railway's default Nixpacks builder was detected automatically. The original `requirements.txt` had `sentence-transformers` which pulls PyTorch (~3.5GB), causing the image to exceed Railway free tier's 4GB limit. Switched to a custom Dockerfile.

**Why scikit-learn over sentence-transformers:** PyTorch is 3.5GB. scikit-learn is 30MB. TF-IDF is fast, interpretable, and works well for product keyword search. The sentence-transformers version is preserved in `app/data_semantic.py` for when/if we upgrade to a paid plan.

---

## 6. Challenges Faced & How I Solved Them

### Challenge 1: Docker image exceeded 4GB Railway free tier limit
**Symptom:** Build succeeded but Railway rejected it: "Image of size 7.9 GB exceeded limit of 4.0 GB"
**Root cause:** `sentence-transformers` installs PyTorch (CUDA build by default = ~3.5GB)
**Solution:** Replaced `sentence-transformers` + PyTorch with `scikit-learn` TF-IDF. Kept semantic version in `data_semantic.py` as a future upgrade path.

### Challenge 2: `$PORT` not expanding in Docker CMD
**Symptom:** Repeated crash: `Error: Invalid value for '--port': '$PORT' is not a valid integer`
**Root cause:** Docker exec form (`CMD ["uvicorn", "--port", "$PORT"]`) does not run through a shell — environment variables are never substituted. Shell form and `railway.json` startCommand both failed too.
**Solution:** Read `PORT` directly in Python: `port = int(os.environ.get("PORT", 8000))`

### Challenge 3: Search returning irrelevant results
**Symptom:** Searching "gaming" returned Dr. Scholl's shoes as top result
**Root cause 1:** The seed JSONL data contained only shoe deals (Amazon Japan shoe deals from 2023)
**Root cause 2:** No score threshold — zero-similarity results were included and sorted arbitrarily
**Solution:** Created `examples/data/mock_discounts.jsonl` with 50 deals across 15 categories (gaming, laptops, phones, TVs, headphones, etc.) + added `score > 0.01` threshold

### Challenge 4: `savings_pct` always returning 0
**Symptom:** All deals showed 0% savings despite having different `old_price` and `deal_price`
**Root cause:** Python operator precedence bug — ternary conditional was evaluated before `or`
**Solution:** Restructured into 3 explicit lines (see section 4.1)

### Challenge 5: Broken langchain imports (in rag-pipeline-simple, same codebase era)
**Context:** Not directly in sales-ai, but good to know for interviews
**Root cause:** LangChain split packages in v0.2+: `langchain.schema` → `langchain_core.documents`, `langchain.text_splitter` → `langchain_text_splitters`
**Learning:** Always pin exact versions in `requirements.txt` for AI/ML projects — the ecosystem moves fast and breaks imports between minor versions.

---

## 7. What I Learned

### Technical
- **TF-IDF vs embeddings tradeoff:** TF-IDF is 100x smaller and faster to build, but purely keyword-based — it won't match "earbuds" to a query for "headphones". Sentence transformers understand semantics but need PyTorch. The right choice depends on your infrastructure budget.
- **Docker image size matters:** AI/ML apps balloon in size because of PyTorch. CPU-only builds (`--index-url https://download.pytorch.org/whl/cpu`) save ~2.7GB. For production, always check image size before deploying.
- **FastAPI lifespan for startup tasks:** Using `@asynccontextmanager` lifespan instead of `@app.on_event("startup")` is the modern FastAPI pattern. It ensures heavy work (loading data, building index) happens once at boot, not per request.
- **Pydantic validation catches bugs early:** Defining strict schemas for API input/output forces you to think about your data contract. It also gives you free OpenAPI docs at `/docs`.
- **Shell vs exec form in Dockerfiles:** A subtle Docker gotcha. `CMD ["cmd", "$VAR"]` doesn't expand env vars. `CMD cmd $VAR` does. Or just read the env var in your code.

### Process
- **Test before deploying:** All 13 tests pass locally before every push. This caught the `savings_pct` bug before it ever hit production.
- **Seed data quality is critical:** The search was useless until the data had variety. Data quality > algorithm sophistication.
- **Graceful degradation:** The app works without `GROQ_API_KEY`. This is important for demos — you can show it without needing to share API keys.

---

## 8. How to Improve This App

### Short-term (easy wins)
| Improvement | Effort | Impact |
|---|---|---|
| Add `GROQ_API_KEY` to Railway env vars | 2 min | AI summaries go live |
| Add more mock deal categories (books, sports, beauty) | 1 hour | Better search coverage |
| Add a "Sort by: Best Match / Price / Discount" toggle | 2 hours | Better UX |
| Cache search results for repeated queries | 2 hours | Faster repeat searches |
| Add category filter chips (Electronics, Gaming, Kitchen) | 3 hours | Better browsability |

### Medium-term
| Improvement | Effort | Impact |
|---|---|---|
| Switch to sentence-transformers (upgrade Railway plan or use Render) | 1 day | Semantic search — "earbuds" matches "headphones" |
| Connect to a real deals API (Rainforest API, SERP API) | 2 days | Live Amazon deals instead of static data |
| Add SQLite for deal persistence across restarts | 1 day | Data survives container restarts |
| Add user accounts + saved deals | 3 days | Personalisation |
| Add price history charts | 3 days | Shows whether a deal is genuinely good |

### Long-term / Architecture changes
| Improvement | Effort | Impact |
|---|---|---|
| Move to Pinecone vector store | 1 week | Scales to millions of deals, true semantic search |
| Add a background scheduler to refresh deals daily | 1 week | Always fresh data |
| Add deal alerts (email/SMS when a product drops below a price) | 2 weeks | High user value |
| Multi-retailer support (Best Buy, Walmart) | 1 month | Much wider inventory |

---

## 9. Interview Talking Points

**"Walk me through the architecture"**
> The app is a FastAPI backend that serves both the API and the frontend HTML. At startup, it loads Amazon deal data from JSONL files and builds a TF-IDF search index in memory. When a user searches, we transform their query into the same vector space and compute cosine similarity to rank deals. The top results go to Groq's Llama 3.1 model which writes a plain English summary. Everything runs in a single Docker container deployed on Railway.

**"Why TF-IDF instead of embeddings?"**
> Practical constraint — Railway's free tier has a 4GB Docker image limit. PyTorch alone is 3.5GB. TF-IDF with scikit-learn is 30MB. For product keyword search it performs well enough because users tend to search with exact product keywords. I've kept the sentence-transformers version in `data_semantic.py` — one rename of the Dockerfile and a plan upgrade switches to semantic search.

**"How do you handle the case where no deals match?"**
> Two mechanisms. First, a cosine similarity threshold of 0.01 — anything below is excluded, so a search for "gaming" won't return shoe deals that score 0. Second, the fallback message in the AI summary layer explicitly tells the user to try broader terms.

**"What would you do differently?"**
> I'd connect to a live deals API from day one — the static seed data limitation was the biggest issue in production. I'd also add a proper database (SQLite at minimum) so deals persist across container restarts. And I'd use semantic search from the start if the infrastructure budget allows it.

**"How did you deploy it?"**
> Docker on Railway. The Dockerfile installs dependencies and runs `python server.py` which reads the PORT environment variable and starts uvicorn. Railway auto-deploys on every push to the main branch via a GitHub webhook.

---

## 10. File Structure Reference

```
sales-ai/
├── app/
│   ├── __init__.py
│   ├── data.py           ← TF-IDF search, deal loading, JSONL parsing
│   ├── data_semantic.py  ← sentence-transformers version (future upgrade)
│   ├── main.py           ← FastAPI routes
│   ├── models.py         ← Pydantic schemas
│   └── search.py         ← Groq AI summary
├── docs/
│   └── system-design.md  ← This document
├── examples/
│   └── data/
│       ├── rainforest_discounts.jsonl  ← Original Amazon shoe deals
│       ├── csv_discounts.jsonl         ← CSV-sourced deals
│       └── mock_discounts.jsonl        ← 50 deals across 15 categories (added)
├── frontend/
│   └── index.html        ← Single-file frontend
├── tests/
│   └── test_api.py       ← 13 tests, all passing
├── Dockerfile            ← Production Docker build
├── server.py             ← Entry point, reads PORT from env
├── railway.json          ← Railway deployment config
└── app_requirements.txt  ← Pinned dependencies (tested versions)
```
