# LinkedIn Post — DealSearch AI

---

🚀 Just shipped DealSearch AI — search Amazon deals in plain English.

Instead of scrolling through endless category pages, just type what you're looking for:

→ "cheap gaming headset under $50"
→ "laptop with good reviews"
→ "kitchen appliance with big discount"

The app returns ranked product cards with prices, savings %, ratings, and an AI-generated summary of the best picks — all in under a second.

🔗 Live: https://web-production-da46.up.railway.app/

---

**What's under the hood:**

🔍 **Natural language search** — I built a TF-IDF vector search engine using scikit-learn. Every deal is embedded as a vector at startup. At query time, the user's query is projected into the same space and cosine similarity ranks all deals. Results below a relevance threshold are excluded entirely — so a search for "gaming" never returns shoe deals.

🤖 **AI summaries** — The top results are passed to Groq's Llama 3.1 8B model, which writes a plain English summary highlighting the best value picks and why. The app degrades gracefully if no API key is available.

⚡ **FastAPI backend** — Single process handles both the API and serves the frontend HTML. Data loads once at startup via a lifespan context manager, so every request is fast with no cold data loading.

🐳 **Deployed on Railway with Docker** — One of the trickier parts was keeping the image under Railway's 4GB free tier limit. PyTorch alone is ~3.5GB. I swapped sentence-transformers for scikit-learn, cutting the image from 7.9GB to ~400MB while preserving the semantic search version for a future upgrade.

📦 **13 automated tests** — All API routes and data parsing logic are tested with mocked external calls. Caught a subtle Python operator precedence bug in production savings calculation before it ever hit users.

---

**The stack:**
- Backend: Python + FastAPI + uvicorn
- Search: scikit-learn TF-IDF + numpy cosine similarity
- AI: Groq (Llama 3.1 8B Instant) — free tier
- Frontend: Vanilla HTML/CSS/JS
- Deploy: Docker on Railway (CI/CD via GitHub webhook)

---

The biggest learning? Data quality beats algorithm sophistication every time. The search was useless until I replaced a single-category seed dataset with diverse products across 15 categories. A perfect algorithm on bad data gives bad results.

Would love to hear what features you'd find most useful — price history tracking, deal alerts, or multi-retailer support?

#Python #FastAPI #AI #MachineLearning #NLP #BuildInPublic #SideProject #WebDevelopment

---

*Copy the text above the `---` line for LinkedIn. Adjust tone as needed.*
