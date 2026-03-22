# DealSearch AI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Credits

Original project by [Bobur Umurzokov](https://github.com/Boburmirzo) — [chatgpt-api-python-sales](https://github.com/Boburmirzo/chatgpt-api-python-sales).

This fork significantly reworks the original into a deployable web application.

## What was added

- **FastAPI backend** replacing the original Pathway streaming pipeline
- **In-memory semantic search** using `sentence-transformers` (no Pinecone/vector DB needed)
- **Natural language deal search** — type "cheap headphones under $50" and get ranked results
- **AI-powered summaries** via Groq (free Llama 3.1) — degrades gracefully without API key
- **Product card UI** with savings %, ratings, Prime badge, filter chips
- **Full frontend** (`frontend/index.html`) — hero search, deal grid, AI summary banner
- **Railway deployment config** (`Procfile` + `railway.json`)
- **13 passing tests** covering data layer, search, filters, and API endpoints
- **Bug fix** in `_record_to_deal` — operator precedence error caused `savings_pct` to always return 0

## Description

Search Amazon deals in plain English. Returns ranked product cards with prices, savings %, ratings, and an AI-generated summary. Works with zero API keys using seed data.

## Tech Stack

- **Backend:** FastAPI + uvicorn
- **Search:** sentence-transformers (all-MiniLM-L6-v2) + numpy cosine similarity
- **AI Summaries:** Groq (free, optional)
- **Data:** Real Amazon deals from seed JSONL files
- **Deploy:** Railway

## Installation

```bash
git clone https://github.com/shivajipanam/sales-ai.git
cd sales-ai

pip install -r app_requirements.txt

cp .env.example .env
# Optionally add GROQ_API_KEY for AI summaries

uvicorn server:app --reload
```

Open `http://localhost:8000` in your browser.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Optional | Enables AI summaries (free at console.groq.com) |
| `RAINFOREST_API_KEY` | Optional | Enables live Amazon deal refresh (paid) |

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
