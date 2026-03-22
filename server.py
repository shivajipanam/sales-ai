"""
New FastAPI entry point for DealSearch AI.
The original main.py (Pathway pipeline) is preserved for reference.
Run locally: uvicorn server:app --reload
"""
from app.main import app  # noqa: F401 — re-exported for uvicorn
