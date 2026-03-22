"""
AI-powered deal search using Groq (free Llama 3.1).
Takes top-k deals from embedding search and asks Groq to:
  - rank them by relevance to the user's query
  - write a brief natural language summary of the best picks
"""
import json
import os

from groq import Groq

from app.models import Deal

GROQ_MODEL = "llama-3.1-8b-instant"


def _deals_to_text(deals: list[Deal]) -> str:
    lines = []
    for i, d in enumerate(deals, 1):
        savings = f"${d.old_price - d.deal_price:.2f} off ({d.savings_pct:.0f}%)" if d.savings_pct > 0 else "no price history"
        prime = " [Prime]" if d.is_prime else ""
        lightning = " ⚡Lightning" if d.is_lightning else ""
        lines.append(
            f"{i}. {d.title}{prime}{lightning}\n"
            f"   Price: ${d.deal_price:.2f} (was ${d.old_price:.2f}) — {savings}\n"
            f"   Rating: {d.rating}/5 ({d.ratings_total} reviews)\n"
            f"   Deal type: {d.deal_type}"
        )
    return "\n\n".join(lines)


def ai_summary(query: str, deals: list[Deal]) -> str:
    """
    Ask Groq to write a 2-3 sentence shopper-friendly summary of the best deals
    for the given query. Returns plain text.
    """
    if not deals:
        return "No matching deals found. Try a broader search term."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Graceful degradation — return a basic summary without AI
        best = deals[0]
        return (
            f"Found {len(deals)} deals matching '{query}'. "
            f"Top pick: {best.title} at ${best.deal_price:.2f} "
            f"({best.savings_pct:.0f}% off). "
            f"Set GROQ_API_KEY for AI-powered summaries."
        )

    deals_text = _deals_to_text(deals[:8])  # top 8 for context window

    prompt = f"""You are a helpful shopping assistant. A user searched for: "{query}"

Here are the top matching deals found:

{deals_text}

Write a 2-3 sentence shopper-friendly summary highlighting the best picks and why they're good value.
Be specific about prices and savings. Keep it concise and enthusiastic but honest."""

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=200,
    )
    return completion.choices[0].message.content.strip()
