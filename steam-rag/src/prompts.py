"""Prompt templates for classification and RAG generation."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, Sequence

TOPIC_DEFINITIONS = {
    "netcode": "Online connectivity, rollback, matchmaking, lag, p2p stability.",
    "combat": "Core fighting systems, move sets, balance, responsiveness.",
    "story": "Narrative quality, characters, cutscenes, campaign writing.",
    "graphics": "Visual fidelity, art style, performance related to visuals.",
    "world_tour": "World Tour / adventure mode content and pacing.",
    "progression": "Unlocks, XP, grind, rewards, leveling systems.",
    "monetization": "Pricing, DLC, microtransactions, battle pass.",
    "ui": "Menus, HUD, controller navigation, readability.",
    "stability": "Crashes, bugs, optimisation, technical issues.",
}

SENTIMENT_LABELS = ["very_positive", "positive", "neutral", "negative", "very_negative"]


def build_classifier_messages(reviews: Sequence[Mapping[str, str]], topics: Sequence[str]) -> list[dict]:
    """Return chat messages instructing the LLM to classify reviews."""

    topic_lines = "\n".join(f"- {topic}: {TOPIC_DEFINITIONS.get(topic, '')}" for topic in topics)
    review_lines = []
    for item in reviews:
        review_lines.append(
            f"review_id={item['review_id']}\n" + item["review_text"].strip().replace("\n", " ")
        )
    reviews_section = "".join("\n---\n" + block + "\n" for block in review_lines)
    user_prompt = (
        "You are an expert analyst labelling Steam reviews. Only use the supplied topic list.\n"
        "Topics:\n"
        f"{topic_lines}\n\n"
        "Return strict JSON with this schema (do not wrap in markdown):\n"
        "{\n  \"reviews\": [\n    {\n      \"review_id\": \"<id>\",\n      \"topics\": [\"topic\"...],\n      \"sentiment_by_topic\": {\"topic\": \"sentiment\"},\n      \"confidence_by_topic\": {\"topic\": 0.0}\n    }\n  ]\n}\n"
        f"Use the five sentiment labels: {', '.join(SENTIMENT_LABELS)}.\n"
        "Only include topics that are clearly present. Confidence is between 0 and 1.\n\n"
        "Reviews:\n"
        f"{reviews_section}"
    )
    messages = [
        {
            "role": "system",
            "content": "You label topics and sentiment for Steam game reviews. Respond with compact JSON only.",
        },
        {"role": "user", "content": user_prompt},
    ]
    return messages


def build_rag_messages(
    query: str,
    snippets: Sequence[Mapping[str, str]],
    model_name: str,
    date_range: tuple[datetime | None, datetime | None] | None = None,
) -> list[dict]:
    """Construct messages for the summarisation agent."""

    date_text = ""
    if date_range:
        start, end = date_range
        if start and end:
            date_text = f"Covering reviews from {start.date()} to {end.date()}."
        elif start:
            date_text = f"Covering reviews from {start.date()} onwards."
        elif end:
            date_text = f"Covering reviews up to {end.date()}."
    lines = []
    for item in snippets:
        created_at = item.get("created_at")
        if isinstance(created_at, datetime):
            created_text = created_at.date().isoformat()
        else:
            created_text = str(created_at)
        lines.append(
            f"[ {item['review_id']} | {created_text} | helpful={item.get('helpful_count', 0)} ] "
            + item.get("review_text", "").replace("\n", " ")
        )
    reviews_text = "".join("\n" + line for line in lines)
    user_prompt = (
        f"Question: {query}\n"
        f"{date_text}\n"
        "Cite using [review_id] brackets. Keep answers concise with factual bullet points.\n\n"
        "Reviews:\n"
        f"{reviews_text}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer strictly from provided reviews; use short, concrete bullets; include citations like "
                "[review_id]; qualify statements with few/some/many; avoid hallucinations."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return messages


__all__ = [
    "TOPIC_DEFINITIONS",
    "SENTIMENT_LABELS",
    "build_classifier_messages",
    "build_rag_messages",
]
