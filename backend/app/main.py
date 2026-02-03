from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.ingestion.rss import parse_rss_feed

app = FastAPI(
    title="Ground Truth API",
    description="Backend API for Ground Truth — AI news summaries",
    version="0.1.0"
)

VALID_CATEGORIES = {"models", "hardware", "research", "policy", "industry", "tools"}


class Source(BaseModel):
    id: str
    name: str
    url: str


class Article(BaseModel):
    id: str
    title: str
    category: str
    published_at: str
    summary: str
    source: Source
    original_url: str


class FeedResponse(BaseModel):
    items: List[Article]

SOURCES = [
    {
        "id": "src_openai",
        "name": "OpenAI",
        "url": "https://openai.com",
        "rss_url": "https://openai.com/news/rss.xml",
        "category": "models",
        "active": True,
    },
    {
        "id": "src_google_ai",
        "name": "Google AI Blog",
        "url": "https://blog.google/technology/ai/",
        "rss_url": "https://blog.google/technology/ai/rss/",
        "category": "research",
        "active": True,
    },
    {
        "id": "src_meta_ai",
        "name": "Meta AI",
        "url": "https://ai.meta.com/blog/",
        "rss_url": "https://ai.meta.com/blog/rss/",
        "category": "industry",
        "active": True,
    },
]

ARTICLES = []
for source in SOURCES:
    if not source["active"]:
        continue

    source_ref = {
        "id": source["id"],
        "name": source["name"],
        "url": source["url"],
    }

    try:
        parsed_articles = parse_rss_feed(source["rss_url"], source_ref, source["category"])
    except Exception as exc:
        print(f"Warning: failed processing source {source['id']}: {exc}")
        continue

    ARTICLES.extend(parsed_articles)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/v1/feed", response_model=FeedResponse)
def get_feed():
    items = sorted(ARTICLES, key=lambda article: article["published_at"], reverse=True)
    return {"items": items}


@app.get("/v1/feed/{category}", response_model=FeedResponse)
def get_feed_by_category(category: str):
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=404, detail="Category not found")

    filtered_items = []
    for article in ARTICLES:
        if article["category"] == category:
            filtered_items.append(article)

    sorted_items = sorted(filtered_items, key=lambda article: article["published_at"], reverse=True)
    return {"items": sorted_items}


@app.get("/v1/articles/{id}", response_model=Article)
def get_article_by_id(id: str):
    for article in ARTICLES:
        if article["id"] == id:
            return article

    raise HTTPException(status_code=404, detail="Article not found")
