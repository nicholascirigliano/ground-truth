from datetime import UTC, datetime
import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from app.db.models import Article as DBArticle
from app.db.models import Source as DBSource
from app.db.models import Summary as DBSummary
from app.db.session import SessionLocal
from app.ingestion.rss import canonicalize_url, parse_rss_feed

app = FastAPI(
    title="Ground Truth API",
    description="Backend API for Ground Truth — AI news summaries",
    version="0.1.0"
)

VALID_CATEGORIES = {"models", "hardware", "research", "policy", "industry", "tools"}
SUMMARY_MODEL = "gpt-4.1-mini"
SUMMARY_VERSION = 1


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


def _parse_iso8601(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _datetime_to_iso8601(value: datetime) -> str:
    if value.tzinfo is None:
        utc_value = value.replace(tzinfo=UTC)
    else:
        utc_value = value.astimezone(UTC)
    return utc_value.isoformat().replace("+00:00", "Z")


def _article_to_response_item(article: DBArticle) -> dict:
    return {
        "id": article.id,
        "title": article.title,
        "category": article.category,
        "published_at": _datetime_to_iso8601(article.published_at),
        "summary": article.summary.summary_text,
        "source": {
            "id": article.source.id,
            "name": article.source.name,
            "url": article.source.url,
        },
        "original_url": article.canonical_url,
    }


def _generate_summary_text(
    *,
    title: str,
    source_name: str,
    original_url: str,
    category: str,
) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY is not set; skipping summarization.")
        return None

    try:
        from openai import OpenAI
    except Exception as exc:
        print(f"Warning: OpenAI client not available; skipping summarization: {exc}")
        return None

    system_prompt = (
        "You are a neutral AI news summarizer. "
        "Write a factual summary in 4-5 sentences and no more than 130 words. "
        "Do not use quotes. Do not add hype, opinion, or speculation. "
        "The summary must answer what happened and why it matters."
    )

    user_prompt = (
        f"Title: {title}\n"
        f"Source: {source_name}\n"
        f"Original URL: {original_url}\n"
        f"Category: {category}\n"
    )

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=SUMMARY_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        print(f"Warning: summarization failed for {original_url}: {exc}")
        return None

    if not completion.choices:
        return None

    message = completion.choices[0].message
    if message is None or message.content is None:
        return None

    return message.content.strip()


def _create_summary_if_missing(db, article: DBArticle, source_name: str) -> None:
    existing_summary = db.get(DBSummary, article.id)
    if existing_summary is not None:
        return

    summary_text = _generate_summary_text(
        title=article.title,
        source_name=source_name,
        original_url=article.canonical_url,
        category=article.category,
    )
    if not summary_text:
        return

    db_summary = DBSummary(
        article_id=article.id,
        summary_text=summary_text,
        model=SUMMARY_MODEL,
        version=SUMMARY_VERSION,
        generated_at=datetime.now(UTC).replace(tzinfo=None),
    )
    db.add(db_summary)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"Warning: failed to persist summary for {article.id}: {exc}")


def _persist_sources_and_articles() -> None:
    db = SessionLocal()
    try:
        for source in SOURCES:
            if not source["active"]:
                continue

            existing_source = db.get(DBSource, source["id"])
            if existing_source is None:
                db_source = DBSource(
                    id=source["id"],
                    name=source["name"],
                    url=source["url"],
                    rss_url=source["rss_url"],
                    default_category=source["category"],
                    active=source["active"],
                )
                db.add(db_source)
                db.commit()

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

            for parsed_article in parsed_articles:
                canonical_url = canonicalize_url(parsed_article["original_url"])
                existing_article = db.execute(
                    select(DBArticle.id).where(DBArticle.canonical_url == canonical_url)
                ).scalar_one_or_none()
                if existing_article is not None:
                    continue

                db_article = DBArticle(
                    id=parsed_article["id"],
                    canonical_url=canonical_url,
                    source_id=source["id"],
                    title=parsed_article["title"],
                    category=parsed_article["category"],
                    published_at=_parse_iso8601(parsed_article["published_at"]),
                    ingested_at=datetime.now(UTC).replace(tzinfo=None),
                )

                db.add(db_article)
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    duplicate_article = db.execute(
                        select(DBArticle.id).where(DBArticle.canonical_url == canonical_url)
                    ).scalar_one_or_none()
                    if duplicate_article is not None:
                        continue
                    raise
                except Exception:
                    db.rollback()
                    raise

                _create_summary_if_missing(db, db_article, source["name"])
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


_persist_sources_and_articles()

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/v1/feed", response_model=FeedResponse)
def get_feed():
    db = SessionLocal()
    try:
        articles = db.execute(
            select(DBArticle)
            .join(DBSummary, DBSummary.article_id == DBArticle.id)
            .options(joinedload(DBArticle.source), joinedload(DBArticle.summary))
            .order_by(DBArticle.published_at.desc())
        ).scalars().all()
        items = [_article_to_response_item(article) for article in articles]
        return {"items": items}
    finally:
        db.close()


@app.get("/v1/feed/{category}", response_model=FeedResponse)
def get_feed_by_category(category: str):
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=404, detail="Category not found")

    db = SessionLocal()
    try:
        articles = db.execute(
            select(DBArticle)
            .join(DBSummary, DBSummary.article_id == DBArticle.id)
            .options(joinedload(DBArticle.source), joinedload(DBArticle.summary))
            .where(DBArticle.category == category)
            .order_by(DBArticle.published_at.desc())
        ).scalars().all()
        items = [_article_to_response_item(article) for article in articles]
        return {"items": items}
    finally:
        db.close()


@app.get("/v1/articles/{id}", response_model=Article)
def get_article_by_id(id: str):
    db = SessionLocal()
    try:
        article = db.execute(
            select(DBArticle)
            .join(DBSummary, DBSummary.article_id == DBArticle.id)
            .options(joinedload(DBArticle.source), joinedload(DBArticle.summary))
            .where(DBArticle.id == id)
        ).scalar_one_or_none()
        if article is not None:
            return _article_to_response_item(article)
    finally:
        db.close()

    raise HTTPException(status_code=404, detail="Article not found")
