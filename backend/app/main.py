from datetime import UTC, datetime
import os
from typing import List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from app.db.models import Article as DBArticle
from app.db.models import Source as DBSource
from app.db.models import Summary as DBSummary
from app.db.session import SessionLocal
from app.ingestion.rss import canonicalize_url, parse_rss_feed
from app.sources import SOURCES

app = FastAPI(
    title="Ground Truth API",
    description="Backend API for Ground Truth — AI news summaries",
    version="0.1.0"
)

VALID_CATEGORIES = {
    "models",
    "research",
    "products",
    "open_source",
    "hardware",
    "regulation",
}
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
    next_cursor: str | None


class ArticleDetailSource(BaseModel):
    name: str
    url: str


class ArticleDetail(BaseModel):
    id: str
    title: str
    summary: str
    category: str
    published_at: str
    original_url: str
    source: ArticleDetailSource


class CategoryItem(BaseModel):
    id: str
    label: str


class CategoriesResponse(BaseModel):
    items: List[CategoryItem]


class SourceItem(BaseModel):
    id: str
    name: str
    url: str


class SourcesResponse(BaseModel):
    items: List[SourceItem]


def _parse_iso8601(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _datetime_to_iso8601(value: datetime) -> str:
    if value.tzinfo is None:
        utc_value = value.replace(tzinfo=UTC)
    else:
        utc_value = value.astimezone(UTC)
    return utc_value.isoformat().replace("+00:00", "Z")


def _parse_cursor(cursor: str) -> datetime:
    try:
        parsed = _parse_iso8601(cursor)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid cursor format")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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


def _article_to_detail_response_item(article: DBArticle) -> dict:
    return {
        "id": article.id,
        "title": article.title,
        "summary": article.summary.summary_text,
        "category": article.category,
        "published_at": _datetime_to_iso8601(article.published_at),
        "original_url": article.canonical_url,
        "source": {
            "name": article.source.name,
            "url": article.source.url,
        },
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
def get_feed(
    limit: int = Query(default=20, ge=1, le=50),
    cursor: str | None = Query(default=None),
    sources: str | None = Query(default=None),
):
    db = SessionLocal()
    try:
        # Discover feed: when sources is omitted, behavior matches current feed.
        # For You feed: when sources is provided, filter by requested active sources.
        source_ids = None
        if sources:
            requested_ids = {value.strip() for value in sources.split(",") if value.strip()}
            if requested_ids:
                source_ids = set(
                    db.execute(
                        select(DBSource.id)
                        .where(DBSource.active.is_(True))
                        .where(DBSource.id.in_(requested_ids))
                    ).scalars().all()
                )
            else:
                source_ids = set()

        if source_ids is not None and len(source_ids) == 0:
            return {"items": [], "next_cursor": None}

        query = (
            select(DBArticle)
            .join(DBSummary, DBSummary.article_id == DBArticle.id)
            .options(joinedload(DBArticle.source), joinedload(DBArticle.summary))
            .order_by(DBArticle.published_at.desc())
            .limit(limit + 1)
        )
        if source_ids is not None:
            query = query.where(DBArticle.source_id.in_(source_ids))
        if cursor is not None:
            cursor_dt = _parse_cursor(cursor).replace(tzinfo=None)
            query = query.where(DBArticle.published_at < cursor_dt)

        articles = db.execute(query).scalars().all()
        page_articles = articles[:limit]
        items = [_article_to_response_item(article) for article in page_articles]

        next_cursor = None
        if len(articles) > limit and page_articles:
            next_cursor = _datetime_to_iso8601(page_articles[-1].published_at)

        return {"items": items, "next_cursor": next_cursor}
    finally:
        db.close()


@app.get("/v1/feed/{category}", response_model=FeedResponse)
def get_feed_by_category(
    category: str,
    limit: int = Query(default=20, ge=1, le=50),
    cursor: str | None = Query(default=None),
    sources: str | None = Query(default=None),
):
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=404, detail="Category not found")

    db = SessionLocal()
    try:
        # Discover feed: when sources is omitted, behavior matches current feed.
        # For You feed: when sources is provided, filter by requested active sources.
        source_ids = None
        if sources:
            requested_ids = {value.strip() for value in sources.split(",") if value.strip()}
            if requested_ids:
                source_ids = set(
                    db.execute(
                        select(DBSource.id)
                        .where(DBSource.active.is_(True))
                        .where(DBSource.id.in_(requested_ids))
                    ).scalars().all()
                )
            else:
                source_ids = set()

        if source_ids is not None and len(source_ids) == 0:
            return {"items": [], "next_cursor": None}

        query = (
            select(DBArticle)
            .join(DBSummary, DBSummary.article_id == DBArticle.id)
            .options(joinedload(DBArticle.source), joinedload(DBArticle.summary))
            .where(DBArticle.primary_category == category)
            .order_by(DBArticle.published_at.desc())
            .limit(limit + 1)
        )
        if source_ids is not None:
            query = query.where(DBArticle.source_id.in_(source_ids))
        if cursor is not None:
            cursor_dt = _parse_cursor(cursor).replace(tzinfo=None)
            query = query.where(DBArticle.published_at < cursor_dt)

        articles = db.execute(query).scalars().all()
        page_articles = articles[:limit]
        items = [_article_to_response_item(article) for article in page_articles]

        next_cursor = None
        if len(articles) > limit and page_articles:
            next_cursor = _datetime_to_iso8601(page_articles[-1].published_at)

        return {"items": items, "next_cursor": next_cursor}
    finally:
        db.close()


@app.get("/v1/articles/{id}", response_model=ArticleDetail)
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
            return _article_to_detail_response_item(article)
    finally:
        db.close()

    raise HTTPException(status_code=404, detail="Article not found")


@app.get("/v1/categories", response_model=CategoriesResponse)
def get_categories():
    db = SessionLocal()
    try:
        categories = db.execute(
            select(DBArticle.primary_category)
            .join(DBSummary, DBSummary.article_id == DBArticle.id)
            .where(DBArticle.primary_category.is_not(None))
            .where(DBArticle.primary_category != "")
            .where(DBArticle.primary_category != "uncategorized")
            .distinct()
            .order_by(DBArticle.primary_category.asc())
        ).scalars().all()

        items = [
            {"id": category, "label": category.replace("_", " ").title()}
            for category in categories
        ]
        return {"items": items}
    finally:
        db.close()


@app.get("/v1/sources", response_model=SourcesResponse)
def get_sources():
    db = SessionLocal()
    try:
        sources = db.execute(
            select(DBSource)
            .where(DBSource.active.is_(True))
            .order_by(DBSource.name.asc())
        ).scalars().all()

        items = [{"id": source.id, "name": source.name, "url": source.url} for source in sources]
        return {"items": items}
    finally:
        db.close()
