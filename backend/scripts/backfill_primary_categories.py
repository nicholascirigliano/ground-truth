#!/usr/bin/env python3
from datetime import UTC, datetime
import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from app.db.models import Article as DBArticle
from app.db.models import Summary as DBSummary
from app.db.session import SessionLocal

ALLOWED_CATEGORIES = {
    "models",
    "research",
    "products",
    "open_source",
    "hardware",
    "regulation",
}
FALLBACK_CATEGORY = "uncategorized"
MODEL_NAME = "gpt-4.1-mini"


def _normalize_category(value: str | None) -> str:
    if not value:
        return FALLBACK_CATEGORY
    normalized = value.strip().lower()
    if normalized in ALLOWED_CATEGORIES:
        return normalized
    return FALLBACK_CATEGORY


def classify_article(
    *,
    title: str,
    source_name: str,
    canonical_url: str | None,
    summary_text: str | None,
    legacy_category: str | None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.")
        return FALLBACK_CATEGORY

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a classifier for AI news categories. "
        "Choose exactly one category from this list and output only the value: "
        "models, research, products, open_source, hardware, regulation. "
        "Do not output any other text or punctuation. "
        "If unsure, pick the closest fit."
    )

    user_lines = [
        f"Title: {title}",
        f"Source: {source_name}",
    ]
    if summary_text:
        user_lines.append(f"Summary: {summary_text}")
    if canonical_url:
        user_lines.append(f"URL: {canonical_url}")
    if legacy_category:
        user_lines.append(f"Legacy Category: {legacy_category}")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
        )
    except Exception as exc:
        print(f"Warning: classification failed for {title}: {exc}")
        return FALLBACK_CATEGORY

    if not completion.choices:
        return FALLBACK_CATEGORY

    message = completion.choices[0].message
    if message is None or message.content is None:
        return FALLBACK_CATEGORY

    return _normalize_category(message.content)


def count_missing_primary_categories(db) -> int:
    return db.execute(
        select(func.count(DBArticle.id))
        .select_from(DBArticle)
        .join(DBSummary, DBSummary.article_id == DBArticle.id)
        .where(DBArticle.primary_category.is_(None))
    ).scalar_one()


def fetch_batch(db, batch_size: int, last_id: str | None) -> list[DBArticle]:
    query = (
        select(DBArticle)
        .join(DBSummary, DBSummary.article_id == DBArticle.id)
        .options(joinedload(DBArticle.source), joinedload(DBArticle.summary))
        .where(DBArticle.primary_category.is_(None))
        .order_by(DBArticle.id.asc())
        .limit(batch_size)
    )
    if last_id is not None:
        query = query.where(DBArticle.id > last_id)

    return db.execute(query).scalars().all()


def backfill(batch_size: int, limit: int | None) -> None:
    db = SessionLocal()
    try:
        total_missing = count_missing_primary_categories(db)
        if total_missing == 0:
            print("No missing primary categories found.")
            return

        target_total = total_missing if limit is None else min(total_missing, limit)
        processed = 0
        fallback_count = 0
        last_id = None

        while processed < target_total:
            remaining = target_total - processed
            batch = fetch_batch(db, min(batch_size, remaining), last_id)
            if not batch:
                break

            for article in batch:
                if processed >= target_total:
                    break

                processed += 1
                last_id = article.id

                try:
                    if article.primary_category is not None:
                        continue

                    assigned = classify_article(
                        title=article.title,
                        source_name=article.source.name,
                        canonical_url=article.canonical_url,
                        summary_text=article.summary.summary_text if article.summary else None,
                        legacy_category=article.category,
                    )
                    if assigned == FALLBACK_CATEGORY:
                        fallback_count += 1

                    article.primary_category = assigned
                    db.add(article)
                    db.commit()
                    print(f"{article.id} -> {assigned}")
                except Exception as exc:
                    db.rollback()
                    fallback_count += 1
                    article.primary_category = FALLBACK_CATEGORY
                    try:
                        db.add(article)
                        db.commit()
                        print(f"{article.id} -> {FALLBACK_CATEGORY} (error: {exc})")
                    except Exception:
                        db.rollback()
                        print(f"Warning: failed to persist fallback for {article.id}: {exc}")

        print(
            f"Done. Classified {processed}/{target_total}; "
            f"fallbacks: {fallback_count}."
        )
    finally:
        db.close()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Backfill primary categories.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of articles to process per batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of articles to process in this run.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0")

    backfill(batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
