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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from app.db.models import Article as DBArticle
from app.db.models import Summary as DBSummary
from app.db.session import SessionLocal

SUMMARY_MODEL = "gpt-4.1-mini"
SUMMARY_VERSION = 1


def generate_summary_text(
    *,
    title: str,
    source_name: str,
    original_url: str,
    category: str,
) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set.")
        return None

    client = OpenAI(api_key=api_key)

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

    completion = client.chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    if not completion.choices:
        return None

    message = completion.choices[0].message
    if message is None or message.content is None:
        return None

    return message.content.strip()


def count_missing_summaries(db) -> int:
    return db.execute(
        select(func.count(DBArticle.id))
        .select_from(DBArticle)
        .outerjoin(DBSummary, DBSummary.article_id == DBArticle.id)
        .where(DBSummary.article_id.is_(None))
    ).scalar_one()


def fetch_missing_batch(db, batch_size: int, last_id: str | None) -> list[DBArticle]:
    query = (
        select(DBArticle)
        .outerjoin(DBSummary, DBSummary.article_id == DBArticle.id)
        .options(joinedload(DBArticle.source))
        .where(DBSummary.article_id.is_(None))
        .order_by(DBArticle.id.asc())
        .limit(batch_size)
    )
    if last_id is not None:
        query = query.where(DBArticle.id > last_id)

    return db.execute(query).scalars().all()


def backfill(batch_size: int, limit: int | None) -> None:
    db = SessionLocal()
    try:
        total_missing = count_missing_summaries(db)
        if total_missing == 0:
            print("No missing summaries found.")
            return

        target_total = total_missing if limit is None else min(total_missing, limit)
        summarized = 0
        processed = 0
        last_id = None

        while processed < target_total:
            remaining = target_total - processed
            batch = fetch_missing_batch(db, min(batch_size, remaining), last_id)
            if not batch:
                break

            for article in batch:
                if processed >= target_total:
                    break

                processed += 1
                last_id = article.id

                try:
                    existing_summary = db.get(DBSummary, article.id)
                    if existing_summary is not None:
                        continue

                    summary_text = generate_summary_text(
                        title=article.title,
                        source_name=article.source.name,
                        original_url=article.canonical_url,
                        category=article.category,
                    )
                    if not summary_text:
                        print(f"Warning: empty summary for article {article.id}")
                        continue

                    db.add(
                        DBSummary(
                            article_id=article.id,
                            summary_text=summary_text,
                            model=SUMMARY_MODEL,
                            version=SUMMARY_VERSION,
                            generated_at=datetime.now(UTC).replace(tzinfo=None),
                        )
                    )
                    db.commit()
                    summarized += 1
                    print(f"Summarized {summarized}/{target_total}")
                except IntegrityError:
                    db.rollback()
                    print(f"Warning: summary already exists for article {article.id}")
                except Exception as exc:
                    db.rollback()
                    print(f"Warning: failed to summarize article {article.id}: {exc}")

        print(f"Done. Summarized {summarized}/{target_total}; processed {processed}.")
    finally:
        db.close()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Backfill missing article summaries.")
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
        help="Maximum number of missing articles to process in this run.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0")

    backfill(batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
