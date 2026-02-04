import hashlib
import socket
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser


TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
}


def _canonicalize_url(url: str) -> str:
    parts = urlsplit(url)
    kept_params = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.startswith("utm_"):
            continue
        if key in TRACKING_QUERY_PARAMS:
            continue
        kept_params.append((key, value))

    canonical_query = urlencode(kept_params, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, canonical_query, ""))


def canonicalize_url(url: str) -> str:
    return _canonicalize_url(url)


def _to_iso8601(entry: dict) -> str | None:
    if getattr(entry, "published_parsed", None):
        published_dt = datetime(*entry.published_parsed[:6], tzinfo=UTC)
        return published_dt.isoformat().replace("+00:00", "Z")

    if getattr(entry, "updated_parsed", None):
        updated_dt = datetime(*entry.updated_parsed[:6], tzinfo=UTC)
        return updated_dt.isoformat().replace("+00:00", "Z")

    published_text = getattr(entry, "published", None)
    if not published_text:
        published_text = getattr(entry, "updated", None)
    if not published_text:
        return None

    try:
        published_dt = parsedate_to_datetime(published_text)
    except (TypeError, ValueError):
        return None

    if published_dt.tzinfo is None:
        published_dt = published_dt.replace(tzinfo=UTC)
    else:
        published_dt = published_dt.astimezone(UTC)
    return published_dt.isoformat().replace("+00:00", "Z")


def parse_rss_feed(feed_url: str, source: dict, category: str) -> list[dict]:
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(10)
    try:
        parsed_feed = feedparser.parse(feed_url)
    except Exception as exc:
        print(f"Warning: failed to load feed {feed_url}: {exc}")
        return []
    finally:
        socket.setdefaulttimeout(previous_timeout)

    if getattr(parsed_feed, "bozo", False):
        exception = getattr(parsed_feed, "bozo_exception", None)
        print(f"Warning: feed parse issue for {feed_url}: {exception}")

    status = getattr(parsed_feed, "status", None)
    if status is not None and status >= 400:
        print(f"Warning: feed returned HTTP {status} for {feed_url}")
        return []

    items: list[dict] = []

    for entry in parsed_feed.entries:
        title = getattr(entry, "title", None)
        link = getattr(entry, "link", None)
        published_at = _to_iso8601(entry)

        if not title or not link or not published_at:
            continue

        original_url = _canonicalize_url(link)
        identity_source = original_url or title
        article_id = f"art_{hashlib.sha1(identity_source.encode('utf-8')).hexdigest()[:12]}"

        items.append(
            {
                "id": article_id,
                "title": title,
                "category": category,
                "published_at": published_at,
                "summary": "Summary pending.",
                "source": source,
                "original_url": original_url,
            }
        )

    return items
