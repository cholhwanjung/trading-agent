"""크립토 뉴스 — 무료 RSS 피드 ([ADR-011]: 무료 데이터만, 키 불필요).

시장 전반 헤드라인(심볼 무관)을 관측 윈도우 [t-3, t-1] 로 필터해 반환.
피드 개별 실패는 조용히 건너뛴다(뉴스는 best-effort 관측 보조) — 단 전체 실패도
빈 리스트일 뿐 예외는 아니다. 파싱은 stdlib(xml.etree + email.utils)로 충분.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date, timezone
from email.utils import parsedate_to_datetime

import httpx

from adapters.base import NewsItem

CRYPTO_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]
TIMEOUT = 10.0


def parse_rss(xml_text: str, source: str) -> list[NewsItem]:
    """RSS 2.0 <item> → NewsItem. pubDate 없는/깨진 아이템은 버린다."""
    items: list[NewsItem] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate")
        if not title or not pub:
            continue
        try:
            published = parsedate_to_datetime(pub)
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        items.append(
            NewsItem(
                published_at=published.astimezone(timezone.utc),
                headline=title,
                source=source,
                url=(item.findtext("link") or "").strip() or None,
            )
        )
    return items


async def fetch_rss_news(
    start: date,
    end: date,
    feeds: list[str] | None = None,
    client: httpx.AsyncClient | None = None,
    max_items: int = 30,
) -> list[NewsItem]:
    """피드들을 모아 [start, end] 구간의 헤드라인만 최신순으로 반환."""
    feeds = feeds or CRYPTO_FEEDS
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)
    collected: list[NewsItem] = []
    try:
        for url in feeds:
            source = httpx.URL(url).host or url
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError:
                continue  # 개별 피드 실패는 건너뛴다
            collected.extend(parse_rss(resp.text, source))
    finally:
        if own_client:
            await client.aclose()

    in_window = [n for n in collected if start <= n.published_at.date() <= end]
    in_window.sort(key=lambda n: n.published_at, reverse=True)
    return in_window[:max_items]
