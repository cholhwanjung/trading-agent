"""KR 종목 뉴스 — Google News RSS (무료, 키 불필요).

KR 유니버스 종목별 회사명으로 Google News RSS 를 질의해 관측 윈도우 [t-3, t-1] 로 필터한다.
RSS 2.0 파싱은 크립토 뉴스와 동일 파서(parse_rss)를 재사용한다. 개별 종목 피드 실패는
조용히 건너뛴다(뉴스는 best-effort 관측 보조).

무료·일간 해상도 데이터만 쓰는 정책에 맞는 원천 — 유료 스크래핑/검색 API 는 쓰지 않는다.
"""

from __future__ import annotations

import urllib.parse
from datetime import date

import httpx

from adapters.base import NewsItem
from adapters.news_rss import parse_rss

# 종목코드 → 질의용 회사명(현 KR 유니버스). 유니버스 확장 시 여기 추가.
KR_STOCK_NAMES: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "035420": "NAVER",
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"  # 무료·키 불필요
TIMEOUT = 10.0


def _feed_url(query: str) -> str:
    """회사명 → ko/KR 로케일 Google News RSS 검색 URL."""
    params = urllib.parse.urlencode({"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"})
    return f"{GOOGLE_NEWS_RSS}?{params}"


async def fetch_kr_news(
    symbols: list[str],
    start: date,
    end: date,
    client: httpx.AsyncClient | None = None,
    max_items: int = 30,
) -> list[NewsItem]:
    """KR 종목명별 Google News RSS 를 모아 [start, end] 헤드라인만 최신순으로 반환."""
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)
    collected: list[NewsItem] = []
    try:
        for symbol in symbols:
            name = KR_STOCK_NAMES.get(symbol)
            if not name:
                continue  # 이름 미매핑 종목은 건너뜀 (요청도 안 함)
            try:
                resp = await client.get(_feed_url(name))
                resp.raise_for_status()
            except httpx.HTTPError:
                continue  # 개별 종목 피드 실패는 무시
            collected.extend(parse_rss(resp.text, f"google-news:{name}"))
    finally:
        if own_client:
            await client.aclose()

    in_window = [n for n in collected if start <= n.published_at.date() <= end]
    in_window.sort(key=lambda n: n.published_at, reverse=True)
    return in_window[:max_items]
