"""US 종목 뉴스 — Alpaca 뉴스 API (무료 원천, 관측 보조).

Alpaca 페이퍼 키로 /v1beta1/news 를 질의해 관측 윈도우 [start, end] 헤드라인만 반환한다.
데이터·체결 venue 분리: US 체결이 KIS 실계좌로 가도 뉴스는 이 채널을 공유한다(KIS 해외는
자체 무료 뉴스 원천이 없음). 키 없음/실패는 best-effort 로 빈 리스트.
"""

from __future__ import annotations

from datetime import date, datetime

import httpx

from adapters.base import NewsItem

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
TIMEOUT = 15.0


def _parse_news(payload: dict, start: date, end: date) -> list[NewsItem]:
    """Alpaca news 응답 → [start, end] NewsItem (윈도우 재확인)."""
    items = []
    for n in payload.get("news") or []:
        published = datetime.fromisoformat(n["created_at"].replace("Z", "+00:00"))
        if start <= published.date() <= end:
            items.append(
                NewsItem(
                    published_at=published,
                    headline=n.get("headline", ""),
                    source=n.get("source") or "alpaca",
                    url=n.get("url"),
                )
            )
    return items


async def fetch_us_news(
    api_key: str | None,
    secret: str | None,
    symbols: list[str],
    start: date,
    end: date,
    limit: int = 50,
) -> list[NewsItem]:
    """Alpaca 뉴스 [start, end]. 키 없음/실패 시 빈 리스트(best-effort 관측 보조)."""
    if not (api_key and secret):
        return []
    async with httpx.AsyncClient(
        headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}, timeout=TIMEOUT
    ) as client:
        try:
            resp = await client.get(
                NEWS_URL,
                params={
                    "symbols": ",".join(symbols),
                    "start": f"{start.isoformat()}T00:00:00Z",
                    "end": f"{end.isoformat()}T23:59:59Z",
                    "limit": limit,
                },
            )
            resp.raise_for_status()
            return _parse_news(resp.json(), start, end)
        except httpx.HTTPError:
            return []
