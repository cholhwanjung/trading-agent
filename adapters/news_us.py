"""US 종목·시장 뉴스 — Alpaca 뉴스 API (무료 원천, 관측 보조).

Alpaca 페이퍼 키로 /v1beta1/news 를 질의해 관측 윈도우 [start, end] 헤드라인만 반환한다.
데이터·체결 venue 분리: US 체결이 KIS 실계좌로 가도 뉴스는 이 채널을 공유한다(KIS 해외는
자체 무료 뉴스 원천이 없음). 키 없음/실패는 best-effort 로 빈 리스트.

질의는 두 갈래다 — 유니버스 종목 태그와 시장 레벨 태그. 원천의 뉴스는 종목 태그로만
색인돼 있어 지수 ETF 에는 한 건도 붙지 않는다(실측: 창을 바꿔가며 SCHX 질의 4회, 매번
0건). 물가·고용 지표 발표, 연준 발언, 관세·지수 브레드스 같은 **지수를 움직이는 재료가
종목 질의에는 전혀 걸리지 않는다** — 지수 ETF 가 배분 대상인데 그 판단 재료만 관측에
없는 상태였다. 원천이 이런 기사에 붙이는 태그가 대표 지수 ETF 라, 그 티커를 질의어로
쓴다(MARKET_QUERY). 보유·관측 대상이 아니라 색인 키다.

두 결과는 그룹으로 나눠 라운드로빈으로 섞는다. 그냥 이어붙이면 소비자가 앞에서 잘라
쓰는 순간 뒤에 붙인 시장 질의가 통째로 사라진다. 그룹은 유니버스 종목별 + 시장이며,
유니버스 종목 태그가 하나라도 붙은 기사는 종목 그룹으로 간다 — 시장 그룹에는 어떤
보유 종목에도 안 걸리는 기사만 남아, 좁은 슬롯이 종목 기사 재탕으로 채워지지 않는다.

무료·일간 해상도 데이터만 쓰는 정책에 맞는 원천 — 유료 뉴스 API 는 쓰지 않는다.
"""

from __future__ import annotations

from datetime import date, datetime

import httpx

from adapters.base import NewsItem, round_robin_news

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
TIMEOUT = 15.0

# 시장 레벨 질의 키 — 원천이 거시·지수 기사에 붙이는 티커. 유니버스 종목이 아니다.
MARKET_QUERY = "SPY"


def _to_item(n: dict) -> NewsItem:
    """Alpaca news 레코드 → NewsItem."""
    return NewsItem(
        published_at=datetime.fromisoformat(n["created_at"].replace("Z", "+00:00")),
        headline=n.get("headline", ""),
        source=n.get("source") or "alpaca",
        url=n.get("url"),
    )


def _parse_news(payload: dict, start: date, end: date) -> list[NewsItem]:
    """Alpaca news 응답 → [start, end] NewsItem (윈도우 재확인)."""
    items = [_to_item(n) for n in payload.get("news") or []]
    return [n for n in items if start <= n.published_at.date() <= end]


def _group_of(record: dict, symbols: list[str]) -> str:
    """기사를 담을 그룹 하나 — 유니버스 순서상 처음 걸리는 종목, 없으면 시장."""
    tagged = set(record.get("symbols") or ())
    return next((s for s in symbols if s in tagged), MARKET_QUERY)


def _merge(records: list[dict], symbols: list[str], start: date, end: date,
           max_items: int) -> list[NewsItem]:
    """두 질의 결과를 종목·시장 그룹으로 나눠 라운드로빈. 같은 기사(id)는 한 번만."""
    groups: dict[str, list[NewsItem]] = {s: [] for s in [*symbols, MARKET_QUERY]}
    seen: set[object] = set()
    for record in sorted(records, key=lambda n: n.get("created_at") or "", reverse=True):
        key = record.get("id")
        if key is not None:
            if key in seen:
                continue  # 두 질의에 함께 걸린 기사
            seen.add(key)
        for item in _parse_news({"news": [record]}, start, end):
            groups[_group_of(record, symbols)].append(item)
    return round_robin_news(list(groups.values()), max_items)


async def _query(client: httpx.AsyncClient, symbols: list[str], start: date, end: date,
                 limit: int) -> list[dict]:
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
    return resp.json().get("news") or []


async def fetch_us_news(
    api_key: str | None,
    secret: str | None,
    symbols: list[str],
    start: date,
    end: date,
    limit: int = 50,
    max_items: int = 30,
) -> list[NewsItem]:
    """Alpaca 뉴스 [start, end]. 키 없음/실패 시 빈 리스트(best-effort 관측 보조).

    반환 순서는 시각순이 아니라 종목·시장 그룹의 라운드로빈이다(모듈 docstring 참조).
    """
    if not (api_key and secret):
        return []
    async with httpx.AsyncClient(
        headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}, timeout=TIMEOUT
    ) as client:
        records: list[dict] = []
        for query in (symbols, [MARKET_QUERY]):
            try:
                records += await _query(client, query, start, end, limit)
            except httpx.HTTPError:
                continue  # 한쪽 질의 실패가 다른 쪽 채널을 막지 않는다
    return _merge(records, symbols, start, end, max_items)
