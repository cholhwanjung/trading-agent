"""KR 종목·시장 뉴스 — Google News RSS (무료, 키 불필요).

KR 유니버스 종목별 회사명 + 시장 레벨 질의("코스피")로 Google News RSS 를 질의해
뉴스 관측 창(최근 N캘린더일)으로 필터한다. 시장 질의는 종목명이 못 잡는 지수 급변·
서킷브레이커·규제·정책 뉴스용 — 2026-07 KOSPI 급락기의 규제 조치(레버리지 ETF 제한
등)가 종목 질의만으로는 관측에 전혀 들어오지 않았다.

품질 필터(결정론): ① 발행처 블록리스트 — Google News 제목 꼬리(" - 발행처")가
블로그·카페·동영상이면 버림(종목명 질의에 생활 블로그 글이 섞여 들어온 실사례).
② 헤드라인 중복 제거 — 같은 기사가 여러 질의에 걸린다.

RSS 2.0 파싱은 크립토 뉴스와 동일 파서(parse_rss)를 재사용한다. 개별 피드 실패는
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

# 시장 레벨 질의 — 지수 급변·서킷브레이커·규제·정책 뉴스는 종목명 질의에 안 걸린다.
MARKET_QUERIES: tuple[str, ...] = ("코스피",)

# 언론 기사가 아닌 발행처(블로그·카페·개인 유료 콘텐츠·동영상) 차단 — 제목 꼬리 소문자 비교
_BLOCKED_PUBLISHERS = (
    "블로그", "blog", "티스토리", "tistory", "브런치", "카페", "youtube", "프리미엄콘텐츠",
)


def _is_editorial(headline: str) -> bool:
    """Google News 제목 꼬리의 발행처가 블로그류면 False — 언론 기사만 통과."""
    if " - " not in headline:
        return True  # 발행처 표기 없는 제목은 통과 (필터는 확실한 비기사만 제거)
    publisher = headline.rsplit(" - ", 1)[-1].lower()
    return not any(b in publisher for b in _BLOCKED_PUBLISHERS)


def _dedup_key(headline: str) -> str:
    """발행처 꼬리를 뗀 본문 제목의 정규화 키 — 언론사만 다른 재발행 기사를 묶는다.

    첫 " - " 앞부분만 취한다 — 일부 언론사는 제목에 자기 이름을 이미 붙여 발행처
    꼬리가 이중으로 붙는다("제목 - 조선비즈 - Chosunbiz"). 키 용도라 과절단 무해.
    """
    return "".join(ch for ch in headline.split(" - ")[0].lower() if ch.isalnum())


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
    """KR 종목명별 + 시장 레벨 Google News RSS 를 모아 [start, end] 헤드라인만 최신순 반환.

    발행처 블록리스트·헤드라인 중복 제거를 거친 뒤 창 필터·정렬·상한을 적용한다.
    """
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True)
    collected: list[NewsItem] = []
    queries = [n for n in (KR_STOCK_NAMES.get(s) for s in symbols) if n]  # 미매핑은 요청 안 함
    queries += list(MARKET_QUERIES)
    try:
        for query in queries:
            try:
                resp = await client.get(_feed_url(query))
                resp.raise_for_status()
            except httpx.HTTPError:
                continue  # 개별 피드 실패는 무시
            collected.extend(parse_rss(resp.text, f"google-news:{query}"))
    finally:
        if own_client:
            await client.aclose()

    seen: set[str] = set()
    unique = []
    for n in collected:
        key = _dedup_key(n.headline)
        if key not in seen and _is_editorial(n.headline):
            seen.add(key)
            unique.append(n)
    in_window = [n for n in unique if start <= n.published_at.date() <= end]
    in_window.sort(key=lambda n: n.published_at, reverse=True)
    return in_window[:max_items]
