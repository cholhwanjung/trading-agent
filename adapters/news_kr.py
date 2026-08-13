"""KR 종목·시장 뉴스 — Google News RSS (무료, 키 불필요).

KR 유니버스 종목별 회사명 + 시장 레벨 질의("코스피")로 Google News RSS 를 질의해
뉴스 관측 창(최근 N캘린더일)으로 필터한다. 시장 질의는 종목명이 못 잡는 지수 급변·
서킷브레이커·규제·정책 뉴스용 — 2026-07 KOSPI 급락기의 규제 조치(레버리지 ETF 제한
등)가 종목 질의만으로는 관측에 전혀 들어오지 않았다.

품질 필터(결정론): ① 발행처 블록리스트 — Google News 제목 꼬리(" - 발행처")가
블로그·카페·동영상이면 버림(종목명 질의에 생활 블로그 글이 섞여 들어온 실사례).
② 헤드라인 중복 제거 — 같은 기사가 여러 질의에 걸린다. ③ 재작성 기사 제거 — 같은
사건을 언론사마다 다르게 쓴 것들(제목이 달라 ②를 통과한다). ④ 질의별 라운드로빈.

③④ 가 필요한 이유: 소비자는 이 리스트를 앞에서 자르는데, 접수·발행 시각순으로만
정렬하면 그날 기사가 많은 종목이 창을 독식한다. 저장된 관측 19일을 세어 보니 상위 10
슬롯 190개의 점유가 종목별 62/53/40/29 인 반면 **시장 질의("코스피")는 6개**였다 —
지수·규제·정책 뉴스를 위해 둔 채널이 사실상 닫혀 있었다. 한 종목 안에서도 같은 사건의
재작성본이 슬롯을 겹쳐 먹는다(레드닷 수상 4건·분기 실적 4건이 한 창에 동시 등장).

RSS 2.0 파싱은 크립토 뉴스와 동일 파서(parse_rss)를 재사용한다. 개별 피드 실패는
조용히 건너뛴다(뉴스는 best-effort 관측 보조).

무료·일간 해상도 데이터만 쓰는 정책에 맞는 원천 — 유료 스크래핑/검색 API 는 쓰지 않는다.
"""

from __future__ import annotations

import urllib.parse
from datetime import date

import httpx

from adapters.base import NewsItem, round_robin_news
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


# 재작성 기사 판정 임계 — 질의어를 뺀 제목의 문자 bigram Jaccard.
#
# **한글 전용 수치다.** 한글은 음절 하나가 문자 하나라 bigram 이 곧 어절 조각이지만,
# 영문은 흔한 알파벳 쌍(" th"·"he "·"s ")이 겹쳐 무관한 기사도 높게 나온다 — 같은 산식을
# 영문 헤드라인에 대보면 서로 다른 사건이 0.29 까지 올라간다. 다른 시장 뉴스에 재사용 금지.
#
# 저장된 관측 19일(상위 10슬롯 쌍 855개)로 보정했고, **경계는 깨끗하지 않다**. 같은 사건의
# 재작성본은 0.11 까지 내려오는데, 서로 다른 사건인데도 0.179 까지 올라오는 쌍이 있다
# (같은 칼럼 연재의 말머리를 공유하는 자동차 시승기들). 겹치는 구간에서는 **덜 자르는 쪽**을
# 택했다 — 놓친 중복은 슬롯 하나를 낭비할 뿐이지만 잘못 묶으면 사건이 관측에서 통째로
# 사라진다. 0.20 은 실측된 최고 오탐(0.179) 바로 위이고 이 표본에서 오탐 0건이다.
_REWRITE_SIMILARITY = 0.20


def _rewrite_key(headline: str, terms: list[str]) -> str:
    """재작성 판정용 정규화 — 발행처 꼬리와 **질의어**를 뺀 본문의 영숫자만.

    질의어(회사명)는 그 질의로 받은 모든 제목에 들어 있어 변별력이 0인데, 짧은 제목에서는
    겹치는 bigram 의 큰 몫을 차지해 무관한 기사를 묶어버린다 — "삼성전자, HBF 표준 규격
    공개" 와 "삼성전자 HBM4 양산 개시" 가 0.278 로 붙었고, 질의어를 빼면 0.071 이 된다.
    """
    body = headline.split(" - ")[0]
    for term in terms:
        body = body.replace(term, "")
    return "".join(ch for ch in body.lower() if ch.isalnum())


def _bigrams(key: str) -> frozenset[str]:
    """정규화 제목의 문자 bigram 집합. 1글자 이하는 그 자체를 원소로."""
    return frozenset(key[i : i + 2] for i in range(len(key) - 1)) or frozenset({key})


def _is_rewrite(bigrams: frozenset[str], accepted: list[frozenset[str]]) -> bool:
    """이미 채택된 제목 중 하나와 임계 이상 겹치면 True(같은 사건의 재작성본)."""
    return any(
        len(bigrams & prev) / len(bigrams | prev) >= _REWRITE_SIMILARITY for prev in accepted
    )


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
    """KR 종목명별 + 시장 레벨 Google News RSS 를 모아 [start, end] 헤드라인을 반환.

    발행처 블록리스트 → 창 필터 → 최신순 정렬 → 중복·재작성 제거 → 질의별 라운드로빈.
    중복 제거를 정렬 뒤에 두는 이유는 같은 사건이면 **가장 최근 판**이 남게 하기 위함이다.
    반환 순서는 질의 간 시각순이 아니라 라운드로빈 순이다(모듈 docstring 참조).
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

    in_window = [
        n
        for n in collected
        if _is_editorial(n.headline) and start <= n.published_at.date() <= end
    ]
    in_window.sort(key=lambda n: n.published_at, reverse=True)

    seen: set[str] = set()
    accepted: list[frozenset[str]] = []
    groups: dict[str, list[NewsItem]] = {f"google-news:{q}": [] for q in queries}
    for n in in_window:
        key = _dedup_key(n.headline)
        if key in seen:
            continue
        bigrams = _bigrams(_rewrite_key(n.headline, queries))
        if _is_rewrite(bigrams, accepted):
            continue
        seen.add(key)
        accepted.append(bigrams)
        groups[n.source].append(n)
    return round_robin_news(list(groups.values()), max_items)
