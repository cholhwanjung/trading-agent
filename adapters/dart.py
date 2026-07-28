"""KR 공시 이벤트 — DART OpenAPI 최근 공시 (무료). 관측 뉴스 창(최근 N캘린더일)에 편입.

유니버스 종목의 접수 공시를 '구조화 이벤트'(보고서명 + 접수일)로 NewsItem 에 매핑해
KR 관측에 뉴스와 함께 넣는다 — rich 리포트 본문은 주입하지 않는다(정예 관측 정책).
조회는 종목코드가 아니라 DART 고유번호(corp_code)로 하므로 유니버스별 매핑을 둔다
(유니버스 확장 시 여기 추가 — corpCode.xml 로 확인). 누출 통제: 접수일(rcept_dt) 상한은
호출부가 넘기는 관측 윈도우 end(t-1). 개별 종목 실패는 best-effort 로 무시한다.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import httpx

from adapters.base import NewsItem

DART_LIST = "https://opendart.fss.or.kr/api/list.json"
DART_VIEWER = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo="
TIMEOUT = 10.0

# 종목코드 → DART 고유번호(corp_code). DART corpCode.xml 로 확인(2026-07-26).
STOCK_TO_CORP: dict[str, str] = {
    "005930": "00126380",  # 삼성전자
    "000660": "00164779",  # SK하이닉스
    "005380": "00164742",  # 현대차(현대자동차)
    "035420": "00266961",  # NAVER
}


def _to_news(row: dict) -> NewsItem | None:
    """DART 공시 행 → NewsItem. 접수일 없으면 None(방어)."""
    rcept = (row.get("rcept_dt") or "").strip()
    if not rcept:
        return None
    day = datetime.strptime(rcept, "%Y%m%d").replace(tzinfo=timezone.utc)
    corp = (row.get("corp_name") or "").strip()
    report = (row.get("report_nm") or "").strip()
    rcpno = (row.get("rcept_no") or "").strip()
    return NewsItem(
        published_at=day,
        headline=f"[공시] {corp} {report}".strip(),
        source="DART",
        url=(DART_VIEWER + rcpno) if rcpno else None,
    )


async def fetch_dart_disclosures(
    api_key: str,
    symbols: list[str],
    start: date,
    end: date,
    client: httpx.AsyncClient | None = None,
    max_items: int = 30,
) -> list[NewsItem]:
    """유니버스 종목의 [start, end] 접수 공시를 NewsItem 으로. corp_code 미매핑 종목은 스킵."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    out: list[NewsItem] = []
    try:
        for symbol in symbols:
            corp = STOCK_TO_CORP.get(symbol)
            if not corp:
                continue
            try:
                resp = await client.get(
                    DART_LIST,
                    params={
                        "crtfc_key": api_key,
                        "corp_code": corp,
                        "bgn_de": start.strftime("%Y%m%d"),
                        "end_de": end.strftime("%Y%m%d"),
                        "page_count": "100",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError):
                continue  # 개별 종목 실패는 무시 (best-effort)
            if data.get("status") not in ("000", "013"):  # 013 = 데이터 없음(정상)
                continue
            for row in data.get("list") or []:
                item = _to_news(row)
                if item and start <= item.published_at.date() <= end:
                    out.append(item)
    finally:
        if own:
            await client.aclose()
    out.sort(key=lambda n: n.published_at, reverse=True)
    return out[:max_items]
