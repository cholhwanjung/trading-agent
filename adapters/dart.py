"""KR 공시·재무 채널 — DART OpenAPI (무료, 키 1개로 두 채널 공용).

두 채널을 별개로 둔다.
  1. **공시 이벤트** — 유니버스 종목의 접수 공시를 '구조화 이벤트'(보고서명 + 접수일)로
     NewsItem 에 매핑해 KR 관측에 뉴스와 함께 넣는다. rich 리포트 본문은 주입하지 않는다.
  2. **재무** — 단일회사 전체 재무제표에서 손익 누적 구간과 재무상태표 시점값을 뽑는다.
     TTM 산식·비율 계산은 여기서 하지 않는다(순수 모듈이 맡는다).

조회는 종목코드가 아니라 DART 고유번호(corp_code)로 하므로 유니버스별 매핑을 둔다
(유니버스 확장 시 여기 추가 — corpCode.xml 로 확인). 누출 통제: 상한 판정 기준은
접수일(공시 제출일)이며 호출부가 넘기는 관측 윈도우 end(t-1)를 넘지 못한다. 회계기간
종료일로 자르면 분기 종료 후 수 주 뒤에야 공시되는 값이 미래 정보로 섞인다.
개별 종목 실패는 best-effort 로 무시한다.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import httpx

from adapters.base import NewsItem
from adapters.financials import BalanceSnapshot, CumulativePeriod, Financials, assemble_financials

DART_LIST = "https://opendart.fss.or.kr/api/list.json"
DART_STATEMENTS = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
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


# ── 재무 ──

# 보고서 코드 → 회계연도 시작부터의 누적 개월 수. 조회 순서는 다루는 기간이 늦은 것부터라
# 첫 성공이 곧 '그 시점에 알 수 있었던 가장 최신 재무'가 된다.
REPORT_MONTHS: dict[str, int] = {"11011": 12, "11014": 9, "11012": 6, "11013": 3}
REPORT_ORDER: tuple[str, ...] = ("11011", "11014", "11012", "11013")

# IFRS 표준 계정 ID 로 뽑는다 — 계정명(당기순이익/분기순이익/반기순이익)은 보고서마다 달라진다.
INCOME_IDS: dict[str, str] = {
    "net_income": "ifrs-full_ProfitLoss",
    "eps_diluted": "ifrs-full_DilutedEarningsLossPerShare",
}
BALANCE_IDS: dict[str, str] = {
    "equity": "ifrs-full_Equity",
    "liabilities": "ifrs-full_Liabilities",
}


def _amount(row: dict, key: str) -> float | None:
    """DART 금액 문자열 → float. 빈 값·'-'·쉼표 표기를 방어한다."""
    raw = (row.get(key) or "").replace(",", "").strip()
    if not raw or raw == "-":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _period_end(year: int, months: int) -> date:
    """회계연도 + 누적 개월 수 → 기간 종료일. 12월 결산(국내 상장사 대부분) 가정.

    DART 응답에 기간 날짜가 없어 코드에서 만들어야 한다. 결산월이 다른 회사면 이 날짜가
    실제와 어긋나지만, TTM 조각들이 모두 같은 방식으로 만들어져 서로 맞물리므로 계산
    결과는 흔들리지 않는다 — 어긋나는 것은 표시용 기간 라벨뿐이다.
    """
    month_end = {3: (3, 31), 6: (6, 30), 9: (9, 30), 12: (12, 31)}[months]
    return date(year, *month_end)


def _parse_statements(rows: list[dict], year: int, months: int, filed: date) -> tuple[list, list]:
    """재무제표 행 → (누적 손익 구간들, 재무상태표 시점값들).

    분기·반기 보고서는 당기 누적과 **전기 동기간 누적**을 함께 담고 있어, 한 번의 조회로
    TTM 산식에 필요한 두 조각이 나온다. 자본총계·순이익은 자본변동표에도 같은 계정 ID 로
    나오므로 손익은 손익계산서, 재무상태는 재무상태표 구분에 한정해 읽는다.

    손익계산서를 따로 내지 않고 포괄손익계산서 하나로 갈음하는 회사가 있어(국내 상장사에
    흔하다) 둘 다 후보로 두되, 손익계산서가 있으면 그쪽만 쓴다 — 두 표의 값이 어긋날 때
    무엇을 읽었는지 모호해지지 않게 한다.
    """
    wanted_ids = set(INCOME_IDS.values())
    income_div = (
        "IS"
        if any(r.get("sj_div") == "IS" and r.get("account_id") in wanted_ids for r in rows)
        else "CIS"
    )
    cumulative: dict[str, float] = {}
    prior_cumulative: dict[str, float] = {}
    balance: dict[str, float] = {}
    for row in rows:
        account = row.get("account_id")
        for metric, wanted in INCOME_IDS.items():
            if account == wanted and row.get("sj_div") == income_div:
                # 연간보고서에는 누적 칸이 없다 — 당기 금액 자체가 1년치다.
                value = _amount(row, "thstrm_add_amount")
                if value is None and months == 12:
                    value = _amount(row, "thstrm_amount")
                prior = _amount(row, "frmtrm_add_amount")
                if prior is None and months == 12:
                    prior = _amount(row, "frmtrm_amount")
                if value is not None:
                    cumulative[metric] = value
                if prior is not None:
                    prior_cumulative[metric] = prior
        for metric, wanted in BALANCE_IDS.items():
            if account == wanted and row.get("sj_div") == "BS":
                value = _amount(row, "thstrm_amount")
                if value is not None:
                    balance[metric] = value

    periods = []
    if cumulative:
        periods.append(CumulativePeriod(months, _period_end(year, months), filed, cumulative))
    if prior_cumulative:
        periods.append(
            CumulativePeriod(months, _period_end(year - 1, months), filed, prior_cumulative)
        )
    balances = [BalanceSnapshot(_period_end(year, months), filed, balance)] if balance else []
    return periods, balances


async def _fetch_statement(
    client: httpx.AsyncClient, api_key: str, corp: str, year: int, code: str
) -> tuple[list[dict], date] | None:
    """(회계연도, 보고서 코드) 재무제표 1건. 미공시·실패는 None. 연결 없으면 별도로 물러선다."""
    for fs_div in ("CFS", "OFS"):
        try:
            resp = await client.get(
                DART_STATEMENTS,
                params={
                    "crtfc_key": api_key,
                    "corp_code": corp,
                    "bsns_year": str(year),
                    "reprt_code": code,
                    "fs_div": fs_div,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        rows = data.get("list") or []
        if data.get("status") == "000" and rows:
            rcept = (rows[0].get("rcept_no") or "")[:8]
            if len(rcept) == 8:
                return rows, datetime.strptime(rcept, "%Y%m%d").date()
    return None


async def fetch_dart_financials(
    api_key: str,
    symbols: list[str],
    cap: date,
    client: httpx.AsyncClient | None = None,
    years_back: int = 3,
) -> dict[str, Financials]:
    """유니버스 종목의 재무 스냅샷. cap 은 공시 접수일 상한(관측 상한 t-1).

    가장 최근 보고서부터 거슬러 올라가며 cap 이내 첫 건을 찾고, 분기·반기면 직전 회계연도
    연간 보고서 1건을 더 받아 TTM 산식 조각을 채운다. 종목당 보통 2~3회 조회.
    """

    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    out: dict[str, Financials] = {}
    try:
        for symbol in symbols:
            corp = STOCK_TO_CORP.get(symbol)
            if not corp:
                continue
            periods: list[CumulativePeriod] = []
            balances: list[BalanceSnapshot] = []
            for year in range(cap.year, cap.year - years_back, -1):
                for code in REPORT_ORDER:
                    months = REPORT_MONTHS[code]
                    if _period_end(year, months) > cap:
                        continue  # 아직 끝나지도 않은 기간
                    found = await _fetch_statement(client, api_key, corp, year, code)
                    if not found or found[1] > cap:
                        continue
                    rows, filed = found
                    periods, balances = _parse_statements(rows, year, months, filed)
                    if months < 12:  # 직전 회계연도 연간 — TTM 산식의 남은 조각
                        prior = await _fetch_statement(client, api_key, corp, year - 1, "11011")
                        if prior and prior[1] <= cap:
                            annual, _ = _parse_statements(prior[0], year - 1, 12, prior[1])
                            periods += annual
                    break
                if periods:
                    break
            fin = assemble_financials(symbol, periods, balances, cap)
            if fin:
                out[symbol] = fin
    finally:
        if own:
            await client.aclose()
    return out
