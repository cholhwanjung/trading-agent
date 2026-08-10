"""US 공시·재무 채널 — SEC EDGAR (무료, API 키 불필요 · User-Agent 헤더만 요구).

어댑터 하나가 US 의 두 공백을 함께 메운다.
  1. **공시 이벤트** — 8-K/10-Q/10-K 접수를 NewsItem 으로 매핑해 기존 US 뉴스 채널에 합류.
     뉴스 헤드라인은 공시의 대체재가 아니다(8-K 는 4영업일 내 의무공시라 적시성·완결성이 다르다).
  2. **재무** — XBRL companyfacts 에서 손익 누적 구간과 재무상태표 시점값을 그대로 뽑는다.
     TTM 산식·비율 계산은 여기서 하지 않는다(순수 모듈이 맡는다).

누출 통제: 상한 판정 기준은 회계기간 종료일이 아니라 **제출일**이다. 분기 종료 후 수 주
뒤에야 공시되므로 종료일로 자르면 그 사이 날짜의 관측에 미래 정보가 섞인다.

공시 원문·MD&A 본문은 주입하지 않는다 — 구조화 필드(보고서 종류·이벤트 코드·수치)만.
개별 종목 실패는 best-effort 로 무시한다(재무 한 종목 때문에 결정 경로를 막지 않는다).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import httpx

from adapters.base import NewsItem
from adapters.financials import BalanceSnapshot, CumulativePeriod, Financials, assemble_financials

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
TIMEOUT = 20.0

# SEC 는 자동 조회에 연락 가능한 식별자를 요구한다(공정 접근 정책). 운영 스크립트가
# .env 로 덮어쓸 수 있고, 미설정이면 이 기본값으로도 조회는 통과한다.
DEFAULT_USER_AGENT = "trading-agent research contact@example.com"

# 티커 → SEC 고유번호(CIK). www.sec.gov/files/company_tickers.json 로 확인(2026-08-10).
# 유니버스 확장 시 여기 추가한다.
TICKER_TO_CIK: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
}

# 관측에 넣을 보고서 종류. 정정(/A)도 같은 사건의 갱신이라 포함한다.
FILING_FORMS = ("8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A")

# 8-K 이벤트 코드 → 뜻. 코드만 넘기면 관측에서 사건 성격이 드러나지 않는다.
ITEM_LABELS: dict[str, str] = {
    "1.01": "주요 계약 체결",
    "1.02": "주요 계약 종료",
    "2.01": "인수·자산매각 완료",
    "2.02": "실적 발표",
    "2.03": "직접 채무 발생",
    "2.05": "사업 철수·구조조정 비용",
    "3.01": "상장폐지 통지",
    "4.01": "감사인 교체",
    "4.02": "기존 재무제표 신뢰불가",
    "5.02": "임원·이사 변동",
    "5.07": "주주총회 의결",
    "7.01": "공정공시",
    "8.01": "기타 사건",
    "9.01": "첨부 재무제표",
}

# XBRL 태그는 회사·업종마다 다르다 — 앞에서부터 먼저 잡히는 것을 쓴다.
DURATION_TAGS: dict[str, tuple[tuple[str, str], ...]] = {
    "net_income": (("NetIncomeLoss", "USD"), ("ProfitLoss", "USD")),
    "eps_diluted": (("EarningsPerShareDiluted", "USD/shares"),),
}
INSTANT_TAGS: dict[str, tuple[tuple[str, str], ...]] = {
    "equity": (
        ("StockholdersEquity", "USD"),
        ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
    ),
    "liabilities": (("Liabilities", "USD"),),
    # 부채총계를 아예 태깅하지 않는 발행인이 있어(실측 확인) 회계 항등식으로 채우기 위한 보조 항목.
    "assets": (("Assets", "USD"), ("LiabilitiesAndStockholdersEquity", "USD")),
    "minority_interest": (("MinorityInterest", "USD"),),
}

MONTHS_ALLOWED = (3, 6, 9, 12)
_DAYS_PER_MONTH = 30.44


def _headers(user_agent: str | None) -> dict[str, str]:
    return {"User-Agent": user_agent or DEFAULT_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _item_text(items: str) -> str:
    """8-K 이벤트 코드 문자열("2.02,9.01") → 읽을 수 있는 라벨. 미등록 코드는 그대로 남긴다."""
    labels = [ITEM_LABELS.get(c.strip(), c.strip()) for c in items.split(",") if c.strip()]
    return ", ".join(labels)


def _to_news(symbol: str, cik: str, recent: dict, index: int) -> NewsItem | None:
    """submissions 의 recent 배열 한 행 → NewsItem. 제출일 없으면 None(방어)."""
    filed = (recent["filingDate"][index] or "").strip()
    if not filed:
        return None
    form = (recent["form"][index] or "").strip()
    label = _item_text((recent.get("items") or [""] * (index + 1))[index] or "")
    accession = (recent["accessionNumber"][index] or "").replace("-", "")
    doc = (recent.get("primaryDocument") or [""] * (index + 1))[index] or ""
    return NewsItem(
        published_at=datetime.strptime(filed, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        headline=f"[공시] {symbol} {form}" + (f" — {label}" if label else ""),
        source="SEC",
        url=ARCHIVE.format(cik=int(cik), accession=accession, doc=doc) if accession else None,
    )


async def fetch_edgar_filings(
    symbols: list[str],
    start: date,
    end: date,
    client: httpx.AsyncClient | None = None,
    user_agent: str | None = None,
    max_items: int = 30,
) -> list[NewsItem]:
    """[start, end] 에 접수된 공시를 NewsItem 으로. CIK 미매핑 종목은 스킵."""

    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, headers=_headers(user_agent))
    out: list[NewsItem] = []
    try:
        for symbol in symbols:
            cik = TICKER_TO_CIK.get(symbol)
            if not cik:
                continue
            try:
                resp = await client.get(SUBMISSIONS.format(cik=cik), headers=_headers(user_agent))
                resp.raise_for_status()
                recent = resp.json()["filings"]["recent"]
            except (httpx.HTTPError, ValueError, KeyError):
                continue  # 개별 종목 실패는 무시 (best-effort)
            for i, form in enumerate(recent.get("form") or []):
                if form not in FILING_FORMS:
                    continue
                item = _to_news(symbol, cik, recent, i)
                if item and start <= item.published_at.date() <= end:
                    out.append(item)
    finally:
        if own:
            await client.aclose()
    out.sort(key=lambda n: n.published_at, reverse=True)
    return out[:max_items]


def _pick_unit(candidates: tuple[tuple[str, str], ...], facts: dict) -> list:
    """태그 후보를 순서대로 훑어 먼저 잡히는 단위 배열을 반환. 전부 없으면 빈 리스트."""
    for tag, unit in candidates:
        rows = (facts.get(tag) or {}).get("units", {}).get(unit)
        if rows:
            return rows
    return []


def _latest_per_key(rows: list, key_fn) -> dict:
    """(구간 키) → (제출일, 값) 중 가장 늦게 제출된 것. 정정공시가 원본을 덮는다."""
    best: dict = {}
    for row in rows:
        if not row.get("filed"):
            continue
        key = key_fn(row)
        if key is None:
            continue
        filed = date.fromisoformat(row["filed"])
        if key not in best or filed >= best[key][0]:
            best[key] = (filed, row["val"])
    return best


def parse_companyfacts(symbol: str, payload: dict, cap: date) -> Financials | None:
    """companyfacts 응답 → 재무 스냅샷. 필요한 태그가 하나도 없으면 None.

    같은 구간에 여러 공시가 있으면 늦게 제출된 값을 쓴다 — 정정공시 반영이며, 제출일
    상한 아래에서 고르므로 그 시점에 알 수 있었던 최신 값이 된다. 지표별로 따로 고른 뒤
    합치므로, 한 지표가 뒤늦게 정정돼도 다른 지표가 밀려나지 않는다.
    """

    facts = payload.get("facts", {}).get("us-gaap", {})
    if not facts:
        return None

    def duration_key(row: dict):
        if "start" not in row:
            return None
        start, end = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
        months = round((end - start).days / _DAYS_PER_MONTH)
        return (months, end) if months in MONTHS_ALLOWED else None

    def instant_key(row: dict):
        return None if "start" in row else date.fromisoformat(row["end"])

    periods: dict[tuple[int, date], CumulativePeriod] = {}
    for metric, candidates in DURATION_TAGS.items():
        for (months, end), (filed, val) in _latest_per_key(
            _pick_unit(candidates, facts), duration_key
        ).items():
            prev = periods.get((months, end))
            values = {**prev.values, metric: val} if prev else {metric: val}
            filed = max(filed, prev.filed) if prev else filed
            periods[(months, end)] = CumulativePeriod(months, end, filed, values)

    balances: dict[date, BalanceSnapshot] = {}
    for metric, candidates in INSTANT_TAGS.items():
        for end, (filed, val) in _latest_per_key(
            _pick_unit(candidates, facts), instant_key
        ).items():
            prev = balances.get(end)
            values = {**prev.values, metric: val} if prev else {metric: val}
            balances[end] = BalanceSnapshot(end, max(filed, prev.filed) if prev else filed, values)

    return assemble_financials(
        symbol, list(periods.values()), [_fill_liabilities(b) for b in balances.values()], cap
    )


def _fill_liabilities(balance: BalanceSnapshot) -> BalanceSnapshot:
    """부채총계 미태깅 발행인을 회계 항등식(자산 = 부채 + 자본)으로 보완.

    비지배지분이 따로 잡혀 있으면 자본 값이 모회사 몫인지 전체인지 구분할 수 없으므로
    보완하지 않는다 — 근사치를 채우느니 값을 비워 두는 편이 낫다.
    """
    v = balance.values
    if "liabilities" in v or "minority_interest" in v or not {"assets", "equity"} <= v.keys():
        return balance
    return BalanceSnapshot(
        balance.end, balance.filed, {**v, "liabilities": v["assets"] - v["equity"]}
    )


async def fetch_edgar_financials(
    symbols: list[str],
    cap: date,
    client: httpx.AsyncClient | None = None,
    user_agent: str | None = None,
) -> dict[str, Financials]:
    """유니버스 종목의 재무 스냅샷. cap 은 공시 제출일 상한(관측 상한 t-1)."""

    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, headers=_headers(user_agent))
    out: dict[str, Financials] = {}
    try:
        for symbol in symbols:
            cik = TICKER_TO_CIK.get(symbol)
            if not cik:
                continue
            try:
                resp = await client.get(COMPANYFACTS.format(cik=cik), headers=_headers(user_agent))
                resp.raise_for_status()
                payload = resp.json()
            except (httpx.HTTPError, ValueError):
                continue  # 개별 종목 실패는 무시 (best-effort)
            fin = parse_companyfacts(symbol, payload, cap)
            if fin:
                out[symbol] = fin
    finally:
        if own:
            await client.aclose()
    return out
