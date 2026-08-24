"""시장 대표 지수 일간 종가 — 벤치마크 표시 전용 채널.

B&H arm 은 에이전트가 매매할 수 있는 **유니버스**를 균등 보유한 것이라, 그 대비 델타는
유니버스를 고정한 채 결정 스킬만 분리한다. 반면 이 채널이 답하는 것은 다른 질문이다 —
"그 기간이 애초에 어떤 장이었나". 둘은 대체재가 아니라 서로 다른 축이고, 승격 판정은
계속 B&H 대비로만 한다(이 채널은 맥락).

원천은 지수를 추종하는 ETF 가 아니라 **지수 그 자체**다. 추종 ETF 는 이미 유니버스
안에 있어(KR 278530 · US SCHX) 그것과 비교하면 에이전트가 가진 것과 거의 같은 것을
비교하게 된다.

| 시장 | 지수 | 원천 |
|---|---|---|
| KR | KOSPI | KIS 국내업종 일자별지수 (FHKUP03500100 · 업종코드 0001) |
| US | NASDAQ Composite | FRED NASDAQCOM |
| CRYPTO | — | 무료 실지수 원천 없음. 대장 코인(BTC)은 지수가 아니라 바스켓 구성종목이다 |

상한은 t−1 (당일 종가 차단). 조회 실패는 비치명 — None 을 돌려 호출부가 건너뛴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from adapters.base import Bar

#: KIS 국내업종 일자별지수 — 1회 응답 50행(주식 일봉의 100행보다 짧다)
KIS_INDEX_TR = "FHKUP03500100"
KIS_INDEX_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
KOSPI_ISCD = "0001"

#: 시장 → (지수 표시명, 원천 라벨). 없는 시장은 벤치마크 없음.
INDEX_NAME = {"KR": "KOSPI", "US": "NASDAQ Composite"}
INDEX_SOURCE = {"KR": f"kis:{KOSPI_ISCD}", "US": "fred:NASDAQCOM"}


@dataclass(frozen=True)
class IndexSeries:
    """한 시장의 지수 종가 시계열. name·source 는 값의 출처를 드러내는 필수 라벨이다."""

    market: str
    name: str
    source: str
    closes: list[tuple[date, float]]  # 오름차순


def _parse_kis_index(rows: list[dict], start: date, end: date) -> list[Bar]:
    """국내업종 응답 → Bar 오름차순. 창 밖 행은 버린다(상한 t−1 재확인)."""
    bars: list[Bar] = []
    for r in rows:
        try:
            day = date(
                int(r["stck_bsop_date"][:4]),
                int(r["stck_bsop_date"][4:6]),
                int(r["stck_bsop_date"][6:8]),
            )
            close = float(r["bstp_nmix_prpr"])
        except (KeyError, ValueError):
            continue
        if start <= day <= end:
            # 지수는 거래량 개념이 약해 종가만 쓴다 — Bar 는 페이지네이션 헬퍼의 규약일 뿐.
            bars.append(Bar(day=day, open=close, high=close, low=close, close=close, volume=0.0))
    return sorted(bars, key=lambda b: b.day)


async def _fetch_kospi(session, start: date, end: date) -> list[tuple[date, float]]:
    """KOSPI 일간 종가. 50행 상한을 주식 일봉과 같은 커서 페이지네이션으로 넘는다."""
    from adapters.kis import paginate_daily

    async def page(cursor: date) -> list[Bar]:
        data = await session.get(
            KIS_INDEX_PATH,
            tr_id=KIS_INDEX_TR,
            params={
                "FID_COND_MRKT_DIV_CODE": "U",  # U = 업종/지수
                "FID_INPUT_ISCD": KOSPI_ISCD,
                "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": cursor.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": "D",
            },
        )
        return _parse_kis_index(data.get("output2") or [], start, cursor)

    return [(b.day, b.close) for b in await paginate_daily(page, start, end)]


async def fetch_index_series(
    market: str,
    asof_day: date,
    lookback_days: int,
    env: dict[str, str] | None = None,
    kis_session=None,
) -> IndexSeries | None:
    """시장 대표 지수 종가 시계열. 벤치마크 없음/키 없음/조회 실패 → None (fail-open).

    상한은 asof_day−1 — 관측 채널과 같은 t−1 규약을 표시 채널에도 그대로 적용한다.
    """
    name = INDEX_NAME.get(market)
    if name is None:
        return None
    end = asof_day - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    closes: list[tuple[date, float]] = []
    try:
        if market == "KR":
            if kis_session is None:
                return None
            closes = await _fetch_kospi(kis_session, start, end)
        elif market == "US":
            api_key = (env or {}).get("FRED_API_KEY")
            if not api_key:
                return None
            from adapters.fred import fetch_fred_history

            closes = await fetch_fred_history(api_key, "NASDAQCOM", start, end)
    except Exception:
        return None
    if not closes:
        return None
    return IndexSeries(market, name, INDEX_SOURCE[market], closes)
