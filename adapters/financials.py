"""재무 원값 공통 자료형 + TTM 산식 — 순수(네트워크·파일 I/O 없음).

시장별 원천(SEC XBRL · DART)이 서로 다른 스키마로 주는 재무를 같은 형태로 모아
비율 계산 직전까지 처리한다. 원천이 늘어도 여기 산식은 하나만 유지한다.

**TTM 을 '최근 분기 4건 합'으로 구하지 않는다.** 4분기 손익을 별도 구간으로 공시하는
회사가 거의 없기 때문이다 — 연간보고서는 연간 누적만 담는다. 그래서 분기 구간만 모아
합치면 4분기가 빠진 채 한 해 전 분기가 딸려 들어와 **조용히 틀린 값**이 된다(실측한
미국 5개 종목 전부 이 갭이 있었다). 대신 누적(YTD) 값만 쓰는 표준 산식을 쓴다:

    TTM = 최신 누적 + 직전 회계연도 연간 − 직전 회계연도의 같은 길이 누적

세 조각이 기간까지 맞물릴 때만 ttm 으로 인정하고, 아니면 직전 연간값으로 물러선다
(basis 로 어느 쪽인지 항상 라벨한다). 둘 다 불가면 값을 만들지 않는다 — 근거 없는
숫자를 결정 경로에 넣는 것은 값이 없는 것보다 나쁘다.

누출 통제: 모든 자료형이 제출일(filed)을 들고 다닌다. 회계기간 종료일(end)로 자르면
분기 종료 후 수 주 뒤에야 공시되는 값이 그 사이 날짜의 관측에 섞여 미래 정보가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

# 회계 기간 매칭 허용 오차(일). 회계연도 말일은 "9월 마지막 토요일"처럼 요일 기준으로
# 정해지는 곳이 많아 해마다 며칠씩 밀린다 — 정확히 365일을 요구하면 정상 데이터가 탈락한다.
PERIOD_TOLERANCE_DAYS = 20
DAYS_PER_MONTH = 30.44


@dataclass(frozen=True)
class CumulativePeriod:
    """회계연도 시작부터의 누적 손익 1구간. months 는 3·6·9·12 중 하나."""

    months: int
    end: date
    filed: date
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BalanceSnapshot:
    """재무상태표 시점값 1건(자본·부채). 기간이 아니라 특정 시점의 잔액이다."""

    end: date
    filed: date
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Financials:
    """한 종목의 재무 스냅샷 — 비율 계산 입력.

    filed 는 이 스냅샷을 구성한 공시 중 **가장 늦은** 제출일이다. 누출 판정은 항상
    이 값으로 한다(period_end 는 표시·신선도 판단용).
    """

    symbol: str
    filed: date
    period_end: date
    basis: str  # "ttm" | "annual" — 손익 값이 어떻게 나왔는지
    net_income: float | None = None
    eps_diluted: float | None = None
    equity: float | None = None
    liabilities: float | None = None


def _close(a: date, b: date, tolerance: int = PERIOD_TOLERANCE_DAYS) -> bool:
    return abs((a - b).days) <= tolerance


def _shift(day: date, months: float) -> date:
    """day 에서 months 개월 앞선 근사 날짜. 매칭은 항상 허용 오차와 함께 쓴다."""
    return day - timedelta(days=round(months * DAYS_PER_MONTH))


def _select_window(
    periods: list[CumulativePeriod], cap: date
) -> tuple[list[tuple[CumulativePeriod, int]], str] | None:
    """TTM 산출에 쓸 (구간, 부호) 조합과 basis 를 고른다. 불가하면 None.

    cap 은 제출일 상한 — 이보다 늦게 공시된 구간은 그 시점에 알 수 없었던 정보다.
    """

    known = [p for p in periods if p.filed <= cap]
    if not known:
        return None

    latest = max(known, key=lambda p: (p.end, p.filed))
    # 최신 구간이 이미 12개월이면 그 자체가 직전 12개월 창이다.
    if latest.months == 12:
        return [(latest, 1)], "ttm"

    annuals = [p for p in known if p.months == 12]
    # 직전 회계연도 연간: 최신 누적이 시작된 회계연도의 바로 앞 해 — 최신 구간 시작 하루 전에 끝난다.
    fy_start = _shift(latest.end, latest.months)
    prior_fy = next((p for p in annuals if _close(p.end, fy_start)), None)
    # 직전 회계연도의 같은 길이 누적: 1년 전 같은 분기까지.
    prior_ytd = next(
        (
            p
            for p in known
            if p.months == latest.months and _close(p.end, _shift(latest.end, 12))
        ),
        None,
    )
    if prior_fy and prior_ytd:
        return [(latest, 1), (prior_fy, 1), (prior_ytd, -1)], "ttm"

    # 조각이 안 맞으면 가장 최근 연간값으로 물러선다 — 최대 1년 가까이 묵을 수 있으므로
    # basis 라벨과 period_end 로 신선도를 그대로 노출한다.
    if annuals:
        return [(max(annuals, key=lambda p: (p.end, p.filed)), 1)], "annual"
    return None


def assemble_financials(
    symbol: str,
    periods: list[CumulativePeriod],
    balances: list[BalanceSnapshot],
    cap: date,
    metrics: tuple[str, ...] = ("net_income", "eps_diluted"),
) -> Financials | None:
    """누적 손익 구간 + 재무상태표 시점값 → 비율 계산용 스냅샷. 근거 부족이면 None.

    cap 은 공시 제출일 상한(관측 상한 t-1). 손익과 재무상태표는 같은 공시에서 와도
    각각 검증하며, filed 는 둘 중 늦은 쪽을 취해 누출 판정이 느슨해지지 않게 한다.

    창 선택은 **요청 지표가 모두 담긴 구간**만 후보로 본다. 한 지표만 있는 구간이 최신이라고
    그걸 고르면 나머지 지표가 통째로 빠진다(실측: 순이익만 롤링 12개월로 공시하고 주당이익은
    회계연도 단위로만 공시하는 발행인이 있다). 손익계산서가 온전한 구간에서만 합산한다.
    """

    complete = [p for p in periods if all(m in p.values for m in metrics)]
    selected = _select_window(complete, cap)
    if selected is None:
        return None
    window, basis = selected

    totals = {m: sum(sign * p.values[m] for p, sign in window) for m in metrics}

    known_balances = [b for b in balances if b.filed <= cap]
    balance = max(known_balances, key=lambda b: (b.end, b.filed)) if known_balances else None

    filed = max(p.filed for p, _ in window)
    period_end = max(p.end for p, _ in window)
    if balance:
        filed = max(filed, balance.filed)

    return Financials(
        symbol=symbol,
        filed=filed,
        period_end=period_end,
        basis=basis,
        net_income=totals.get("net_income"),
        eps_diluted=totals.get("eps_diluted"),
        equity=balance.values.get("equity") if balance else None,
        liabilities=balance.values.get("liabilities") if balance else None,
    )
