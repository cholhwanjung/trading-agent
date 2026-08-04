"""배분 집중도·실효 분산 — 명목 분산 뒤에 숨은 동조 리스크 계량 (결정론·순수).

종목당 상한(max_weight_per_asset)은 **명목** 분산만 보장한다. 상관 0.9 로 동조하는 두
종목에 각각 상한까지 배분하면 명목 2종목이지만 실질은 1종목이고, 종목당 상한 규칙은
이를 위반으로 보지 않는다. 2026-07 KOSPI 급락일(삼성전자 −13.4% · SK하이닉스 −14.7%
동반 하락, 두 종목이 지수 절반)이 그 사각의 실증이다.

지표(현금 제외·위험자산만 재정규화):
    hhi              허핀달 지수 Σwᵢ² — 배분의 명목 집중도
    effective_n      1/hhi — 명목 유효 종목 수
    effective_n_corr 1/(wᵀΡw) — **상관 조정** 유효 종목 수. 전 종목 상관 1 이면 1 로
                     붕괴한다(=실질 단일 베팅). 핵심 지표. 보유 종목 수로 상한.
    avg_corr         배분 가중 평균 쌍상관 — effective_n_corr 해석용

I/O·클럭 없음 — 호출부가 상한 t−1 봉만 넘긴다(누출 통제는 관측 계층 책임).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

CORR_WINDOW = 60  # 상관 추정 거래일 — 국면 변화 반영과 추정 안정성의 절충
MIN_CORR_BARS = 20  # 이보다 짧으면 상관 추정 포기(None)


@dataclass(frozen=True)
class ConcentrationResult:
    hhi: float
    effective_n: float
    effective_n_corr: float | None  # 상관 추정 불가 시 None
    avg_corr: float | None
    n_assets: int  # 배분 > 0 인 위험자산 수


def risk_weights(weights: dict[str, float], cash_key: str = "CASH") -> dict[str, float]:
    """현금 제외 위험자산 배분을 합 1 로 재정규화. 위험자산 없으면 빈 dict."""
    risky = {s: w for s, w in weights.items() if s != cash_key and w > 0}
    total = sum(risky.values())
    return {s: w / total for s, w in risky.items()} if total > 0 else {}


def to_returns(closes: list[float]) -> list[float]:
    """종가 → 단순수익률. 직전 종가가 0 이하인 구간은 건너뛴다."""
    return [
        closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] > 0
    ]


def _pearson(a: list[float], b: list[float]) -> float | None:
    """최근 CORR_WINDOW 구간(끝 정렬) 피어슨 상관. 표본 부족·무변동이면 None."""
    n = min(len(a), len(b), CORR_WINDOW)
    if n < MIN_CORR_BARS:
        return None
    x, y = a[-n:], b[-n:]
    try:
        return statistics.correlation(x, y)
    except statistics.StatisticsError:  # 한쪽이 상수 — 상관 정의 불가
        return None


def concentration(
    weights: dict[str, float], returns_by_symbol: dict[str, list[float]]
) -> ConcentrationResult | None:
    """배분 + 종목별 수익률 → 집중도·실효 분산. 위험자산 0 이면 None(전액 현금).

    상관을 못 구한 쌍은 **1.0(완전 동조)으로 보수 처리** — 모르는 분산 효과를
    있다고 가정하지 않는다(리스크 지표의 안전한 방향).
    """
    w = risk_weights(weights)
    if not w:
        return None

    hhi = sum(x * x for x in w.values())
    symbols = sorted(w)
    rets = {s: to_returns(returns_by_symbol.get(s) or []) for s in symbols}

    quad = 0.0  # wᵀΡw
    pair_sum = 0.0  # 배분 가중 쌍상관 합 (i≠j)
    pair_weight = 0.0
    estimated = False
    for i, si in enumerate(symbols):
        for j, sj in enumerate(symbols):
            if i == j:
                quad += w[si] * w[sj]  # ρᵢᵢ = 1
                continue
            rho = _pearson(rets[si], rets[sj])
            if rho is None:
                rho = 1.0  # 보수적 기본값 — 분산 효과를 임의로 인정하지 않는다
            else:
                estimated = True
            quad += w[si] * w[sj] * rho
            pair_sum += w[si] * w[sj] * rho
            pair_weight += w[si] * w[sj]

    # 강한 음의 상관이면 wᵀΡw → 0 (이론상 무한 분산). 실효 분산을 보유 종목 수로
    # 상한 — 이 지표의 관심 영역은 **낮은 쪽**(1 에 수렴 = 실질 단일 베팅)이라
    # 상한을 씌워도 경보 정보가 손실되지 않고 해석이 단순해진다.
    cap = float(len(w))
    eff_corr = cap if quad <= 0 else min(1 / quad, cap)

    return ConcentrationResult(
        hhi=round(hhi, 4),
        effective_n=round(1 / hhi, 2),
        effective_n_corr=round(eff_corr, 2),
        avg_corr=round(pair_sum / pair_weight, 3) if estimated and pair_weight > 0 else None,
        n_assets=len(w),
    )
