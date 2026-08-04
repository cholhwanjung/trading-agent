"""시장 실현변동성 게이지 — 지수 프록시 일간 봉 → CALM/STRESS/CRISIS (결정론·순수).

전역 매크로 게이지(VIX)는 미국 내재변동성이라 한 시장에 국한된 위기를 못 본다 —
2026-07 KOSPI 월간 −28% 급락일에도 VIX 는 18 로 "평온"이었다. 시장별 지수 봉의
실현변동성(최근 20거래일, 연율화 %)을 같은 관용 임계(20/30)로 분류해 그 사각을 메운다.
실현 20일 변동성은 VIX(내재 30일)의 표준 역사적 대응치라 임계를 재사용한다(무튜닝).
VKOSPI 같은 지역 내재변동성 지수는 무료 API 부재 — 실현변동성 대체(추가 의존성 0).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from regime.macro import CALM, CRISIS, STRESS, VIX_CRISIS, VIX_STRESS

VOL_WINDOW = 20  # 거래일 — VIX(30캘린더일)의 관용 실현 대응 창
_ANNUALIZE = 252**0.5


@dataclass(frozen=True)
class VolResult:
    state: str  # CALM | STRESS | CRISIS
    realized_vol: float  # 연율화 %, VIX 스케일
    n_bars: int


def realized_vol_pct(closes: list[float], window: int = VOL_WINDOW) -> float | None:
    """최근 window 일 단순수익률 표준편차 × √252 × 100 (연율화 %). 표본 부족 → None."""
    if len(closes) < window + 1:
        return None
    tail = closes[-(window + 1) :]
    rets = [tail[i] / tail[i - 1] - 1 for i in range(1, len(tail)) if tail[i - 1] > 0]
    if len(rets) < window:
        return None
    return statistics.pstdev(rets) * _ANNUALIZE * 100


def classify_vol(closes: list[float]) -> VolResult | None:
    """종가 시퀀스(오름차순, 상한 t−1) → VolResult. 표본 부족 시 None(판정 보류)."""
    rv = realized_vol_pct(closes)
    if rv is None:
        return None
    if rv >= VIX_CRISIS:
        state = CRISIS
    elif rv >= VIX_STRESS:
        state = STRESS
    else:
        state = CALM
    return VolResult(state=state, realized_vol=round(rv, 2), n_bars=len(closes))
