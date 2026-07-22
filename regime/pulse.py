"""시장 국면(regime) 상태기계 — O'Neil "market direction (M)" 이식 ([ADR-023]).

무료 일간 지수 봉만으로 시장 건강을 3개 상태로 분류하는 결정론 FSM (순수 함수,
I/O·클럭·네트워크 없음 → 단위 테스트 가능). prism-insight `cores/market_pulse.py`
계보이나 trading-agent `Bar` 에 맞춰 재구현·간소화.

상태:
    UPTREND         — 정상; 롤링 윈도우 내 분산일(DD) ≤ 3
    UNDER_PRESSURE  — DD 4~5 (기관 매도 누적 경고)
    CORRECTION      — DD ≥ 6, 또는 롤링 피크 종가 대비 −10% 낙폭. 탈출은
                      Follow-Through Day(FTD) 또는 사전-조정 피크 상회 회복.

상수 출처: IBD / William O'Neil 공개 방법론(60년 시장사) — 우리 표본으로 튜닝하지 않음.
    | 분산일(DD)        | 종가 ≤ −0.2% & 거래량 > 전일 | IBD 표준
    | DD 만료           | 25 세션 경과, 또는 +5% 회복   | IBD 표준
    | CORRECTION 진입   | DD ≥ 6, 또는 피크 대비 −10%   | IBD "market in correction"
    | UNDER_PRESSURE    | DD ∈ {4, 5}                   | IBD
    | Follow-Through Day| 랠리 4일차+ & +1.25% & 거래량↑| O'Neil HTMMIS
    | 회복 탈출         | 종가 > 사전-조정 피크         | O'Neil (신고가 = 정의상 상승)
"""

from __future__ import annotations

from dataclasses import dataclass

from adapters.base import Bar

UPTREND = "UPTREND"
UNDER_PRESSURE = "UNDER_PRESSURE"
CORRECTION = "CORRECTION"

# IBD/O'Neil 상수 (튜닝 금지 — 출처는 모듈 docstring)
DD_DROP = 0.002  # 분산일 종가 하락 임계 (−0.2%)
DD_WINDOW = 25  # 분산일 카운트 롤링 세션 수
DD_RECOVERY = 0.05  # 분산일 종가 대비 +5% 회복 시 만료
CORRECTION_DD = 6  # DD ≥ 6 → CORRECTION
PRESSURE_DD = 4  # DD ∈ {4,5} → UNDER_PRESSURE
DRAWDOWN_TRIGGER = 0.10  # 피크 대비 −10% → CORRECTION
FTD_MIN_GAIN = 0.0125  # Follow-Through Day 최소 상승 (+1.25%)
FTD_MIN_DAY = 4  # 랠리 4일차 이상에서만 FTD 인정
MIN_BARS = 30  # 판정에 필요한 최소 봉 수 (미만이면 None 상태)


@dataclass(frozen=True)
class RegimeResult:
    state: str  # UPTREND | UNDER_PRESSURE | CORRECTION
    distribution_days: int
    drawdown: float  # 현재 롤링 피크 대비 낙폭 (0~1)
    n_bars: int


def _is_distribution(bar: Bar, prev: Bar) -> bool:
    """분산일: 종가 −0.2% 이상 하락 + 거래량 전일 초과 (기관 매도)."""
    if prev.close <= 0 or bar.volume <= 0 or prev.volume <= 0:
        return False
    return bar.close <= prev.close * (1 - DD_DROP) and bar.volume > prev.volume


def classify_regime(bars: list[Bar]) -> RegimeResult | None:
    """일간 봉 시퀀스(오름차순)를 replay 해 현재 국면을 판정. 봉 부족 시 None.

    순수 함수 — 전체 시퀀스를 매 호출 재생(O(n)). 호출부는 상한 t−1 봉만 넘긴다
    ([ADR-013] 누출 통제와 동일 — 국면은 전일 종가 기준).
    """
    bars = [b for b in bars if b.volume is not None]
    if len(bars) < MIN_BARS:
        return None

    active_dd: list[tuple[int, float]] = []  # (index, 종가) — 활성 분산일
    peak = bars[0].close  # 롤링 참조 피크 (CORRECTION 중엔 사전-조정 피크로 동결)
    state = UPTREND
    rally_day = 0  # CORRECTION 중 랠리 시도 경과일 (0 = 시도 없음)
    rally_low = 0.0

    for t in range(1, len(bars)):
        bar, prev = bars[t], bars[t - 1]

        # 분산일 만료(25세션 경과 또는 +5% 회복) 후 신규 분산일 추가
        active_dd = [
            (j, dc) for j, dc in active_dd
            if (t - j) < DD_WINDOW and bar.close < dc * (1 + DD_RECOVERY)
        ]
        if _is_distribution(bar, prev):
            active_dd.append((t, bar.close))
        dd_count = len(active_dd)

        if state != CORRECTION:
            peak = max(peak, bar.close)
            drawdown = (peak - bar.close) / peak if peak > 0 else 0.0
            if dd_count >= CORRECTION_DD or drawdown >= DRAWDOWN_TRIGGER:
                state = CORRECTION
                rally_day, rally_low = 0, bar.low  # 사전-조정 피크(peak)는 동결
            elif dd_count >= PRESSURE_DD:
                state = UNDER_PRESSURE
            else:
                state = UPTREND
        else:
            # CORRECTION — 랠리 추적 + FTD/회복 탈출
            up_close = bar.close > prev.close
            if rally_day == 0:
                if up_close:  # 저점 후 첫 상승 종가 = 랠리 1일차
                    rally_day, rally_low = 1, min(bar.low, prev.low)
            elif bar.low < rally_low:  # 랠리 저점 이탈 → 시도 무효
                rally_day, rally_low = 0, bar.low
            else:
                rally_day += 1

            gain = bar.close / prev.close - 1 if prev.close > 0 else 0.0
            vol_up = prev.volume > 0 and bar.volume > prev.volume
            ftd = rally_day >= FTD_MIN_DAY and gain >= FTD_MIN_GAIN and vol_up
            if ftd or bar.close > peak:  # FTD 또는 사전-조정 피크 회복 → 탈출
                state = UPTREND
                peak = bar.close  # 피크 리셋(edge-trigger) — 다음 피크는 새로 형성

    final_peak = peak
    drawdown = (final_peak - bars[-1].close) / final_peak if final_peak > 0 else 0.0
    return RegimeResult(
        state=state,
        distribution_days=len(active_dd),
        drawdown=round(max(0.0, drawdown), 4),
        n_bars=len(bars),
    )
