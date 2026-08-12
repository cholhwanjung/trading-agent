"""ETF 괴리율 관측 — 시장가와 순자산가치의 차이.

이것은 **구조 검증이 아니라 집행 비용**이다. 지수 ETF 를 산다는 것은 그 지수를 사는
것인데, 실제로 지불하는 것은 시장가다. 둘이 벌어져 있으면 지수 판단이 맞아도 그만큼
손해로 시작한다 — 예산 제약과 같은 성격의 관측이며 배분 판단이 아니라 집행 판단에 쓰인다.

누출 통제: 전일 최종 순자산가치와 t-1 종가로만 계산한다. 원천에는 당일 순자산가치와
당일 괴리율도 있지만 장중에 그 값을 받으면 관측이 당일 가격을 아는 셈이 된다.

**날짜 정합은 보장되지 않는다.** 원천이 순자산가치의 날짜를 함께 주지 않아 "직전
거래일"이라는 전제로 쓴다. 전제가 깨지면 t-1 종가와의 괴리가 비정상적으로 벌어지므로,
정상 범위를 벗어난 값은 계산하지 않고 버린다. 날짜 증명이 아니라 하한선일 뿐이다 —
조용한 오정렬은 잡지 못하고 큰 오정렬만 잡는다.
"""

from __future__ import annotations

from adapters.base import Observation

# 괴리율 정상 범위. 유동성 공급자가 붙는 지수 ETF 는 통상 ±0.5% 안에서 관리되고,
# 국내 규정도 일정 수준을 넘으면 관리 의무를 지운다. 이 범위를 벗어나면 값이 정말
# 이상하거나 순자산가치가 다른 날짜의 것이며, 어느 쪽이든 그대로 실을 값이 아니다.
MAX_ABS_PREMIUM = 0.03


def compute_premium(close: float | None, nav: float | None) -> float | None:
    """(t-1 종가 - 전일 순자산가치) / 전일 순자산가치. 범위 밖이면 None."""
    if not close or not nav or nav <= 0:
        return None
    premium = round(close / nav - 1.0, 5)
    return None if abs(premium) > MAX_ABS_PREMIUM else premium


def observed_etf_premium(obs: Observation) -> dict[str, dict]:
    """관측에서 종목별 괴리율 블록을 뽑는다. 순자산가치 원천이 없는 시장은 빈 dict."""

    out: dict[str, dict] = {}
    for symbol, nav in obs.etf_nav.items():
        bars = obs.bars.get(symbol) or []
        premium = compute_premium(bars[-1].close if bars else None, nav)
        if premium is not None:
            out[symbol] = {"nav": nav, "premium": premium}
    return out
