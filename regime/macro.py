"""매크로 국면 분류 — FRED 일간 지표 → calm/stress/crisis (결정론, 순수 함수).

주 게이지는 VIX, 침체 수정자는 장단기 금리차. 임계는 시장 관용(VIX 20/30)에서 취하고
우리 표본으로 튜닝하지 않는다(pulse.py 와 동일 원칙). usdkrw·dollar·fed_funds 는 지금은
컨텍스트 로깅용(향후 FX·리스크 게이트 승격 대상) — 분류에는 미사용.

지수 국면(pulse.py, 시장별)과 별개의 전역 매크로 신호 — v1 은 shadow(계산·로깅만).
"""

from __future__ import annotations

from dataclasses import dataclass

CALM = "CALM"
STRESS = "STRESS"
CRISIS = "CRISIS"

# 시장 관용 임계 (튜닝 금지) — VIX 20=변동성 상승, 30=고공포
VIX_STRESS = 20.0
VIX_CRISIS = 30.0
VIX_INVERSION_CRISIS = 28.0  # 금리 역전(장단기<0) 동반 시 crisis 문턱 하향


@dataclass(frozen=True)
class MacroRegime:
    state: str  # CALM | STRESS | CRISIS
    vix: float | None
    yield_spread: float | None
    values: dict  # 전체 원시값 (컨텍스트 로깅)


def classify_macro(values: dict) -> MacroRegime | None:
    """FRED 지표 dict → MacroRegime. 주 게이지(VIX) 결측이면 None(판정 보류)."""
    vix = values.get("vix")
    ys = values.get("yield_spread")
    if vix is None:
        return None
    inverted = ys is not None and ys < 0
    if vix >= VIX_CRISIS or (vix >= VIX_INVERSION_CRISIS and inverted):
        state = CRISIS
    elif vix >= VIX_STRESS:
        state = STRESS
    else:
        state = CALM
    return MacroRegime(state=state, vix=vix, yield_spread=ys, values=values)
