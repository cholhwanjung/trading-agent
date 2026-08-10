"""Trader 관측 재무 — 정예 3종.

가격 파생 feature 와는 **별개 채널**이다. 분기마다 한 번 갱신되는 저속 값이라 일간
모멘텀 벡터에 섞으면 성격이 다른 값이 한 묶음에 들어가고, 그 벡터는 메모리 패턴 키의
재료라서 분기마다 값이 바뀌면 같은 패턴의 반복 관측이 쌓이지 않는다.

왜 3종인가 — 플레이북이 결정 근거로 요구하는 축(사업의 장기 경제성 · 재무 구조 ·
안전마진)에 하나씩 대응시킨 최소 집합이다. 확보한 재무 원값은 이보다 많지만 관측에
넣는 것은 여기까지다. 나머지는 국면 판정·메모리 통계 검증·챗 문답에서만 쓴다.
확보량과 주입량을 같게 두면 곧바로 관측 과부하가 된다.

적자·자본잠식 구간에서는 비율이 음수로 나오는데, 그 값은 "싸다"가 아니라 "해당 없음"을
뜻한다. 숫자로 넘기면 LLM 이 크기 비교를 해버리므로 계산하지 않고 비운다.
"""

from __future__ import annotations

from adapters.base import Observation
from adapters.financials import Financials

# 정예 재무 목록 — 이 튜플이 유일한 진실. 가격 feature 예산과는 별도 채널이며 3개를 넘기지 않는다.
FUNDAMENTAL_NAMES: tuple[str, ...] = (
    "pe_ttm",          # 주가 / 주당순이익(TTM) — 안전마진
    "roe_ttm",         # 순이익(TTM) / 자본총계 — 장기 경제성
    "debt_to_equity",  # 부채총계 / 자본총계 — 재무 구조
)
assert len(FUNDAMENTAL_NAMES) <= 3, "관측 재무는 3개 이하"


def compute_fundamentals(fin: Financials, price: float | None) -> dict | None:
    """재무 스냅샷 + t-1 종가 → 정예 비율. 셋 다 계산 불가면 None.

    basis·period_end 를 함께 실어 값이 얼마나 묵었는지 드러낸다 — 분기 공시라
    신선도가 판단의 일부다(직전 연간값으로 물러선 경우 basis 가 그 사실을 말한다).
    """

    ratios: dict[str, float | None] = dict.fromkeys(FUNDAMENTAL_NAMES)
    if price and fin.eps_diluted and fin.eps_diluted > 0:
        ratios["pe_ttm"] = round(price / fin.eps_diluted, 2)
    if fin.equity and fin.equity > 0:
        if fin.net_income is not None:
            ratios["roe_ttm"] = round(fin.net_income / fin.equity, 4)
        if fin.liabilities is not None:
            ratios["debt_to_equity"] = round(fin.liabilities / fin.equity, 2)
    if all(v is None for v in ratios.values()):
        return None
    return {**ratios, "basis": fin.basis, "period_end": str(fin.period_end)}


def observed_fundamentals(obs: Observation) -> dict[str, dict]:
    """관측에서 종목별 재무 블록을 뽑는다. 재무 원천이 없는 시장은 빈 dict.

    주가는 관측 창의 마지막 봉 종가(t-1)를 쓴다 — 재무는 t-1 이전 제출분이므로
    둘 다 관측 상한 안에 있다.
    """

    out: dict[str, dict] = {}
    for symbol, fin in obs.financials.items():
        bars = obs.bars.get(symbol) or []
        block = compute_fundamentals(fin, bars[-1].close if bars else None)
        if block:
            out[symbol] = block
    return out
