"""메타 배분 입력 패널 — 시장별 집계 신호를 id 가 붙은 값으로 만든다.

신호 id 는 credit assignment 의 열쇠다. 제안이 인용한 id 의 유형별 적중률을 결과 기입
(`meta.outcomes`)이 누적하고, 그것이 다음 제안의 신뢰도 표가 된다. id 형식은
`<type>:<market>[:<param>]` 이며 `signal_type` 이 앞부분을 돌려준다.

1주 수익률은 일부러 뺐다 — 지난주 오른 시장을 늘리는 것은 추세 추격이라서다.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from datetime import date, timedelta

from adapters.base import Bar
from regime.live import LOOKBACK_DAYS
from regime.pulse import classify_regime

MARKETS = ("CRYPTO", "KR", "US")
PERIODS_PER_YEAR = {"CRYPTO": 365, "KR": 252, "US": 252}
TRAILING = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
VOL_WINDOWS = (20, 60)


@dataclass(frozen=True)
class MarketPanel:
    market: str
    regime: str | None
    distribution_days: int | None
    drawdown: float | None  # 롤링 피크 대비 (0~1)
    trailing: dict[str, float | None]  # 기간 → 수익률(소수)
    vol: dict[str, float | None]  # "20"/"60" → 연율 변동성 %
    valuation: dict | None  # 지수 밸류에이션(pe·pb·dividend_yield) — 원천 없는 시장은 None


def signal_type(signal_id: str) -> str:
    """`mom:KR:3m` → `mom`. 구분자가 없으면 전체."""
    return signal_id.split(":", 1)[0]


def _trailing(closes: list[float], n: int) -> float | None:
    if len(closes) <= n or closes[-1 - n] <= 0:
        return None
    return closes[-1] / closes[-1 - n] - 1.0


def _vol(closes: list[float], n: int, ppy: int) -> float | None:
    if len(closes) <= n:
        return None
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - n, len(closes))]
    return statistics.pstdev(rets) * math.sqrt(ppy) * 100


def market_panel(
    market: str, bars: list[Bar], asof_day: date, valuation: dict | None = None
) -> MarketPanel:
    """일봉(오름차순, ≤ asof−1) → 집계 신호. 국면은 라이브와 같은 창으로 다시 계산한다."""
    lo = asof_day - timedelta(days=LOOKBACK_DAYS)
    regime = classify_regime([b for b in bars if lo <= b.day < asof_day])
    closes = [b.close for b in bars if b.day < asof_day]
    ppy = PERIODS_PER_YEAR[market]
    return MarketPanel(
        market=market,
        regime=regime.state if regime else None,
        distribution_days=regime.distribution_days if regime else None,
        drawdown=round(regime.drawdown, 4) if regime else None,
        trailing={k: _trailing(closes, n) for k, n in TRAILING.items()},
        vol={str(w): _vol(closes, w, ppy) for w in VOL_WINDOWS},
        valuation=valuation,
    )


def build_panel(
    panels: dict[str, MarketPanel],
    asof_day: date,
    anchor: dict[str, float],
    max_tilt: float,
    macro: dict | None = None,
    prev_proposal: dict | None = None,
    actual_split: dict[str, float] | None = None,
) -> dict:
    """제안자 입력 전체 + 유효 signal id 목록. 프롬프트와 감사가 같은 객체를 본다."""
    markets: dict[str, dict] = {}
    ids: list[str] = []
    for m, p in panels.items():
        d = asdict(p)
        d.pop("market")
        markets[m] = d
        ids += [f"regime:{m}", f"dd:{m}"]
        ids += [f"mom:{m}:{k}" for k, v in p.trailing.items() if v is not None]
        ids += [f"vol:{m}:{w}" for w, v in p.vol.items() if v is not None]
        if p.valuation:
            ids.append(f"val:{m}")
    if macro:
        ids += [f"macro:{k}" for k in ("state", "vix", "yield_spread") if macro.get(k) is not None]
    return {
        "asof_day": asof_day.isoformat(),
        "anchor": dict(anchor),
        "max_tilt": max_tilt,
        "markets": markets,
        "macro": macro,
        "prev_proposal": prev_proposal,
        "actual_split": actual_split,
        "signal_ids": ids,
    }
