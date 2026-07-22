"""active 팩터 → Trader 관측 신호 주입 (정예만: top-3 상한).

팩터 스코어는 관측 보조 신호일 뿐 알파 원천이 아니다. 주입 형식은
당일 횡단면 z-score × IC 부호 — 값이 클수록 "팩터가 기대하는 익일 상대 우위".
Trader 는 signal id (`alpha:<name>`) 로 인용한다 (credit assignment).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from alpha_lab.data import fetch_crypto_panel
from alpha_lab.dsl import DSLError, evaluate
from alpha_lab.library import FactorLibrary

MAX_SIGNALS = 3  # 정예만 주입
SIGNAL_LOOKBACK_DAYS = 200  # 팩터 워밍업(MAX_WINDOW=120) + 여유


def _effective_ic(factor) -> float:
    """랭킹 기준 — 라이브 IC 가 갱신됐으면 그것을(감쇠 반영), 없으면 admission OOS IC.

    알파는 감쇠하므로 최신 라이브 추정치가 우선. 주간 review_decay 가
    live_ic 를 갱신하고 우위 소멸분은 이미 retire 하므로, 여기선 살아남은 팩터의
    현재 강도로 top-3 를 고른다.
    """
    return abs(factor.live_ic) if factor.live_ic is not None else abs(factor.oos_ic)


async def compute_alpha_signals(
    library_path: Path | str,
    trading_universe: list[str],
    asof_day: date | None = None,
    panel_fn=fetch_crypto_panel,
) -> dict:
    """{"asof": day, "signals": {"alpha:<name>": {"oos_ic":…, "scores": {sym: z}}}}.

    panel_fn 은 시장별 연구 패널 소스(CRYPTO=fetch_crypto_panel · US=make_us_panel_fn).
    라이브러리 없음/팩터 없음/계산 실패 → 빈 signals (비치명, 관측 보조일 뿐).
    """
    library_path = Path(library_path)
    if not library_path.exists():
        return {"asof": None, "signals": {}}
    library = FactorLibrary(library_path)
    top = sorted(library.active(), key=_effective_ic, reverse=True)[:MAX_SIGNALS]
    if not top:
        return {"asof": None, "signals": {}}

    panel, symbols, dates = await panel_fn(
        lookback_days=SIGNAL_LOOKBACK_DAYS, asof_day=asof_day
    )
    idx = {s: j for j, s in enumerate(symbols)}
    signals: dict[str, dict] = {}
    for factor in top:
        try:
            scores = evaluate(factor.expression, panel)
        except DSLError:
            continue
        row = scores[-1]
        mean, std = np.nanmean(row), np.nanstd(row)
        if not np.isfinite(std) or std == 0:
            continue
        z = (row - mean) / std
        signals[f"alpha:{factor.name}"] = {
            "oos_ic": factor.oos_ic,
            "live_ic": factor.live_ic,  # None = 아직 라이브 표본 부족 (감쇠 미평가)
            "hypothesis": factor.hypothesis,
            "scores": {
                s: round(float(z[idx[s]]) * factor.sign, 3)
                for s in trading_universe
                if s in idx and np.isfinite(z[idx[s]])
            },
        }
    return {"asof": dates[-1].isoformat(), "signals": signals}
