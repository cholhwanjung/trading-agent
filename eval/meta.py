"""상위 자본 배분 v1 — 고정 비율 가상 결합 지수 (Phase 4 · [ADR-018]).

세 페이퍼 계좌는 브로커·통화가 분리(USDT/USD/KRW)라 실자본 이동이 불가능하고
계좌 크기도 임의다. 따라서 v1 상위 배분은 **수익률 공간의 결합**: 시장별 equity 를
각자 시작점으로 정규화한 뒤 고정 자본 비율로 가중합한다. FX 환산 불필요 (R16).

규약:
- 아직 시작 안 된 시장의 ratio = 1.0 (현금 대기와 동일) — 늦게 합류해도 지수 연속.
- 데이터가 전혀 없는 시장은 제외하고 비중 재정규화.
- 실자본 재배분(고정비율 리밸런스 집행)은 Phase 6 실계좌 전환 후 리뷰.
"""

from __future__ import annotations

import json
from pathlib import Path

MARKET_CAPITAL_WEIGHTS = {"CRYPTO": 1 / 3, "US": 1 / 3, "KR": 1 / 3}  # 초기 고정 비율


def _load_history(state_dir: Path, market: str, arm: str) -> list[dict]:
    path = state_dir / f"{market}_{arm}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("history") or []


def combined_index(
    state_dir: Path | str,
    arm: str,
    weights: dict[str, float] | None = None,
) -> dict | None:
    """시장별 가상 arm equity → 고정비율 결합 지수 (시작=1.0). 데이터 없으면 None."""
    state_dir = Path(state_dir)
    weights = weights or MARKET_CAPITAL_WEIGHTS
    histories = {
        m: h for m in weights if (h := _load_history(state_dir, m, arm))
    }
    if not histories:
        return None
    total_w = sum(weights[m] for m in histories)

    days = sorted({p["day"] for h in histories.values() for p in h})
    curve: list[dict] = []
    for day in days:
        index = 0.0
        for market, history in histories.items():
            first = history[0]["equity"]
            ratio = 1.0  # 시장 시작 전 = 현금 대기
            for p in history:  # day 오름차순, 수백 건 수준 — 선형 탐색 충분
                if p["day"] <= day:
                    ratio = p["equity"] / first
                else:
                    break
            index += weights[market] / total_w * ratio
        curve.append({"day": day, "index": round(index, 6)})

    peak, mdd = curve[0]["index"], 0.0
    for point in curve:
        peak = max(peak, point["index"])
        mdd = max(mdd, 1 - point["index"] / peak)
    return {
        "arm": arm,
        "days": len(curve),
        "markets": {m: round(weights[m] / total_w, 4) for m in histories},
        "first_day": curve[0]["day"],
        "last_day": curve[-1]["day"],
        "index": curve[-1]["index"],
        "ret_pct": (curve[-1]["index"] - 1) * 100,
        "mdd_pct": mdd * 100,
        "curve": curve,
    }
