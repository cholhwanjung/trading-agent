"""상위 자본 배분 v1 — 고정 비율 가상 결합 지수.

세 페이퍼 계좌는 브로커·통화가 분리(USDT/USD/KRW)라 실자본 이동이 불가능하고
계좌 크기도 임의다. 따라서 v1 상위 배분은 **수익률 공간의 결합**: 시장별 equity 를
각자 시작점으로 정규화한 뒤 고정 자본 비율로 가중합한다. FX 환산 불필요.

규약:
- 아직 시작 안 된 시장의 ratio = 1.0 (현금 대기와 동일) — 늦게 합류해도 지수 연속.
- 데이터가 전혀 없는 시장은 제외하고 비중 재정규화.
- 실자본 재배분(고정비율 리밸런스 집행)은 실계좌 전환 후 리뷰.
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


def _eq_at(history: list[dict], day: str) -> float | None:
    """day 이하 최신 equity (step). 시작 전이면 None. 수백 건 선형 탐색 충분."""
    val = None
    for p in history:
        if p["day"] <= day:
            val = p["equity"]
        else:
            break
    return val


def combined_index_dynamic(
    state_dir: Path | str,
    arm: str,
    weights_by_day: dict[str, dict[str, float]],
) -> dict | None:
    """날짜별 메타 가중치로 **리밸런싱**한 결합 지수 ([ADR-025] verify).

    index_0 = 1.0; index_t = index_{t-1} · (1 + Σ_m w_m,t · r_m,t),
    r_m,t = 시장 m 일간 수익률(step equity). 미시작 시장은 수익 0 기여(현금 대기).

    weights_by_day: {proposal_day(iso): {market: weight}} — 각 거래일은 그 이하 최신
    제안 적용(step). **빈 dict 또는 첫 제안 이전 구간은 균등 anchor** → 고정 균등 baseline.

    주의: 리밸런싱 포트폴리오다. 공정 baseline 은 같은 함수에 빈 dict(상수 균등)를 넣은
    것 — buy&hold 인 combined_index 와 방법이 다르다(리밸런싱 효과와 배분 스킬 분리).
    """
    state_dir = Path(state_dir)
    markets = sorted({m for w in weights_by_day.values() for m in w})
    if not markets:  # 균등 baseline — 시장은 상태파일에서 유추
        markets = [m for m in MARKET_CAPITAL_WEIGHTS if _load_history(state_dir, m, arm)]
    histories = {m: h for m in markets if (h := _load_history(state_dir, m, arm))}
    if not histories:
        return None

    days = sorted({p["day"] for h in histories.values() for p in h})
    prop_days = sorted(weights_by_day)

    def weights_for(day: str) -> dict[str, float]:
        applicable = [pd for pd in prop_days if pd <= day]
        if applicable:
            return weights_by_day[applicable[-1]]
        return {m: 1.0 / len(histories) for m in histories}  # 제안 전 = 균등 anchor

    index = 1.0
    curve = [{"day": days[0], "index": 1.0}]
    for prev_day, day in zip(days, days[1:]):
        w = weights_for(day)
        port_ret = 0.0
        for m, h in histories.items():
            e0, e1 = _eq_at(h, prev_day), _eq_at(h, day)
            if e0 and e1 and e0 > 0:
                port_ret += w.get(m, 0.0) * (e1 / e0 - 1.0)
        index *= 1.0 + port_ret
        curve.append({"day": day, "index": round(index, 6)})

    peak, mdd = curve[0]["index"], 0.0
    for point in curve:
        peak = max(peak, point["index"])
        mdd = max(mdd, 1 - point["index"] / peak)
    return {
        "arm": arm,
        "days": len(curve),
        "markets": sorted(histories),
        "first_day": curve[0]["day"],
        "last_day": curve[-1]["day"],
        "index": curve[-1]["index"],
        "ret_pct": (curve[-1]["index"] - 1) * 100,
        "mdd_pct": mdd * 100,
        "curve": curve,
    }
