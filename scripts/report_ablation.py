"""ablation 델타 리포트 — 가상 arm equity 곡선 비교 (Phase 2 완료 기준 측정 도구).

사용법:
    uv run python scripts/report_ablation.py

arm 의미:
    llm       = 메모리 블렌딩 최종 배분 (실계좌와 동일)
    llm_base  = 무메모리 base 배분 (No-Memory ablation arm)
    bh        = Buy&Hold 균등 배분 / random = 무작위(일별 seed)

핵심 지표: memory_delta = llm − llm_base (메모리 영향력의 순기여, R9 ablation)
           alpha_vs_bh = llm − bh (PRD 성공기준의 분자)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "state" / "virtual"


def load_arm(market: str, arm: str) -> dict | None:
    path = STATE / f"{market}_{arm}.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    history = state.get("history") or []
    if not history:
        return None
    equities = [h["equity"] for h in history]
    peak = equities[0]
    mdd = 0.0
    for eq in equities:
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return {
        "days": len(history),
        "first_day": history[0]["day"],
        "last_day": history[-1]["day"],
        "equity": equities[-1],
        "ret_pct": (equities[-1] / 100_000.0 - 1) * 100,
        "mdd_pct": mdd * 100,
        "total_cost": round(sum(h.get("cost", 0.0) for h in history), 2),
    }


def main() -> int:
    markets = sorted({p.name.split("_")[0] for p in STATE.glob("*.json")})
    if not markets:
        print("status=empty detail=가상 상태 없음 — run_paper_step 이 먼저 돌아야 한다")
        return 1

    for market in markets:
        arms = {a: load_arm(market, a) for a in ("llm", "llm_base", "bh", "random")}
        print(f"\n=== {market} ===")
        for name, s in arms.items():
            if s is None:
                print(f"arm={name} status=no_data")
                continue
            print(
                f"arm={name} days={s['days']} equity={s['equity']:,.2f}"
                f" ret={s['ret_pct']:+.3f}% mdd={s['mdd_pct']:.3f}% cost={s['total_cost']}"
            )
        llm, base, bh = arms.get("llm"), arms.get("llm_base"), arms.get("bh")
        if llm and base:
            print(f"memory_delta_pct={llm['ret_pct'] - base['ret_pct']:+.4f}  # llm − llm_base (R9)")
        if llm and bh:
            print(f"alpha_vs_bh_pct={llm['ret_pct'] - bh['ret_pct']:+.4f}  # PRD 성공기준 분자")
    print(
        "\nnote=단기 표본은 통계력 없음 — 보조 지표(메모리 승격/퇴출률·인용 기여)와 함께 볼 것 (PRD)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
