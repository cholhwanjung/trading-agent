"""ablation 델타 리포트 — 가상 arm equity 곡선 비교 측정 도구.

사용법:
    uv run python scripts/report_ablation.py

arm 의미:
    llm       = 메모리 블렌딩 최종 배분 (실계좌와 동일)
    llm_base  = 무메모리 base 배분 (No-Memory ablation arm)
    bh        = Buy&Hold 균등 배분 / random = 무작위(일별 seed)

핵심 지표: memory_delta = llm − llm_base (메모리 영향력의 순기여, ablation)
           alpha_vs_bh = llm − bh (성공기준의 분자)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATE = ROOT / "data" / "state" / "virtual"

from eval.meta import load_arm_history, max_drawdown  # noqa: E402


def load_arm(market: str, arm: str) -> dict | None:
    history = load_arm_history(STATE, market, arm)
    if not history:
        return None
    equities = [h["equity"] for h in history]
    return {
        "days": len(history),
        "first_day": history[0]["day"],
        "last_day": history[-1]["day"],
        "equity": equities[-1],
        "ret_pct": (equities[-1] / 100_000.0 - 1) * 100,
        "mdd_pct": max_drawdown(equities) * 100,
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
            print(f"memory_delta_pct={llm['ret_pct'] - base['ret_pct']:+.4f}  # llm − llm_base")
        if llm and bh:
            print(f"alpha_vs_bh_pct={llm['ret_pct'] - bh['ret_pct']:+.4f}  # 성공기준 분자")

        # rolling-k delta — 승격 판정 입력 (일관성: 누적치 1개가 아니라 창 승률)
        from eval.rolling import ROLLING_K, rolling_report

        rolled = rolling_report(STATE, market)
        for name, r in (("memory", rolled["memory"]), ("alpha", rolled["alpha"])):
            if r is None:
                print(f"rolling_{name} status=insufficient need_days>={ROLLING_K + 1}")
                continue
            p = f"{r['p_value']:.4f}" if r["p_value"] is not None else f"n/a(청크 {r['n_chunks']}<5)"
            print(
                f"rolling_{name} k={r['k']} win_rate={r['win_rate']:.2f}"
                f" mean={r['mean_delta_pct']:+.3f}% latest={r['latest_delta_pct']:+.3f}%"
                f" sign_p={p}  # 비중첩 {r['n_chunks']}청크 중 양성 {r['chunks_positive']}"
            )

    # 상위 결합 지수 — 고정비율 가상 배분
    from eval.meta import combined_index

    meta = {a: combined_index(STATE, a) for a in ("llm", "llm_base", "bh")}
    if any(meta.values()):
        markets = next(s for s in meta.values() if s)["markets"]
        print(f"\n=== META (고정비율 결합: {markets}) ===")
        for name, s in meta.items():
            if s is None:
                print(f"arm={name} status=no_data")
                continue
            print(
                f"arm={name} days={s['days']} index={s['index']:.4f}"
                f" ret={s['ret_pct']:+.3f}% mdd={s['mdd_pct']:.3f}%"
            )
        if meta["llm"] and meta["bh"]:
            print(f"meta_alpha_vs_bh_pct={meta['llm']['ret_pct'] - meta['bh']['ret_pct']:+.4f}")

    # 동적 메타 배분 vs 고정 균등 — shadow 검증. 같은 리밸런싱법으로 배분 스킬만 분리.
    from eval.meta import load_meta_shadow
    from eval.rolling import ROLLING_K, meta_shadow_delta

    weights_by_day = load_meta_shadow(ROOT / "data" / "state" / "meta_shadow.json")
    if weights_by_day:
        print(f"\n=== META SHADOW (동적 vs 고정균등 리밸런싱, 제안 {len(weights_by_day)}일) ===")
        for arm in ("llm", "bh"):
            r = meta_shadow_delta(STATE, arm, weights_by_day)
            if r is None:
                print(f"arm={arm} status=insufficient need_days>={ROLLING_K + 1}")
                continue
            p = f"{r['p_value']:.4f}" if r["p_value"] is not None else f"n/a(청크 {r['n_chunks']}<5)"
            print(
                f"arm={arm} k={r['k']} win_rate={r['win_rate']:.2f}"
                f" mean={r['mean_delta_pct']:+.3f}% latest={r['latest_delta_pct']:+.3f}%"
                f" sign_p={p}  # dynamic−equal 비중첩 {r['n_chunks']}청크 양성 {r['chunks_positive']}"
            )
    print(
        "\nnote=단기 표본은 통계력 없음 — 보조 지표(메모리 승격/퇴출률·인용 기여)와 함께 볼 것"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
