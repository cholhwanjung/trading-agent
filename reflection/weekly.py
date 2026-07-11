"""주간 reflection (FinMem extended · 설계 §3.5) — 주 단위 성과·근거 재평가.

- 기여 계측은 이미 일간에 계산된 outcome(행동−무행동)을 인용 단위로 집계한다.
- **credit assignment**: 인용된 memory 의 importance 를 결과 방향으로 소폭 갱신
  (bounded 0.1~0.9) — retrieval 순위에 반영될 뿐, admission/veto 게이트와 무관.
- 요약은 LLM(fast) 이 쓰되 실패해도 통계 리포트는 저장된다.
- semantic 승격은 여기서 하지 않는다 — admission 게이트 단일 경로 (ADR-004).
"""

from __future__ import annotations

from datetime import date, timedelta

from memory.store import MemoryStore

WINDOW_DAYS = 7
MIN_ENTRIES = 3
IMPORTANCE_STEP = 0.05
IMPORTANCE_MIN, IMPORTANCE_MAX = 0.1, 0.9


def compute_weekly_report(store: MemoryStore, market: str, asof_day: date) -> dict | None:
    """최근 7일 episodic(결과 있는 것만) 집계 — 결정론, LLM 불필요."""
    since = asof_day - timedelta(days=WINDOW_DAYS)
    entries = [
        e
        for e in store.query(market, store="episodic")
        if e.outcome is not None
        and since < e.day <= asof_day
        and e.data.get("kind") != "weekly_reflection"
    ]
    if len(entries) < MIN_ENTRIES:
        return None

    outcomes = [e.outcome for e in entries]
    signal_credit: dict[str, list[float]] = {}
    memory_credit: dict[str, list[float]] = {}
    for e in entries:
        for sid in e.data.get("cited_signal_ids", []):
            signal_credit.setdefault(sid, []).append(e.outcome)
        for mid in e.data.get("cited_memory_ids", []):
            memory_credit.setdefault(mid, []).append(e.outcome)

    def _agg(values: dict[str, list[float]]) -> dict[str, dict]:
        return {
            k: {"n": len(v), "mean_outcome": round(sum(v) / len(v), 5)}
            for k, v in sorted(values.items())
        }

    return {
        "market": market,
        "window": [since.isoformat(), asof_day.isoformat()],
        "n_decisions": len(entries),
        "mean_outcome": round(sum(outcomes) / len(outcomes), 5),
        "win_rate": round(sum(1 for o in outcomes if o > 0) / len(outcomes), 3),
        "signal_credit": _agg(signal_credit),
        "memory_credit": _agg(memory_credit),
    }


def apply_importance_updates(store: MemoryStore, market: str, report: dict) -> list[dict]:
    """인용 메모리 importance 를 기여 방향으로 bounded 갱신 (retrieval 스코어 입력)."""
    events = []
    for mem_id, stats in report.get("memory_credit", {}).items():
        entry = store.get(mem_id)
        if entry is None or entry.market != market:
            continue
        direction = 1 if stats["mean_outcome"] > 0 else -1
        new_imp = min(
            IMPORTANCE_MAX, max(IMPORTANCE_MIN, entry.importance + IMPORTANCE_STEP * direction)
        )
        if new_imp != entry.importance:
            store.update(mem_id, importance=new_imp)
            events.append(
                {"event": "importance_updated", "id": mem_id,
                 "from": entry.importance, "to": new_imp}
            )
    return events


async def run_weekly(
    store: MemoryStore, market: str, asof_day: date, router=None
) -> tuple[dict | None, list[dict]]:
    """주간 리포트 생성 + credit 적용 + episodic 저장. (report, events) 반환.

    같은 주(asof) 리포트가 이미 있으면 멱등 스킵.
    """
    existing = store.query(market, store="episodic", day=asof_day)
    if any(e.data.get("kind") == "weekly_reflection" for e in existing):
        return None, []

    report = compute_weekly_report(store, market, asof_day)
    if report is None:
        return None, []
    events = apply_importance_updates(store, market, report)

    summary = ""
    if router is not None:
        try:
            import json

            resp = await router.complete(
                "fast",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "다음 주간 매매 리포트를 3문장 이내 한국어로 요약하라. "
                            "잘 통한 신호와 안 통한 신호를 구분할 것.\n"
                            + json.dumps(report, ensure_ascii=False)
                        ),
                    }
                ],
                max_tokens=2048,
            )
            summary = resp.text.strip()[:500]
        except Exception:
            summary = ""  # 요약 실패는 비치명

    store.add(
        market,
        "episodic",
        asof_day,
        content=f"[{market} 주간 reflection {asof_day}] " + (summary or "통계 리포트"),
        data={"kind": "weekly_reflection", "report": report},
        importance=0.6,
    )
    return report, events
