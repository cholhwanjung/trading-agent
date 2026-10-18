"""META 네임스페이스 메모리 — 제안을 episodic 으로 적고, 결과로 채우고, 같은 게이트로 승격한다.

시장 네임스페이스와 같은 `MemoryStore` 를 쓰되 `market="META"` 로 격리된다(조회 API 가
네임스페이스를 요구하므로 시장 메모리와 교차하지 않는다). episodic → admission(반복 n ·
부호검정 · 중복 · probation) → semantic/procedural 은 트레이더와 **같은 함수**다. 단일 제안으로는
아무것도 승격되지 않는다. 승격 교훈은 confidence 와 함께 제안자에 주입되고, 그 기여는
base(교훈 없음) 대비 confidence 로 축소된 bounded residual 로만 들어간다.
"""

from __future__ import annotations

from datetime import date

from adapters.allocation import CASH
from memory.influence import BlendResult
from memory import (
    MemoryStore,
    blend_allocations,
    lessons_payload,
    promote_candidates,
    retrieve,
    review_probation,
    review_retention,
)

META = "META"
VOL_BUCKETS = ((20.0, "low"), (40.0, "mid"))  # 60일 연율 변동성 평균(%) — 그 위는 high


def meta_pattern_key(
    regimes: dict[str, str | None], macro_state: str | None, vol60: dict[str, float | None]
) -> str:
    """(3시장 국면 조합, 매크로, 변동성 버킷) — 주간 표본에서 반복이 쌓이도록 거칠게."""
    r = "/".join((regimes.get(m) or "none")[:4].lower() for m in sorted(regimes))
    vols = [v for v in vol60.values() if v is not None]
    if vols:
        avg = sum(vols) / len(vols)
        bucket = next((b for lim, b in VOL_BUCKETS if avg < lim), "high")
    else:
        bucket = "unk"
    return f"regime={r}|macro={(macro_state or 'none').lower()}|vol={bucket}"


def record_meta_episode(store: MemoryStore, record: dict, pattern: str, day: date) -> str | None:
    """제안 1건을 episodic 으로. 같은 날은 멱등(None). outcome 은 지평 뒤 `fill_meta_outcomes`."""
    if store.query(META, store="episodic", day=day):
        return None
    weights = {m: round(v, 3) for m, v in record["weights"].items()}
    content = f"[META {day}] {pattern} — 제안 {weights}. 근거: {record.get('rationale', '')[:200]}"
    return store.add(
        META,
        "episodic",
        day,
        content,
        data={
            "week": record.get("week"),
            "weights": record["weights"],
            "anchor": record.get("anchor"),
            "cited_signal_ids": [c for c in record.get("cited") or [] if not c.startswith("mem_")],
            "cited_memory_ids": [c for c in record.get("cited") or [] if c.startswith("mem_")],
            "rationale": record.get("rationale", ""),
            "prompt_rev": record.get("prompt_rev"),
            "audit_flags": (record.get("audit") or {}).get("flags", []),
        },
        pattern_key=pattern,
    )


def fill_meta_outcomes(store: MemoryStore, ledger_history: list[dict]) -> list[tuple[str, float]]:
    """원장에 결과가 기입된 주의 episodic 에 outcome(anchor 대비 델타)을 소급 기입."""
    by_day = {h["day"]: h["outcome"]["delta_vs_anchor"] for h in ledger_history if h.get("outcome")}
    filled: list[tuple[str, float]] = []
    for entry in store.query(META, store="episodic", outcome_missing=True):
        delta = by_day.get(entry.day.isoformat())
        if delta is None:
            continue
        store.update(entry.id, outcome=delta)
        filled.append((entry.id, delta))
    return filled


def admit_meta(store: MemoryStore, today: date) -> list[dict]:
    """admission → probation 심사 → retention. 트레이더와 같은 게이트·같은 임계."""
    return (
        promote_candidates(store, META, today)
        + review_probation(store, META, today)
        + review_retention(store, META, today)
    )


def meta_lessons(store: MemoryStore, today: date, k: int = 5) -> list[dict]:
    """active 교훈만, confidence 동반. probation 중인 것은 제안에 들어가지 않는다."""
    return lessons_payload(retrieve(store, META, today, k=k))


def blend_meta(
    base: dict[str, float], mem: dict[str, float], cited_lessons: list[dict]
) -> tuple[dict[str, float], BlendResult]:
    """base + confidence·bounded(mem − base). 시장 비중만 돌려준다(현금 키 없음)."""
    result = blend_allocations(base, mem, cited_lessons)
    weights = {m: v for m, v in result.weights.items() if m != CASH}
    total = sum(weights.values())
    weights = {m: round(v / total, 6) for m, v in weights.items()} if total > 0 else dict(base)
    return weights, result
