"""Retention — diversity 우선 퇴출 게이트 (QuantaAlpha 실증).

퇴출 기준은 두 가지뿐:
1. **중복성** — active 엔트리끼리 임베딩 유사도가 임계 초과면 신뢰도 낮은 쪽 퇴출.
   (population 을 줄이는 게 아니라 정보 중복만 제거 — diversity 보존)
2. **최근 라이브 유효성** — 승격 후 축적된 최근 OOS 표본이 기대 부호와 반대로
   유의하게 뒤집히면 퇴출.

recency 단독 퇴출은 구현하지 않는다 — 오래됐지만 유일한 교훈은 살아남는다(불변 규칙).
"""

from __future__ import annotations

from datetime import date, timedelta

from memory.admission import cosine
from memory.influence import lesson_confidence
from memory.store import MemoryStore

REDUNDANCY_COSINE = 0.95  # 중복 판정 임계 (admission DUP 0.90 보다 보수적)
VALIDITY_WINDOW_DAYS = 30
MIN_VALIDITY_N = 3


def review_retention(store: MemoryStore, market: str, asof_day: date) -> list[dict]:
    """active semantic/procedural 을 중복성 + 최근 유효성으로 심사. 이벤트 반환."""
    events: list[dict] = []

    for store_name in ("semantic", "procedural"):
        active = store.query(market, store=store_name, status="active")

        # 1. 최근 라이브 유효성 — 부호 뒤집힘 퇴출
        for entry in active:
            expected_sign = 1 if entry.data.get("kind") == "success" else -1
            since = asof_day - timedelta(days=VALIDITY_WINDOW_DAYS)
            source_ids = set(entry.data.get("source_ids", []))
            recent = [
                e
                for e in store.query(market, store="episodic", pattern_key=entry.pattern_key)
                if e.outcome is not None and e.id not in source_ids and since <= e.day <= asof_day
            ]
            if len(recent) < MIN_VALIDITY_N:
                continue  # 표본 부족 = 무효 증거 아님 — 유지 (diversity 보존)
            mean = sum(e.outcome for e in recent) / len(recent)
            if mean * expected_sign < 0:
                store.update(entry.id, status="retired")
                events.append(
                    {"event": "retention_invalidated", "id": entry.id,
                     "recent_n": len(recent), "recent_mean": round(mean, 5)}
                )

        # 2. 중복성 — 임베딩 근사 중복 쌍에서 신뢰도 낮은 쪽 퇴출
        survivors = store.query(market, store=store_name, status="active")
        retired_ids: set[str] = set()  # 이 심사에서 퇴출된 엔트리 — 재비교 제외
        for i, a in enumerate(survivors):
            for b in survivors[i + 1 :]:
                if not a.embedding or not b.embedding:
                    continue
                if a.id in retired_ids or b.id in retired_ids:
                    continue
                if cosine(a.embedding, b.embedding) >= REDUNDANCY_COSINE:
                    loser = a if lesson_confidence(a) <= lesson_confidence(b) else b
                    store.update(loser.id, status="retired")
                    retired_ids.add(loser.id)
                    events.append(
                        {"event": "retention_redundant", "id": loser.id,
                         "kept": (b if loser is a else a).id}
                    )
    return events
