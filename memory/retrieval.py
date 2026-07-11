"""Retrieval — FinMem 3항 스코어 (recency · relevancy · importance).

**active(=admission+probation 통과) 엔트리만 반환** — 검증 전 메모리는 결정에
개입할 수 없다는 하드룰 3 이 여기서 구조적으로 집행된다. probation/retired 는
retrieval 대상 자체가 아니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from memory.admission import cosine
from memory.store import MemoryEntry, MemoryStore

RECENCY_TAU_DAYS = 45.0  # 스코어 감쇠 (retention 은 diversity 기준 — ADR-005 와 별개)


@dataclass(frozen=True)
class ScoredMemory:
    entry: MemoryEntry
    score: float
    recency: float
    relevancy: float


def retrieve(
    store: MemoryStore,
    market: str,
    asof_day: date,
    query_embedding: list[float] | None = None,
    stores: tuple[str, ...] = ("semantic", "procedural"),
    k: int = 5,
) -> list[ScoredMemory]:
    scored: list[ScoredMemory] = []
    for store_name in stores:
        for entry in store.query(market, store=store_name, status="active"):
            days = max(0, (asof_day - entry.day).days)
            recency = math.exp(-days / RECENCY_TAU_DAYS)
            relevancy = (
                cosine(query_embedding, entry.embedding)
                if query_embedding and entry.embedding
                else 0.0
            )
            score = recency + relevancy + entry.importance
            scored.append(
                ScoredMemory(entry=entry, score=score, recency=recency, relevancy=relevancy)
            )
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:k]
