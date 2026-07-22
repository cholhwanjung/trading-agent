"""Retrieval — FinMem 3항 스코어 (recency · relevancy · importance).

**active(=admission+probation 통과) 엔트리만 반환** — 검증 전 메모리는 결정에
개입할 수 없다는 원칙이 여기서 구조적으로 집행된다. probation/retired 는
retrieval 대상 자체가 아니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from memory.admission import cosine
from memory.store import MemoryEntry, MemoryStore

RECENCY_TAU_DAYS = 45.0  # 스코어 감쇠 (retention 은 diversity 기준 — 별개)


def build_query_text(market: str, features: dict[str, dict | None]) -> str:
    """관측 feature → relevancy 질의 텍스트 (결정론).

    저장측 임베딩(episodic content·rationale)과 같은 도메인 어휘(rsi·macd·낙폭·배분)로
    구성해 코사인 비교가 의미를 갖게 한다. LLM 호출 없음 — 임베딩 1회만 쓴다.
    """
    valid = {s: f for s, f in features.items() if f}
    if not valid:
        return f"[{market}] 관측 feature 없음(이력 부족) 상태에서의 배분 결정"
    parts = [
        f"{s} rsi={f['rsi_14']:.0f} macd_hist={f['macd_hist']:+.3f}"
        f" 20일수익률={f['ret_20d']:+.1%} 60일낙폭={f['drawdown_60d']:+.1%}"
        for s, f in sorted(valid.items())
    ]
    return f"[{market}] 현재 시장 상태에서의 배분 결정: " + " / ".join(parts)


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
