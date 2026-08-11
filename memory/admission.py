"""Admission 게이트 — episodic → semantic/procedural 승격.

4단계: ① 반복 n≥5 ② 부호검정 p≤0.05 ③ 임베딩 중복 기각 ④ probation(라이브 OOS 유예).
비대칭(APV): 유의한 성공 패턴 → semantic(소프트 프라이어),
유의한 실패 패턴 → procedural Forbidden(고신뢰 시 하드 veto 소스).
단일 매매 승격은 구조적으로 불가 — n 과 유의성이 게이트다.

동점(outcome == 0)은 표본에서 제외한다 — 부호검정 표준. outcome 은 행동 수익 −
무행동 수익이므로 0 은 "행동이 결과를 바꾸지 못했다"는 뜻이지 실패가 아니다. 휴장일
결정(가격 불변)과 배분 무변동 결정이 여기 해당하는데, 이를 음(-)으로 세면 Forbidden
쪽으로만 편향된다 — 동점 4건에 실제 음수 1건이면 p=0.031 로 하드 veto 가 서는 식이다.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from memory.store import MemoryEntry, MemoryStore

MIN_N = 5
ALPHA = 0.05
DUP_COSINE = 0.90
PROBATION_DAYS = 7
MIN_PROBATION_N = 2


def sign_test_p(k: int, n: int) -> float:
    """단측 부호검정: P(X ≥ k), X~Bin(n, 0.5). 표준 라이브러리만 사용."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _is_duplicate(
    candidate_embedding: list[float] | None,
    candidate_key: str,
    existing: list[MemoryEntry],
) -> bool:
    """기존 승격 엔트리와 의미 중복이면 기각. 임베딩 없으면 pattern_key 일치로 폴백."""
    for entry in existing:
        if entry.status == "retired":
            continue
        if entry.data.get("pattern_key") == candidate_key:
            return True
        if (
            candidate_embedding
            and entry.embedding
            and cosine(candidate_embedding, entry.embedding) >= DUP_COSINE
        ):
            return True
    return False


def promote_candidates(
    store: MemoryStore,
    market: str,
    asof_day: date,
    min_n: int = MIN_N,
    alpha: float = ALPHA,
) -> list[dict]:
    """패턴별 통계 검증 후 probation 상태로 승격. 결과 이벤트 리스트 반환(로그용)."""
    episodic = [
        e
        for e in store.query(market, store="episodic", status="active")
        if e.outcome is not None and e.outcome != 0 and e.pattern_key
    ]
    by_pattern: dict[str, list[MemoryEntry]] = {}
    for e in episodic:
        by_pattern.setdefault(e.pattern_key, []).append(e)

    existing = store.query(market, store="semantic") + store.query(market, store="procedural")
    already_promoted = {
        src for entry in existing for src in entry.data.get("source_ids", [])
    }
    events: list[dict] = []

    for key, entries in by_pattern.items():
        fresh = [e for e in entries if e.id not in already_promoted]
        n = len(fresh)
        if n < min_n:
            continue
        k_pos = sum(1 for e in fresh if e.outcome > 0)
        mean = sum(e.outcome for e in fresh) / n
        p_pos = sign_test_p(k_pos, n)
        p_neg = sign_test_p(n - k_pos, n)

        if p_pos <= alpha and mean > 0:
            target, kind = "semantic", "success"
            p = p_pos
        elif p_neg <= alpha and mean < 0:
            target, kind = "procedural", "forbidden"  # APV: 실패는 하드 veto 소스
            p = p_neg
        else:
            continue

        embedding = next((e.embedding for e in fresh if e.embedding), None)
        if _is_duplicate(embedding, key, existing):
            events.append({"event": "admission_dup_rejected", "pattern": key})
            continue

        content = (
            f"[{market}] 패턴 {key}: n={n}, 양성 {k_pos}/{n}, p={p:.4f}, "
            f"평균 초과수익 {mean:+.4f} → {'성공 프라이어' if kind == 'success' else 'Forbidden(실패)'}"
        )
        entry_id = store.add(
            market,
            target,
            asof_day,
            content,
            data={
                "pattern_key": key,
                "kind": kind,
                "n": n,
                "k_pos": k_pos,
                "p_value": p,
                "mean_outcome": mean,
                "source_ids": [e.id for e in fresh],
                "promoted_day": asof_day.isoformat(),
                "probation_until": (asof_day + timedelta(days=PROBATION_DAYS)).isoformat(),
            },
            pattern_key=key,
            status="probation",
            importance=min(0.9, 0.5 + abs(mean) * 10),
            embedding=embedding,
        )
        events.append(
            {"event": "admission_promoted", "id": entry_id, "pattern": key, "kind": kind,
             "n": n, "p": round(p, 4), "mean": round(mean, 5)}
        )
    return events


def review_probation(store: MemoryStore, market: str, asof_day: date) -> list[dict]:
    """probation 만료 엔트리를 라이브 OOS 로 재검증 — 유지되면 active, 아니면 retired."""
    events: list[dict] = []
    for target in ("semantic", "procedural"):
        for entry in store.query(market, store=target, status="probation"):
            until = date.fromisoformat(entry.data["probation_until"])
            if asof_day < until:
                continue
            promoted = date.fromisoformat(entry.data["promoted_day"])
            source_ids = set(entry.data.get("source_ids", []))
            oos = [
                e
                # veto 로 미집행된 원안도 표본이다 — 하드 veto 가 서면 그 패턴의 집행
                # 기록이 끊겨 검증 자체가 불가능해진다. 이 표본은 제약을 푸는 방향으로만
                # 작동한다(승격 경로는 episodic 만 본다).
                for name in ("episodic", "counterfactual")
                for e in store.query(market, store=name, pattern_key=entry.pattern_key)
                # 승격에 쓰인 원천 표본은 OOS 가 아니다 — in-sample 재사용 = 자기 검증 오염.
                # 동점도 제외 — 전부 동점이면 oos_mean == 0 이라 부호 판정이 무조건 실패로
                # 떨어져, 검증되지 않은 패턴이 아니라 '검증할 기회가 없던' 패턴이 퇴출된다.
                if e.outcome is not None
                and e.outcome != 0
                and e.id not in source_ids
                and promoted < e.day <= asof_day
            ]
            expected_sign = 1 if entry.data["kind"] == "success" else -1
            if len(oos) >= MIN_PROBATION_N:
                oos_mean = sum(e.outcome for e in oos) / len(oos)
                verdict = "active" if oos_mean * expected_sign > 0 else "retired"
            else:
                # OOS 표본 부족 — 패턴 미출현은 무효 증거가 아니므로 유예 연장
                extended = (asof_day + timedelta(days=PROBATION_DAYS)).isoformat()
                store.update(entry.id, data={**entry.data, "probation_until": extended})
                events.append({"event": "probation_extended", "id": entry.id, "oos_n": len(oos)})
                continue
            store.update(entry.id, status=verdict)
            events.append(
                {"event": f"probation_{verdict}", "id": entry.id,
                 "oos_n": len(oos), "oos_mean": round(oos_mean, 5)}
            )
    return events
