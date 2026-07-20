"""grounded context 조립 — 게이트웨이 답변의 유일한 사실원 (R15).

브로커 API 를 직접 치지 않는다 — 일일 루프가 갱신하는 로그·상태 파일과 메모리
store 만 읽는다(결정론·감사 가능). 모든 항목은 인용 가능한 안정 ID 를 갖는다:

    decision:{market}:{day}   일일 결정 (배분·근거·인용·risk)
    risk:{market}             현재 목표 배분 + equity 고점
    equity:{market}:{arm}     가상 arm 성과 (llm/llm_base/bh/random)
    alpha:{name}              active 팩터 (OOS IC·가설)
    mem_*                     메모리 엔트리 (id 그대로)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from memory import MemoryStore

MARKETS = ("CRYPTO", "US", "KR")
RECENT_DECISIONS = 5
RECENT_EPISODIC = 5


def _read_jsonl_decisions(log_dir: Path, market: str) -> list[dict]:
    records = []
    market_dir = log_dir / market
    if not market_dir.exists():
        return records
    for path in sorted(market_dir.glob("*.jsonl"))[-7:]:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "daily_step":
                records.append(rec)
    return records[-RECENT_DECISIONS:]


def build_context(root: Path | str, markets: tuple[str, ...] = MARKETS) -> dict:
    """{"generated_at", "items": [{"id", "kind", "content"}...]}."""
    root = Path(root)
    items: list[dict] = []

    for market in markets:
        # 결정 로그
        for rec in _read_jsonl_decisions(root / "data" / "logs", market):
            day = str(rec.get("asof_day", ""))[:10]
            decision = rec.get("decision") or {}
            items.append(
                {
                    "id": f"decision:{market}:{day}",
                    "kind": "decision",
                    "content": {
                        "day": day,
                        "policy": rec.get("policy"),
                        "weights": rec.get("weights"),
                        "rationale": decision.get("rationale"),
                        "cited_signal_ids": decision.get("cited_signal_ids"),
                        "cited_memory_ids": decision.get("cited_memory_ids"),
                        "scenario_invalidation": decision.get("scenario_invalidation"),
                        "risk_violations": decision.get("risk_violations"),
                        "accepted": rec.get("accepted"),
                    },
                }
            )
        # 리스크 상태 (현재 목표 배분)
        risk_path = root / "data" / "state" / f"risk_{market}.json"
        if risk_path.exists():
            state = json.loads(risk_path.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": f"risk:{market}",
                    "kind": "risk_state",
                    "content": {
                        "target_weights": state.get("prev_weights"),
                        "peak_equity": state.get("peak_equity"),
                    },
                }
            )
        # 가상 arm 성과
        for arm in ("llm", "llm_base", "bh", "random"):
            arm_path = root / "data" / "state" / "virtual" / f"{market}_{arm}.json"
            if not arm_path.exists():
                continue
            history = json.loads(arm_path.read_text(encoding="utf-8")).get("history") or []
            if not history:
                continue
            items.append(
                {
                    "id": f"equity:{market}:{arm}",
                    "kind": "equity",
                    "content": {
                        "days": len(history),
                        "equity": history[-1]["equity"],
                        "ret_pct": round((history[-1]["equity"] / 100_000 - 1) * 100, 4),
                        "last_day": history[-1]["day"],
                    },
                }
            )

    # active 팩터
    lib_path = root / "data" / "state" / "alpha_library_CRYPTO.json"
    if lib_path.exists():
        for f in json.loads(lib_path.read_text(encoding="utf-8"))["factors"]:
            if f.get("status") == "active":
                items.append(
                    {
                        "id": f"alpha:{f['name']}",
                        "kind": "alpha_factor",
                        "content": {
                            "hypothesis": f.get("hypothesis"),
                            "oos_ic": f.get("oos_ic"),
                            "oos_icir": f.get("oos_icir"),
                        },
                    }
                )

    # 메모리 (검증 통과 교훈 전부 + 최근 episodic)
    db_path = root / "data" / "memory.sqlite"
    if db_path.exists():
        store = MemoryStore(db_path)
        try:
            for market in markets:
                for store_name in ("semantic", "procedural"):
                    for e in store.query(market, store=store_name):
                        if e.status in ("active", "probation"):
                            items.append(
                                {
                                    "id": e.id,
                                    "kind": f"memory_{store_name}",
                                    "content": {"text": e.content, "status": e.status,
                                                "importance": e.importance},
                                }
                            )
                episodic = store.query(market, store="episodic")[-RECENT_EPISODIC:]
                for e in episodic:
                    items.append(
                        {
                            "id": e.id,
                            "kind": "memory_episodic",
                            "content": {"text": e.content, "outcome": e.outcome},
                        }
                    )
        finally:
            store.close()

    return {"generated_at": datetime.now(timezone.utc).isoformat(), "items": items}


def allowed_ids(context: dict) -> set[str]:
    return {item["id"] for item in context["items"]}
