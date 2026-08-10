"""grounded context 조립 — 게이트웨이 답변의 유일한 사실원.

브로커 API 를 직접 치지 않는다 — 일일 루프가 갱신하는 로그·상태 파일과 메모리
store 만 읽는다(결정론·감사 가능). 모든 항목은 인용 가능한 안정 ID 를 갖는다:

    decision:{market}:{day}      일일 결정 (배분·근거·인용·risk)
    fundamentals:{market}:{day}  결정에 주입된 재무 비율 (PER·ROE·부채비율)
    disclosures:{market}:{day}   관측 창의 규제기관 접수 공시 (제목·접수일)
    risk:{market}                현재 목표 배분 + equity 고점
    equity:{market}:{arm}        가상 arm 성과 (llm/llm_base/bh/random)
    alpha:{name}                 active 팩터 (OOS IC·가설)
    mem_*                        메모리 엔트리 (id 그대로)

재무·공시를 별도 항목으로 두는 이유: 결정 경로는 이 둘을 관측하는데 챗은 못 보던 시기가
있었고, 그 상태에서 "왜 이 종목을 줄였나" 같은 질문에는 배분·근거만으로 답할 수밖에 없어
답변이 안전마진 판단의 실제 재료를 인용하지 못했다. 결정 항목 안에 묻어두지 않고 따로
꺼내는 것은 답변이 결정 전체를 끌어오지 않고 그 근거만 인용할 수 있게 하기 위함이다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from adapters.base import DISCLOSURE_SOURCES
from harness.jsonlog import iter_events
from memory import MemoryStore

MARKETS = ("CRYPTO", "US", "KR")
RECENT_DECISIONS = 5
RECENT_EPISODIC = 5
RECENT_DISCLOSURES = 10


def _read_jsonl_decisions(log_dir: Path, market: str) -> list[dict]:
    return list(iter_events(log_dir, market, "daily_step"))[-RECENT_DECISIONS:]


def _latest_fundamentals(records: list[dict]) -> tuple[str, dict] | None:
    """최근 결정들 중 재무가 실린 **가장 마지막** 것 → (day, 비율들).

    최신 결정이 비어 있을 수 있다(어댑터 실패는 fail-open 이라 재무 없이도 결정이 난다).
    분기 재무는 며칠 묵어도 유효하므로 비어 있으면 하루씩 거슬러 올라간다.
    """
    for rec in reversed(records):
        fundamentals = (rec.get("decision") or {}).get("fundamentals")
        if fundamentals:
            return str(rec.get("asof_day", ""))[:10], fundamentals
    return None


def _snapshot_disclosures(root: Path, market: str) -> tuple[str, list[dict]] | None:
    """가장 최근 관측 스냅샷의 공시 → (asof_day, 항목들). 뉴스는 제외한다.

    스냅샷은 일일 루프가 남기는 '그때 실제로 본 관측'이라 브로커를 다시 치지 않아도 된다.
    """
    market_dir = root / "data" / "state" / "observations" / market
    if not market_dir.exists():
        return None
    for path in sorted(market_dir.glob("*.json"), reverse=True):
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items = [
            {"day": str(n.get("published_at", ""))[:10], "headline": n.get("headline")}
            for n in snapshot.get("news") or []
            if n.get("source") in DISCLOSURE_SOURCES
        ]
        if items:
            return str(snapshot.get("asof_day", path.stem))[:10], items[:RECENT_DISCLOSURES]
    return None


def build_context(root: Path | str, markets: tuple[str, ...] = MARKETS) -> dict:
    """{"generated_at", "items": [{"id", "kind", "content"}...]}."""
    root = Path(root)
    items: list[dict] = []

    for market in markets:
        # 결정 로그
        decisions = _read_jsonl_decisions(root / "data" / "logs", market)
        for rec in decisions:
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
        # 재무 비율 — 결정에 실제로 주입된 값(원시 재무가 아니라 계산된 비율)
        latest = _latest_fundamentals(decisions)
        if latest:
            day, fundamentals = latest
            items.append(
                {
                    "id": f"fundamentals:{market}:{day}",
                    "kind": "fundamentals",
                    "content": {"day": day, "by_symbol": fundamentals},
                }
            )
        # 공시 — 관측 스냅샷에서(뉴스와 분리)
        disclosures = _snapshot_disclosures(root, market)
        if disclosures:
            day, entries = disclosures
            items.append(
                {
                    "id": f"disclosures:{market}:{day}",
                    "kind": "disclosures",
                    "content": {"day": day, "items": entries},
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
