"""대시보드 데이터 로더 — 순수 stdlib(pandas/streamlit 미의존, 테스트 가능).

일일 루프가 남긴 관측 스냅샷(`data/state/observations/`)과 결정 로그
(`data/logs/`)를 읽어 UI 가 바로 쓰는 평범한 dict/list 로 정규화한다.
브로커 API 를 치지 않고, 파일만 읽는다(결정론·감사 가능).
"""

from __future__ import annotations

import json
from pathlib import Path


def list_observation_days(obs_dir: Path, market: str) -> list[str]:
    """해당 시장의 관측 스냅샷 날짜(YYYY-MM-DD)를 최신순으로 반환."""

    market_dir = obs_dir / market
    if not market_dir.exists():
        return []
    return sorted((p.stem for p in market_dir.glob("*.json")), reverse=True)


def load_observation(obs_dir: Path, market: str, day: str) -> dict | None:
    """{obs_dir}/{market}/{day}.json 스냅샷을 읽는다. 없으면 None."""

    path = obs_dir / market / f"{day}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_recent_decisions(log_dir: Path, market: str, limit: int = 30) -> list[dict]:
    """daily_step 로그 → asof_day 오름차순 정규화 결정 행.

    로그 파일명은 실행일이지만 여기서는 레코드의 asof_day 로 키를 잡는다
    (관측 스냅샷과 같은 축). 같은 asof 재실행은 최신 레코드로 덮어쓴다.
    """

    market_dir = log_dir / market
    if not market_dir.exists():
        return []
    rows: dict[str, dict] = {}
    for path in sorted(market_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "daily_step":
                continue
            day = str(rec.get("asof_day", ""))[:10]
            if not day:
                continue
            d = rec.get("decision") or {}
            rows[day] = {
                "day": day,
                "policy": rec.get("policy"),
                "weights": rec.get("weights") or {},
                "accepted": rec.get("accepted"),
                "features": d.get("features") or {},
                "rationale": d.get("rationale"),
                "debate": d.get("debate"),
                "risk_violations": d.get("risk_violations") or [],
                "weights_pre_risk": d.get("weights_pre_risk"),
                "mdd": d.get("mdd"),
                "cited_signal_ids": d.get("cited_signal_ids") or [],
                "cited_memory_ids": d.get("cited_memory_ids") or [],
            }
    return [rows[k] for k in sorted(rows)][-limit:]


def decision_for_day(rows: list[dict], day: str) -> dict | None:
    """정규화된 결정 행 목록에서 특정 날짜의 결정을 찾는다."""

    return next((r for r in rows if r["day"] == day), None)


def veto_rows(rows: list[dict]) -> list[dict]:
    """risk 위반이 있었던 날만 추려 타임라인 표로."""

    return [
        {"day": r["day"], "violations": "; ".join(r["risk_violations"]), "mdd": r.get("mdd")}
        for r in rows
        if r["risk_violations"]
    ]
