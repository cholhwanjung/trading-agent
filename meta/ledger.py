"""LLM 메타 제안 원장 — `data/state/meta_shadow_llm.json`.

`eval.meta.load_meta_shadow` 가 읽는 모양({"history": [{"day", "weights", …}]})을 그대로
지켜, 결정론 국면 틸트 원장(`meta_shadow.json`)과 같은 채점기를 통과한다. 주 1회 제안이라
같은 ISO 주의 재실행은 그 주 레코드를 갱신한다(멱등).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def week_key(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def load_llm_ledger(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("history") or []


def record_llm_proposal(path: Path | str, record: dict) -> bool:
    """레코드(day·weights 필수)를 주 단위로 기록. 같은 주가 있으면 교체. 성공 True."""
    if not record.get("day") or not record.get("weights"):
        return False
    p = Path(path)
    week = week_key(date.fromisoformat(record["day"]))
    history = [h for h in load_llm_ledger(p) if h.get("week") != week]
    history.append({**record, "week": week})
    history.sort(key=lambda h: h["day"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"history": history}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )
    return True


def last_record(path: Path | str) -> dict | None:
    history = load_llm_ledger(path)
    return history[-1] if history else None


def save_ledger(path: Path | str, history: list[dict]) -> None:
    """결과 기입 등 레코드 일괄 갱신용."""
    Path(path).write_text(
        json.dumps(
            {"history": sorted(history, key=lambda h: h["day"])},
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
