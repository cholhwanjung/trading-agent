"""제안 결과 기입과 신호 신뢰도 — 결정론 채점기의 제안 단위 판.

지평 H 가 지난 제안마다 **anchor 대비 실현 델타**를 가상 arm equity 로 소급 기입한다.
델타 = Σ_m (w_m − anchor_m) · r_m, r_m = 시장 m 의 [day, day+H] 수익률. 틸트한 시장의
방향 적중(증액 시장이 세 시장 평균을 이겼나)과, 인용 신호 **유형별** 적중을 누적해
다음 제안에 신뢰도 표로 되돌린다 — 계산값이지 LLM 이 쓴 문장이 아니다.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from eval.meta import _eq_at, load_arm_history
from meta.ledger import load_llm_ledger, save_ledger
from meta.panel import signal_type

HORIZON_DAYS = 28
RELIABILITY_WEEKS = 26


def _realized(
    histories: dict[str, list[dict]], day: date, horizon_days: int
) -> dict[str, float] | None:
    end = day + timedelta(days=horizon_days)
    out: dict[str, float] = {}
    for m, h in histories.items():
        if not h or date.fromisoformat(h[-1]["day"]) < end:
            return None  # 지평이 아직 안 찼다
        e0, e1 = _eq_at(h, day.isoformat()), _eq_at(h, end.isoformat())
        if not e0 or not e1:
            return None
        out[m] = e1 / e0 - 1.0
    return out


def fill_outcomes(
    ledger_path: Path | str,
    state_dir: Path | str,
    arm: str = "llm",
    horizon_days: int = HORIZON_DAYS,
) -> list[dict]:
    """결과 없는 레코드 중 지평이 찬 것에 outcome 을 기입. 기입 이벤트 목록 반환."""
    history = load_llm_ledger(ledger_path)
    if not history:
        return []
    markets = sorted(history[-1]["weights"])
    histories = {m: load_arm_history(Path(state_dir), m, arm) for m in markets}
    events: list[dict] = []
    for rec in history:
        if rec.get("outcome") or not rec.get("anchor"):
            continue
        rets = _realized(histories, date.fromisoformat(rec["day"]), horizon_days)
        if rets is None:
            continue
        anchor, w = rec["anchor"], rec["weights"]
        delta = sum((w[m] - anchor[m]) * rets[m] for m in markets)
        mean_ret = sum(rets.values()) / len(rets)
        hits = {}
        for m in markets:
            tilt = w[m] - anchor[m]
            if abs(tilt) < 1e-6:
                continue
            hits[m] = (rets[m] > mean_ret) if tilt > 0 else (rets[m] < mean_ret)
        rec["outcome"] = {
            "horizon_days": horizon_days,
            "arm": arm,
            "returns": {m: round(r, 6) for m, r in rets.items()},
            "delta_vs_anchor": round(delta, 6),
            "direction_hits": hits,
        }
        events.append(
            {
                "day": rec["day"],
                "week": rec.get("week"),
                "delta_vs_anchor": rec["outcome"]["delta_vs_anchor"],
                "direction_hits": hits,
            }
        )
    if events:
        save_ledger(ledger_path, history)
    return events


def signal_reliability(ledger_path: Path | str, weeks: int = RELIABILITY_WEEKS) -> dict:
    """최근 N 주 결과가 있는 레코드에서 인용 신호 유형별 (n, hits). hit = 델타 > 0."""
    history = [h for h in load_llm_ledger(ledger_path) if h.get("outcome")][-weeks:]
    table: dict[str, dict[str, int]] = {}
    for rec in history:
        hit = rec["outcome"]["delta_vs_anchor"] > 0
        for t in {signal_type(c) for c in rec.get("cited") or []}:
            row = table.setdefault(t, {"n": 0, "hits": 0})
            row["n"] += 1
            row["hits"] += int(hit)
    overall = {
        "n": len(history),
        "hits": sum(1 for r in history if r["outcome"]["delta_vs_anchor"] > 0),
    }
    return {"weeks": weeks, "overall": overall, "by_type": table}


def save_reliability(path: Path | str, table: dict) -> None:
    Path(path).write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")


def reliability_lines(table: dict) -> list[str]:
    """프롬프트 주입용 라벨 — 표본 0 인 유형은 넣지 않는다."""
    lines = []
    for t, row in sorted((table.get("by_type") or {}).items()):
        if row["n"] > 0:
            lines.append(
                f"{t}: 최근 {table['weeks']}주 제안 중 인용 {row['n']}회, 적중 {row['hits']}회"
            )
    return lines
