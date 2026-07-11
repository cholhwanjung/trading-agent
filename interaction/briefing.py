"""일일 브리핑 — 결정 + 근거 + 성과의 결정론 요약 (설계 §3.7 ③).

LLM 없이 로그·상태에서 조립한다(감사 가능). 일일 루프 말미에 자동 생성되어
data/briefings/{date}.md 로 저장 + 게이트웨이 GET /briefing 이 서빙한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from interaction.context import build_context


def build_briefing(root: Path | str) -> str:
    root = Path(root)
    context = build_context(root)
    by_kind: dict[str, list[dict]] = {}
    for item in context["items"]:
        by_kind.setdefault(item["kind"], []).append(item)

    today = datetime.now(timezone.utc).date().isoformat()
    lines = [f"# 일일 브리핑 {today}", ""]

    decisions = by_kind.get("decision", [])
    latest_by_market: dict[str, dict] = {}
    for d in decisions:
        latest_by_market[d["id"].split(":")[1]] = d
    for market, d in sorted(latest_by_market.items()):
        c = d["content"]
        lines += [
            f"## {market} — 결정 ({c['day']}) `{d['id']}`",
            f"- 배분: `{c['weights']}`",
            f"- 근거: {c['rationale'] or '(없음)'}",
            f"- 인용 신호: {c['cited_signal_ids'] or []} · 인용 메모리: {c['cited_memory_ids'] or []}",
            f"- 무효화 조건: {c['scenario_invalidation'] or '(없음)'}",
            f"- risk 위반: {c['risk_violations'] or '없음'} · 주문 접수: {c['accepted']}",
            "",
        ]

    equity = by_kind.get("equity", [])
    if equity:
        lines += ["## 성과 (가상 arm, 초기 $100k)", "", "| id | days | equity | ret% |", "|---|---|---|---|"]
        for e in sorted(equity, key=lambda x: x["id"]):
            c = e["content"]
            lines.append(f"| `{e['id']}` | {c['days']} | {c['equity']:,.0f} | {c['ret_pct']:+.3f} |")
        lines.append("")

    factors = by_kind.get("alpha_factor", [])
    if factors:
        lines += ["## Active 팩터", ""]
        for f in factors:
            c = f["content"]
            lines.append(f"- `{f['id']}` OOS IC {c['oos_ic']:+.4f} — {c['hypothesis']}")
        lines.append("")

    lessons = by_kind.get("memory_semantic", []) + by_kind.get("memory_procedural", [])
    lines.append(f"## 메모리: 승격 교훈 {len(lessons)}건")
    for m in lessons:
        lines.append(f"- `{m['id']}` [{m['content']['status']}] {m['content']['text'][:100]}")
    return "\n".join(lines) + "\n"


def write_briefing(root: Path | str) -> Path:
    root = Path(root)
    out = root / "data" / "briefings" / f"{datetime.now(timezone.utc).date().isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_briefing(root), encoding="utf-8")
    return out
