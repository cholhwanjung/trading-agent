"""월간 self-improve 제안 생성 — **자동 적용 경로 없음**.

사용법:
    uv run python scripts/propose_improvements.py

승격된 교훈·주간 리포트·ablation 지표를 모아 시스템 프롬프트/플레이북 개선안을
LLM 이 *제안서*(data/proposals/{YYYY-MM}.md)로 작성한다. 적용은 사용자가 제안서를
읽고 코드/문서를 직접 수정할 때만 일어난다 — 이 스크립트는 어떤 프롬프트/설정
파일도 수정하지 않는다 (검증 안 된 자기서사 축적 방지).
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import load_env, make_usage_sink  # noqa: E402
from llm import LLMRouter  # noqa: E402
from memory import MemoryStore  # noqa: E402

PROPOSAL_DIR = ROOT / "data" / "proposals"
MARKETS = ("CRYPTO", "US", "KR")


def gather_evidence(memory: MemoryStore) -> dict:
    evidence: dict = {"markets": {}}
    for market in MARKETS:
        lessons = [
            {"content": e.content, "importance": e.importance, "status": e.status}
            for store_name in ("semantic", "procedural")
            for e in memory.query(market, store=store_name)
            if e.status in ("active", "probation")
        ]
        weeklies = [
            e.data.get("report")
            for e in memory.query(market, store="episodic")
            if e.data.get("kind") == "weekly_reflection"
        ][-4:]
        evidence["markets"][market] = {"lessons": lessons, "weekly_reports": weeklies}
    lib_path = ROOT / "data" / "state" / "alpha_library_CRYPTO.json"
    if lib_path.exists():
        lib = json.loads(lib_path.read_text(encoding="utf-8"))
        evidence["alpha_factors"] = [
            {k: f[k] for k in ("name", "oos_ic", "oos_icir", "status")} for f in lib["factors"]
        ]
    return evidence


async def main() -> int:
    env = load_env(ROOT / ".env")
    memory = MemoryStore(ROOT / "data" / "memory.sqlite")
    router = LLMRouter(env, usage_sink=make_usage_sink(ROOT))
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    out_path = PROPOSAL_DIR / f"{month}.md"

    try:
        evidence = gather_evidence(memory)
        n_lessons = sum(len(m["lessons"]) for m in evidence["markets"].values())
        n_weekly = sum(len(m["weekly_reports"]) for m in evidence["markets"].values())
        if n_lessons == 0 and n_weekly == 0:
            print("status=skip detail=증거 없음 (승격 교훈·주간 리포트 0건) — 제안할 근거가 없다")
            return 0

        resp = await router.complete(
            "smart",
            purpose="self_improve",
            system=(
                "너는 트레이딩 에이전트의 self-improve 리뷰어다. 검증된 증거에서만 개선안을 "
                "도출하라. 각 제안은 [근거 → 제안 diff → 예상 효과 → 리스크] 구조. "
                "증거 없는 제안 금지. 마크다운으로 출력."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{month} 월간 증거:\n" + json.dumps(evidence, ensure_ascii=False, indent=1)
                        + "\n\n시스템 프롬프트(trader/agent.py SYSTEM_PROMPT)·Risk limits·"
                        "관측 구성에 대한 개선 제안서를 작성하라."
                    ),
                }
            ],
            max_tokens=8192,
        )
        PROPOSAL_DIR.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Self-improve 제안 {month}\n\n"
            "> **자동 적용되지 않는다**. 검토 후 채택분만 사용자가 직접 반영하고,\n"
            "> 반영 시 변경 로그에 기록할 것. 정책 변경의 최종 판정은 라이브 A/B.\n\n"
        )
        out_path.write_text(header + resp.text, encoding="utf-8")
        print(f"status=ok proposal={out_path} lessons={n_lessons} weekly={n_weekly}")
        return 0
    finally:
        memory.close()
        await router.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
