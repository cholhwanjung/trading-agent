"""제안 감사 — 결정론 검사가 먼저, LLM 은 의미 판정만.

결정론으로 판정할 수 있는 것(인용 id 실재·범위 초과·추세 추격·근거 없음)은 LLM 에게
묻지 않는다. 유일한 하드 거부 사유는 **날조 인용**이다 — "근거가 약하다" 같은 판단성 거부를
LLM 에게 주면 그것이 곧 LLM 이 통제하는 두 번째 배분기가 된다. 감사는 비중을 바꾸지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm import LLMRouter, extract_json
from meta.propose import MetaLLMProposal
from trader.prompt_store import PromptSpec

FLOW_CHASING_PERIOD = "12m"


@dataclass
class AuditResult:
    flags: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)
    reject: bool = False
    llm: dict | None = None


def deterministic_audit(proposal: MetaLLMProposal, valid_ids: set[str], panel: dict) -> AuditResult:
    """인용 실재 · clamp 여부 · 무인용 · 추세 추격(틸트 방향 = 전 시장 12개월 수익률 부호)."""
    result = AuditResult()
    result.fabricated = [c for c in proposal.cited if c not in valid_ids]
    if result.fabricated:
        result.flags.append("fabricated_citation")
        result.reject = True
    if proposal.clamped:
        result.flags.append("clamped")
    if not proposal.cited:
        result.flags.append("no_citation")
    anchor = panel["anchor"]
    tilts = {m: proposal.weights[m] - anchor[m] for m in anchor}
    if all(abs(t) < 1e-6 for t in tilts.values()):
        result.flags.append("anchor")
        return result
    moms = {
        m: (panel["markets"].get(m) or {}).get("trailing", {}).get(FLOW_CHASING_PERIOD)
        for m in anchor
    }
    tilted = [m for m, t in tilts.items() if abs(t) >= 1e-6 and moms.get(m) is not None]
    if tilted and all((tilts[m] > 0) == (moms[m] > 0) for m in tilted):
        result.flags.append("flow_chasing")
    return result


def _audit_prompt(proposal: MetaLLMProposal, panel: dict, digest: dict[str, list[dict]]) -> str:
    return json.dumps(
        {
            "proposal": {
                "weights": proposal.weights,
                "anchor": panel["anchor"],
                "cited": proposal.cited,
                "rationale": proposal.rationale,
            },
            "inputs_summary": {
                "markets": {
                    m: {
                        "regime": d.get("regime"),
                        "drawdown": d.get("drawdown"),
                        "trailing": d.get("trailing"),
                    }
                    for m, d in panel["markets"].items()
                },
                "macro": panel.get("macro"),
                "news": {m: [ln["text"] for ln in lines] for m, lines in digest.items()},
            },
        },
        ensure_ascii=False,
        indent=1,
        default=str,
    )


async def llm_audit(
    router: LLMRouter,
    spec: PromptSpec,
    proposal: MetaLLMProposal,
    panel: dict,
    digest: dict[str, list[dict]],
    purpose: str = "meta_audit",
) -> dict:
    """의미 판정 3종 + notes. 출력이 깨지면 빈 dict(감사 실패는 제안을 막지 않는다)."""
    resp = await router.complete(
        spec.tier,
        purpose=purpose,
        system=spec.text,
        messages=[{"role": "user", "content": _audit_prompt(proposal, panel, digest)}],
        json_mode=True,
        temperature=0.0,
    )
    data = extract_json(resp.text)
    if not isinstance(data, dict):
        return {}
    return {
        "forbidden_inputs_referenced": bool(data.get("forbidden_inputs_referenced")),
        "rationale_matches_weights": bool(data.get("rationale_matches_weights", True)),
        "relative_argument": bool(data.get("relative_argument", True)),
        "notes": str(data.get("notes", ""))[:300],
    }


def llm_flags(verdict: dict) -> list[str]:
    flags = []
    if verdict.get("forbidden_inputs_referenced"):
        flags.append("forbidden_inputs")
    if verdict and not verdict.get("rationale_matches_weights", True):
        flags.append("rationale_mismatch")
    if verdict and not verdict.get("relative_argument", True):
        flags.append("absolute_argument")
    return flags
