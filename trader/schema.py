"""결정 출력 스키마 (R5) — LLM 출력을 구조화 검증한다.

인용 memory/신호 ID 는 credit assignment 의 원천 데이터(하드룰 10) — 필드 자체를
스키마에 강제해 Phase 2+ 에서 소급 가능하게 한다 (Phase 1 은 빈 리스트 허용).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from adapters.allocation import CASH
from harness.loop import AllocationError, validate_weights


class DecisionParseError(ValueError):
    """LLM 출력이 결정 스키마를 위반 — 호출측은 주문 없이 스텝을 실패시킨다(안전 no-op)."""


@dataclass(frozen=True)
class TradeDecision:
    allocation: dict[str, float]
    rationale: str
    scenario_expected: str
    scenario_invalidation: str  # 무효화 조건 — 사후 검증(reflection)의 기준점
    cited_memory_ids: list[str] = field(default_factory=list)
    cited_signal_ids: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict:
    """코드펜스/전후 산문을 걷어내고 첫 JSON 객체를 파싱."""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise DecisionParseError(f"JSON 객체 없음: {text[:120]!r}")
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise DecisionParseError(f"JSON 파싱 실패: {e}") from e


def parse_decision(text: str, universe: list[str]) -> TradeDecision:
    """LLM 텍스트 → 검증된 TradeDecision.

    검증: 필수 필드 존재 · 배분 대상 ⊆ universe ∪ {CASH} · ∑=1 · long-only (R5, R6).
    """

    data = _extract_json(text)

    allocation = data.get("allocation")
    if not isinstance(allocation, dict) or not allocation:
        raise DecisionParseError("allocation 누락 또는 비어 있음")
    allowed = set(universe) | {CASH}
    unknown = set(allocation) - allowed
    if unknown:
        raise DecisionParseError(f"universe 밖 심볼: {sorted(unknown)}")
    try:
        allocation = {s: float(w) for s, w in allocation.items()}
        validate_weights(allocation)
    except (TypeError, ValueError, AllocationError) as e:
        raise DecisionParseError(f"배분 벡터 위반: {e}") from e

    scenario = data.get("scenario") or {}
    rationale = data.get("rationale", "").strip()
    expected = str(scenario.get("expected", "")).strip()
    invalidation = str(scenario.get("invalidation", "")).strip()
    if not rationale or not expected or not invalidation:
        raise DecisionParseError("rationale/scenario.expected/scenario.invalidation 필수")

    return TradeDecision(
        allocation=allocation,
        rationale=rationale,
        scenario_expected=expected,
        scenario_invalidation=invalidation,
        cited_memory_ids=[str(x) for x in data.get("cited_memory_ids") or []],
        cited_signal_ids=[str(x) for x in data.get("cited_signal_ids") or []],
    )
