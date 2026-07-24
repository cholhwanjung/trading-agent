"""조건부 Bull/Bear debate — 상시 멀티에이전트 배제.

소집 트리거(결정론, 이 밖에서는 절대 미발생):
① signal_conflict — alpha 신호에 유의 크기(|z|≥0.5)의 양·음수가 혼재
② large_position_change — 제안 배분이 직전 배분 대비 L1(CASH 포함) ≥ 0.4
③ user_request — 사용자 강제(--debate)

프로토콜: Bull/Bear 는 fast tier 병렬 1회씩 → 결정자(smart)가 토론 반영 재결정.
추가 LLM 호출은 트리거 시에만 3회(fast 2 + smart 1). 결론은 결정 메타에
구조화 기록(telephone effect 방지 — 자유 대화 전달 없음).
"""

from __future__ import annotations

import asyncio
import json

from llm import extract_json

SIGNAL_CONFLICT_MIN = 0.5  # z-score 크기 — 이 이상의 양·음 신호 혼재 = 충돌
LARGE_CHANGE_L1 = 0.4  # 직전 배분 대비 L1 (CASH 포함) — Risk turnover 캡(0.5) 직전 수준

BULL_SYSTEM = """\
너는 {market} 배분 토론의 낙관(Bull) 토론자다. 관측 데이터에 근거한 상방 논거만 제시한다.
제안된 배분이 상방 관점에서 충분히 공격적인지도 평가하라.
JSON 만 출력: {{"thesis": "<핵심 주장 1문장>", "points": ["<관측 인용 근거>", ...]}} (points 최대 3개)"""

BEAR_SYSTEM = """\
너는 {market} 배분 토론의 비관(Bear) 토론자다. 관측 데이터에 근거한 하방 리스크만 제시한다.
제안된 배분이 하방 관점에서 충분히 방어적인지도 평가하라.
JSON 만 출력: {{"thesis": "<핵심 주장 1문장>", "points": ["<관측 인용 근거>", ...]}} (points 최대 3개)"""


def _signal_scores(signals: dict) -> list[float]:
    """alpha 신호 dict → 숫자 스코어 평탄화 (형식: {name: {"scores": {sym: z}}} 또는 {name: z})."""
    out: list[float] = []
    for value in signals.values():
        if isinstance(value, dict):
            out.extend(v for v in (value.get("scores") or {}).values() if isinstance(v, (int, float)))
        elif isinstance(value, (int, float)):
            out.append(value)
    return out


def debate_trigger(
    proposed: dict[str, float],
    prev_weights: dict[str, float] | None,
    signals: dict,
    forced: bool = False,
) -> str | None:
    """소집 사유 반환, 조건 밖이면 None (verify 의 원천)."""
    if forced:
        return "user_request"
    scores = _signal_scores(signals)
    has_pos = any(v >= SIGNAL_CONFLICT_MIN for v in scores)
    has_neg = any(v <= -SIGNAL_CONFLICT_MIN for v in scores)
    if has_pos and has_neg:
        return "signal_conflict"
    if prev_weights is not None:
        l1 = sum(
            abs(proposed.get(k, 0.0) - prev_weights.get(k, 0.0))
            for k in set(proposed) | set(prev_weights)
        )
        if l1 >= LARGE_CHANGE_L1:
            return "large_position_change"
    return None


def _parse_side(text: str) -> dict:
    """토론자 출력 파싱 — 실패해도 토론은 자문일 뿐이라 원문으로 폴백(비치명)."""
    data = extract_json(text)
    if isinstance(data, dict):
        return {"thesis": str(data.get("thesis", ""))[:300],
                "points": [str(p)[:200] for p in data.get("points") or []][:3]}
    return {"thesis": text.strip()[:300], "points": []}


async def run_debate(
    router,
    market: str,
    user_payload: str,
    proposed: dict[str, float],
    trigger: str,
) -> tuple[dict, dict]:
    """Bull/Bear 병렬 소집. (구조화 결론, 토큰 사용량) 반환."""
    content = (
        user_payload
        + "\n\nproposed_allocation: "
        + json.dumps({k: round(v, 4) for k, v in proposed.items()}, ensure_ascii=False)
    )
    bull_resp, bear_resp = await asyncio.gather(
        router.complete(
            "fast", purpose="debate", market=market, system=BULL_SYSTEM.format(market=market),
            messages=[{"role": "user", "content": content}], max_tokens=1024,
        ),
        router.complete(
            "fast", purpose="debate", market=market, system=BEAR_SYSTEM.format(market=market),
            messages=[{"role": "user", "content": content}], max_tokens=1024,
        ),
    )
    tokens = {
        "in": (bull_resp.input_tokens or 0) + (bear_resp.input_tokens or 0),
        "out": (bull_resp.output_tokens or 0) + (bear_resp.output_tokens or 0),
    }
    conclusion = {
        "trigger": trigger,
        "proposed_before": {k: round(v, 4) for k, v in proposed.items()},
        "bull": _parse_side(bull_resp.text),
        "bear": _parse_side(bear_resp.text),
    }
    return conclusion, tokens
