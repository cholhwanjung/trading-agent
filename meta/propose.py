"""메타 배분 제안 — LLM 출력을 anchor ± max_tilt 로 clamp 한 결정론 결과.

제어권은 clamp 에 있다. 모델이 범위를 넘긴 값은 잘리고 그 사실이 `clamped` 로 남는다.
`confidence` 는 기록만 한다 — 자기보고 확신을 배율로 쓰려면 먼저 그 예측력이 증명돼야 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from llm import LLMRouter, extract_json
from trader.prompt_store import PromptSpec


class MetaProposalError(ValueError):
    """출력이 스키마를 위반 — 그 주는 제안 없음(anchor)으로 기록한다."""


@dataclass(frozen=True)
class MetaLLMProposal:
    asof_day: date
    weights_raw: dict[str, float]
    weights: dict[str, float]
    confidence: float
    cited: list[str]
    rationale: str
    horizon: str
    prompt_rev: str
    clamped: bool


def clamp_weights(
    raw: dict[str, float], anchor: dict[str, float], max_tilt: float
) -> tuple[dict[str, float], bool]:
    """시장별 [anchor−t, anchor+t] 로 자른 뒤, 합이 1 이 되도록 **밴드 안에서** 조정한다.

    단순 재정규화는 잘린 값을 다시 밴드 밖으로 밀어낸다. 대신 부족분·초과분을 아직 경계에
    닿지 않은 시장들에 균등 분배하고, 경계에 닿는 시장을 고정해 가며 반복한다(물 채우기).
    """
    lo = {m: max(0.0, anchor[m] - max_tilt) for m in anchor}
    hi = {m: anchor[m] + max_tilt for m in anchor}
    w = {m: min(max(float(raw.get(m, anchor[m])), lo[m]), hi[m]) for m in anchor}
    clamped = any(abs(w[m] - float(raw.get(m, anchor[m]))) > 1e-9 for m in anchor)
    for _ in range(len(anchor) + 1):
        excess = sum(w.values()) - 1.0
        if abs(excess) < 1e-12:
            break
        free = [
            m
            for m in anchor
            if (excess > 0 and w[m] > lo[m] + 1e-12) or (excess < 0 and w[m] < hi[m] - 1e-12)
        ]
        if not free:
            break
        share = excess / len(free)
        for m in free:
            w[m] = min(max(w[m] - share, lo[m]), hi[m])
    if abs(sum(w.values()) - 1.0) > 1e-9:  # 밴드 안에서 합 1 이 불가능한 anchor — 재정규화로 후퇴
        total = sum(w.values())
        w = {m: v / total for m, v in w.items()}
        clamped = True
    return {m: round(v, 6) for m, v in w.items()}, clamped


def parse_proposal(text: str, markets: tuple[str, ...]) -> dict:
    """LLM 텍스트 → 검증된 dict(weights·confidence·cited·rationale·horizon)."""
    data = extract_json(text)
    if not isinstance(data, dict):
        raise MetaProposalError(f"JSON 객체 없음: {text[:120]!r}")
    weights = data.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(markets):
        raise MetaProposalError(f"weights 는 {sorted(markets)} 전부를 가져야 한다: {weights}")
    try:
        weights = {m: float(weights[m]) for m in markets}
    except (TypeError, ValueError) as e:
        raise MetaProposalError(f"weights 숫자 아님: {e}") from e
    if any(v < 0 for v in weights.values()):
        raise MetaProposalError("음수 비중")
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    rationale = str(data.get("rationale", "")).strip()
    if not rationale:
        raise MetaProposalError("rationale 필수")
    return {
        "weights": weights,
        "confidence": confidence,
        "cited": [str(x) for x in (data.get("cited") or [])],
        "rationale": rationale,
        "horizon": str(data.get("horizon") or "4w"),
    }


def build_user_prompt(
    panel: dict,
    digest: dict[str, list[dict]],
    lessons: list[dict] | None = None,
    reliability: list[str] | None = None,
) -> str:
    """패널·서사·교훈·신뢰도 표를 하나의 JSON 으로 — 모델이 보는 것이 곧 감사가 보는 것."""
    payload = {
        "asof_day": panel["asof_day"],
        "anchor": panel["anchor"],
        "max_tilt": panel["max_tilt"],
        "markets": panel["markets"],
        "macro": panel.get("macro"),
        "news": {
            m: [
                {"ids": ln["ids"], "text": ln["text"], "direction": ln["direction"]} for ln in lines
            ]
            for m, lines in digest.items()
        },
        "prev_proposal": panel.get("prev_proposal"),
        "actual_split": panel.get("actual_split"),
        "lessons": lessons or [],
        "reliability": reliability or [],
        "valid_signal_ids": panel["signal_ids"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1, default=str)


async def propose(
    router: LLMRouter,
    spec: PromptSpec,
    panel: dict,
    digest: dict[str, list[dict]],
    lessons: list[dict] | None = None,
    reliability: list[str] | None = None,
    purpose: str = "meta_proposal",
) -> MetaLLMProposal:
    """제안 1건. 파싱 실패는 MetaProposalError 로 올린다(호출부가 anchor 로 기록)."""
    resp = await router.complete(
        spec.tier,
        purpose=purpose,
        system=spec.text,
        messages=[
            {"role": "user", "content": build_user_prompt(panel, digest, lessons, reliability)}
        ],
        json_mode=True,
    )
    markets = tuple(panel["anchor"])
    parsed = parse_proposal(resp.text, markets)
    weights, clamped = clamp_weights(parsed["weights"], panel["anchor"], panel["max_tilt"])
    return MetaLLMProposal(
        asof_day=date.fromisoformat(panel["asof_day"]),
        weights_raw={m: round(v, 6) for m, v in parsed["weights"].items()},
        weights=weights,
        confidence=round(parsed["confidence"], 4),
        cited=parsed["cited"],
        rationale=parsed["rationale"],
        horizon=parsed["horizon"],
        prompt_rev=spec.rev,
        clamped=clamped,
    )
