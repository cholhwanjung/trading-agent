"""영향력 제어 — residual + confidence gating + bounded deviation.

메모리는 base 정책에 대한 *calibrated residual* 이다:
    final = base + scale · (mem − base),  scale = confidence (상한: max_deviation)

- base = 교훈 미주입 결정, mem = 교훈 주입 결정. 편차(mem−base)가 메모리의 기여분.
- confidence 는 admission 통계(관측수 n, 승률)에서 유도 — 증거가 얇으면 0 으로 수렴,
  메모리가 스스로를 증명하기 전엔 base 그대로.
- mem 결정이 교훈을 인용하지 않았으면 편차는 메모리 귀속이 아니다 → base 반환.
- Forbidden(실패) 패턴은 이 경로가 아니라 risk.guard 의 하드 veto (APV 비대칭).
"""

from __future__ import annotations

from dataclasses import dataclass

from adapters.allocation import CASH
from memory.store import MemoryEntry

MAX_DEVIATION = 0.20  # 메모리가 배분에 줄 수 있는 섭동 상한 (L1/2)
CONFIDENCE_N_HALF = 5  # n/(n+k) 반포화 상수 — n=5 에서 0.5


def lesson_confidence(entry: MemoryEntry) -> float:
    """admission 통계 → (0,1) 신뢰도. c = n/(n+5) × 승률. 증거 얇으면 0 수렴."""
    n = entry.data.get("n", 0)
    k_pos = entry.data.get("k_pos", 0)
    if n <= 0:
        return 0.0
    win_rate = k_pos / n if entry.data.get("kind") == "success" else (n - k_pos) / n
    return n / (n + CONFIDENCE_N_HALF) * win_rate


def lessons_payload(scored: list) -> list[dict]:
    """retrieval 결과 → 프롬프트 주입/블렌딩용 페이로드 (confidence 포함)."""
    return [
        {
            "id": s.entry.id,
            "content": s.entry.content,
            "confidence": round(lesson_confidence(s.entry), 4),
        }
        for s in scored
    ]


@dataclass(frozen=True)
class BlendResult:
    weights: dict[str, float]
    applied: bool  # False = base 그대로 (교훈 미인용 or 기여 0)
    confidence: float
    deviation_l1: float  # 스케일 전 |mem−base| 편차 (L1/2)
    scale: float  # 실제 적용 배율 (confidence × bound 클램프)


def blend_allocations(
    base: dict[str, float],
    mem: dict[str, float],
    cited_lessons: list[dict],
    max_deviation: float = MAX_DEVIATION,
) -> BlendResult:
    """base + confidence·bounded(mem − base). 반환 배분은 ∑=1, long-only 보장."""
    symbols = (set(base) | set(mem)) - {CASH}
    deviation = 0.5 * sum(abs(mem.get(s, 0.0) - base.get(s, 0.0)) for s in symbols | {CASH})

    if not cited_lessons or deviation == 0.0:
        return BlendResult(dict(base), False, 0.0, deviation, 0.0)

    confidence = sum(le.get("confidence", 0.0) for le in cited_lessons) / len(cited_lessons)
    scale = confidence
    if deviation * scale > max_deviation:
        scale = max_deviation / deviation
    if scale <= 0.0:
        return BlendResult(dict(base), False, confidence, deviation, 0.0)

    final: dict[str, float] = {}
    for s in symbols:
        blended = base.get(s, 0.0) + scale * (mem.get(s, 0.0) - base.get(s, 0.0))
        final[s] = max(0.0, blended)  # long-only
    final[CASH] = max(0.0, 1.0 - sum(final.values()))
    total = sum(final.values())
    if abs(total - 1.0) > 1e-9:  # long-only 클램프 잔차 정규화
        final = {s: w / total for s, w in final.items()}
    return BlendResult(final, True, confidence, deviation, scale)
