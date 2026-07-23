"""Treasury Guard — Risk Engine 의 자금이체판 ([ADR-026]).

시장 간(버킷 간) 자본 이체의 결정론 코어. 순수 함수 — I/O·클럭·네트워크 없음
(now 는 인자 주입, watcher/triggers 와 동일 규약) → 단위 테스트 가능.

두 단계:
    plan_transfers   — 메타 목표 split vs 현재 venue 잔고 → 이체 의도
                       (드리프트 게이트로 churn 회피 · 건당 상한 클램프)
    enforce_transfer — 집행 직전 최종 게이트: allowlist·건당/일일 상한·쿨다운·잔고 검증

allowlist 는 코드 상수(배선 시 하드코딩). LLM/관측/트리거가 목적지를 유입시키는 경로는
없다 — 인젝션·환각發 자금 유출을 구조적으로 차단([ADR-026] ①②).
모든 금액은 공통 기준통화(KRW) 절대액.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# 배선 시 사용자 본인 소유 venue 만 하드코딩. 절대 런타임/관측/LLM 에서 채우지 않는다.
ALLOWLIST: frozenset[str] = frozenset()

_EPS = 1e-9


@dataclass(frozen=True)
class TransferIntent:
    """이체 의도 1건. amount 는 공통 기준통화 절대액, reason 은 발동 근거(credit)."""

    from_venue: str
    to_venue: str
    amount: float
    reason: str


@dataclass(frozen=True)
class TreasuryLimits:
    """운영 캘리브레이션 대상. Risk Engine turnover 상한의 자금이체판."""

    per_transfer_cap: float          # 건당 상한
    daily_cap: float                 # 일일 누적 상한
    min_drift_to_fire: float = 0.10  # split 편차 ≥ 이 값일 때만 이체(무이체 = churn 회피)
    max_share_moved: float = 0.25    # 1회 이동이 총자본의 이 비율 초과 금지
    cooldown_h: float = 24.0         # 직전 이체 후 재이체 금지 시간


@dataclass(frozen=True)
class TreasuryDecision:
    allow: bool
    intent: TransferIntent
    violations: list[str] = field(default_factory=list)  # key=value 로그용


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def plan_transfers(
    target_split: dict[str, float],
    venue_equity: dict[str, float],
    limits: TreasuryLimits,
    reason: str = "meta_rebalance",
) -> list[TransferIntent]:
    """메타 목표 split 과 현재 venue 잔고 차이 → 이체 의도. 순수 함수.

    - 최대 split 편차 < min_drift_to_fire → 빈 리스트(무이체, churn 회피).
    - 잉여 venue(음의 delta) → 부족 venue(양의 delta) greedy 매칭.
    - 각 이체는 max_share_moved · 총자본 으로 클램프 — 초과분은 다음 사이클
      (실 잔고로 재계산되므로 자기보정).
    """
    total = sum(venue_equity.values())
    if total <= 0:
        return []
    current = {v: eq / total for v, eq in venue_equity.items()}
    drift = max((abs(target_split.get(v, 0.0) - current[v]) for v in venue_equity), default=0.0)
    if drift < limits.min_drift_to_fire:
        return []

    deltas = {v: target_split.get(v, 0.0) * total - eq for v, eq in venue_equity.items()}
    cap = limits.max_share_moved * total
    sources = sorted(((v, -d) for v, d in deltas.items() if d < -_EPS), key=lambda x: -x[1])
    dests = sorted(((v, d) for v, d in deltas.items() if d > _EPS), key=lambda x: -x[1])

    intents: list[TransferIntent] = []
    si = 0
    src_v, src_avail = sources[0] if sources else (None, 0.0)
    for dst_v, need in dests:
        while need > _EPS and src_v is not None:
            amount = min(need, src_avail, cap)
            intents.append(TransferIntent(src_v, dst_v, round(amount, 2), reason))
            need -= amount
            src_avail -= amount
            if src_avail <= _EPS:  # 이 소스 소진 → 다음 소스로 계속 채운다
                si += 1
                src_v, src_avail = sources[si] if si < len(sources) else (None, 0.0)
            elif amount >= cap - _EPS:  # 건당 cap 도달 → 이 dest 는 다음 사이클에 마저
                break
    return intents


def enforce_transfer(
    intent: TransferIntent,
    limits: TreasuryLimits,
    state: dict,
    live_balance: float,
    now: datetime,
    allowlist: frozenset[str] | None = None,
) -> TreasuryDecision:
    """집행 직전 최종 결정론 게이트 (RiskEngine.enforce 대칭). now 주입(무클럭).

    allowlist 위반은 **최우선 하드 거부**(다른 검사 이전) — 오목적지 유출 원천 차단.
    state: {"last_transfer_at": iso|None, "daily_moved": float} (일자 리셋은 호출부 책임).
    live_balance: from_venue 실 조회 잔고 — reconciliation(미정산 in-flight 이중이체 방지).
    """
    allow = ALLOWLIST if allowlist is None else allowlist
    if intent.from_venue not in allow or intent.to_venue not in allow:
        return TreasuryDecision(
            False, intent,
            [f"not_allowlisted from={intent.from_venue} to={intent.to_venue}"],
        )

    violations: list[str] = []
    if intent.amount > limits.per_transfer_cap:
        violations.append(
            f"per_transfer_cap amount={intent.amount} cap={limits.per_transfer_cap}"
        )
    daily_moved = state.get("daily_moved", 0.0)
    if daily_moved + intent.amount > limits.daily_cap:
        violations.append(
            f"daily_cap moved={daily_moved} amount={intent.amount} cap={limits.daily_cap}"
        )
    last = _parse(state.get("last_transfer_at"))
    if last is not None:
        elapsed_h = (now - last).total_seconds() / 3600
        if elapsed_h < limits.cooldown_h:
            violations.append(f"cooldown elapsed_h={elapsed_h:.3f} required={limits.cooldown_h}")
    if live_balance < intent.amount:
        violations.append(f"insufficient_balance live={live_balance} amount={intent.amount}")

    return TreasuryDecision(allow=not violations, intent=intent, violations=violations)
