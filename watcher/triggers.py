"""실시간 이벤트 트리거 (단계 1) — 순수 판정 로직.

가격 급변만 감지한다: 직전 참조가(reference) 대비 |move| 가 임계 이상이면 트리거.
참조가는 롤링 윈도우 — 트리거 발동 또는 ref TTL 경과 시 현재가로 갱신되어
"최근 N시간 내 급변"을 의미한다. 뉴스·regime 트리거는 v2.

상주 데몬이 아니라 주기 check-once 로 호출된다(scripts/run_watcher.py). 상태는
JSON 파일에 영속 — 이 모듈은 (state, prices, now) → (trigger?, new_state) 순수 함수.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class TriggerConfig:
    """시장별 트리거 파라미터. 개별 종목 변동성에 맞춰 조정 (리스크 한도 정신)."""

    move_threshold: float  # |Δ| ≥ 이 값이면 발동 (예: 0.08 = 8%)
    cooldown_s: int  # 직전 발동 후 이 시간 내 재발동 금지
    ref_ttl_s: int  # 무발동 시 참조가를 현재가로 갱신하는 주기(롤링 윈도우 폭)


# v1 은 CRYPTO 전용(24/7, 장중 게이팅 불필요). US/KR 은 장 시간 게이팅 도입 후 추가.
DEFAULTS = {
    "CRYPTO": TriggerConfig(move_threshold=0.08, cooldown_s=3600, ref_ttl_s=10800),
}


def config_for(market: str) -> TriggerConfig:
    if market not in DEFAULTS:
        raise KeyError(f"트리거 미지원 시장={market} (v1 은 {sorted(DEFAULTS)})")
    return DEFAULTS[market]


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def _moves(current: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    """공통 심볼의 부호 있는 변동률. ref 가격이 0/누락이면 제외."""
    return {s: px / ref[s] - 1.0 for s, px in current.items() if ref.get(s)}


def max_drift(current: dict[str, float], ref: dict[str, float]) -> float:
    """참조가 대비 최대 절대 변동률 — 무발동 시 관측성 로깅용."""
    moves = _moves(current, ref)
    return max((abs(m) for m in moves.values()), default=0.0)


def evaluate(
    state: dict, current: dict[str, float], now: datetime, config: TriggerConfig
) -> tuple[dict | None, dict]:
    """(트리거 payload | None, 새 state) 반환. 순수 함수 — I/O 없음.

    state: {"ref": {sym: px}, "ref_at": iso, "last_fire_at": iso|null}
    트리거 payload 는 부호 있는 moves 를 담아 LLM 이 방향(급락/급등)을 알 수 있게 한다.
    """
    now = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
    ref = dict(state.get("ref") or {})
    ref_at = _parse(state.get("ref_at"))
    last_fire_iso = state.get("last_fire_at")

    # 최초 실행(참조가 없음) → 시드만 하고 발동 안 함
    if not ref:
        return None, {"ref": dict(current), "ref_at": now.isoformat(), "last_fire_at": last_fire_iso}

    moves = _moves(current, ref)
    worst = max(moves, key=lambda s: abs(moves[s]), default=None)
    drift = abs(moves[worst]) if worst else 0.0
    last_fire = _parse(last_fire_iso)
    in_cooldown = last_fire is not None and (now - last_fire).total_seconds() < config.cooldown_s

    if worst and drift >= config.move_threshold and not in_cooldown:
        trigger = {
            "reason": "price_move",
            "fired_at": now.isoformat(),
            "threshold": config.move_threshold,
            "worst_symbol": worst,
            "worst_move": round(moves[worst], 4),
            "moves": {s: round(m, 4) for s, m in moves.items()},
            "current_prices": dict(current),
            "ref_prices": ref,
        }
        # 발동 후 참조가 리셋 — 다음 이동은 여기서부터 측정(재발동 폭주 방지)
        fired = {"ref": dict(current), "ref_at": now.isoformat(), "last_fire_at": now.isoformat()}
        return trigger, fired

    # 무발동: ref 가 오래됐으면 전체 갱신(롤링 윈도우), 아니면 신규 심볼만 흡수
    if ref_at is None or (now - ref_at).total_seconds() >= config.ref_ttl_s:
        return None, {"ref": dict(current), "ref_at": now.isoformat(), "last_fire_at": last_fire_iso}
    new_syms = set(current) - set(ref)
    if new_syms:
        merged = {**ref, **{s: current[s] for s in new_syms}}
        return None, {"ref": merged, "ref_at": state.get("ref_at"), "last_fire_at": last_fire_iso}
    return None, state
