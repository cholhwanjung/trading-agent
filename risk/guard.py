"""RiskGuardedPolicy — 정책(LLM 포함)을 감싸 Risk Engine 통과를 강제 (R14, 하드룰 5).

LLM 은 이 레이어의 존재를 모른다 — 정책 출력이 무엇이든 enforce 후의 배분만 어댑터로
간다. 직전 목표 배분·평가액 고점은 시장별 상태 파일(JSON)에 영속 — turnover·MDD 입력.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from adapters.base import Observation, Position
from harness.policy import Policy
from risk.engine import RiskEngine


class RiskGuardedPolicy:
    def __init__(
        self,
        inner: Policy,
        engine: RiskEngine,
        state_path: Path | str,
        equity_fn: Callable[[], Awaitable[float]] | None = None,
        forbidden: frozenset[str] = frozenset(),
        forbidden_patterns_fn: Callable[[], set[str]] | None = None,
    ) -> None:
        self.inner = inner
        self.engine = engine
        self.state_path = Path(state_path)
        self.equity_fn = equity_fn
        self.forbidden = forbidden
        # admission 통과(active) Forbidden 패턴 집합 — 당일 결정의 pattern_key 가
        # 여기 걸리면 직전 배분으로 동결 (ADR-003 APV 하드 veto)
        self.forbidden_patterns_fn = forbidden_patterns_fn
        self.name = f"risk_guarded({inner.name})"
        self.last_decision: dict | None = None

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    async def decide(
        self, obs: Observation, positions: list[Position], trigger: dict | None = None
    ) -> dict[str, float]:
        # trigger 는 실시간 이벤트 소집([ADR-021]) 시에만 전달 — 일간 경로는 kwarg 없이
        # 호출해 기존 동작을 그대로 유지(baseline 정책 래핑 호환).
        raw = await (
            self.inner.decide(obs, positions, trigger=trigger)
            if trigger
            else self.inner.decide(obs, positions)
        )
        state = self._load_state()

        # Forbidden 패턴 하드 veto — 결정의 패턴이 검증된 실패 패턴이면 직전 배분 동결
        if self.forbidden_patterns_fn:
            from memory.journal import pattern_key as _pattern_key

            inner_meta = getattr(self.inner, "last_decision", None) or {}
            key = _pattern_key(
                inner_meta.get("features", {}), raw, state.get("prev_weights")
            )
            if key in self.forbidden_patterns_fn():
                frozen = state.get("prev_weights") or {"CASH": 1.0}
                self.last_decision = {
                    **inner_meta,
                    "weights_pre_risk": raw,
                    "risk_violations": [f"forbidden_pattern key={key}"],
                    "circuit_open": False,
                    "equity": None,
                    "mdd": 0.0,
                }
                self._save_state({**state, "prev_weights": frozen})
                return frozen

        equity = await self.equity_fn() if self.equity_fn else None
        peak = state.get("peak_equity")
        mdd = 0.0
        if equity is not None:
            peak = max(peak or equity, equity)
            mdd = 1.0 - equity / peak if peak > 0 else 0.0

        decision = self.engine.enforce(
            raw,
            prev_weights=state.get("prev_weights"),
            mdd=mdd,
            forbidden=self.forbidden,
        )

        inner_meta = getattr(self.inner, "last_decision", None) or {}
        self.last_decision = {
            **inner_meta,
            "weights_pre_risk": raw,
            "risk_violations": decision.violations,
            "circuit_open": decision.circuit_open,
            "equity": equity,
            "mdd": round(mdd, 4),
        }
        self._save_state(
            {"prev_weights": decision.weights, "peak_equity": peak if peak is not None else equity}
        )
        return decision.weights
