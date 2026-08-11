"""RiskGuardedPolicy — 정책(LLM 포함)을 감싸 Risk Engine 통과를 강제.

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


def account_fingerprint(adapter: object) -> str:
    """어댑터(계좌) 지문. 리스크 상태 파일은 시장 단위라, 같은 시장을 다른 어댑터/계좌로
    바꾸면(예: 페이퍼→실계좌) 이전 계좌의 equity 이력·직전배분을 물려받아 MDD 서킷이
    오발동한다. 이 지문이 바뀌면 상태를 리셋한다(전환 감지 키)."""
    return ":".join(
        [type(adapter).__name__, getattr(adapter, "mode", "") or "", getattr(adapter, "cano", "") or ""]
    )


class RiskGuardedPolicy:
    def __init__(
        self,
        inner: Policy,
        engine: RiskEngine,
        state_path: Path | str,
        equity_fn: Callable[[], Awaitable[float]] | None = None,
        forbidden: frozenset[str] = frozenset(),
        forbidden_patterns_fn: Callable[[], set[str]] | None = None,
        account_key: str | None = None,
    ) -> None:
        self.inner = inner
        self.engine = engine
        self.state_path = Path(state_path)
        self.equity_fn = equity_fn
        self.forbidden = forbidden
        # admission 통과(active) Forbidden 패턴 집합 — 당일 결정의 pattern_key 가
        # 여기 걸리면 직전 배분으로 동결 (APV 하드 veto)
        self.forbidden_patterns_fn = forbidden_patterns_fn
        # 계좌 지문 — 어댑터/계좌 전환 시 stale equity 이력 리셋 트리거 (account_fingerprint)
        self.account_key = account_key
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
        # trigger 는 실시간 이벤트 소집 시에만 전달 — 일간 경로는 kwarg 없이
        # 호출해 기존 동작을 그대로 유지(baseline 정책 래핑 호환).
        raw = await (
            self.inner.decide(obs, positions, trigger=trigger)
            if trigger
            else self.inner.decide(obs, positions)
        )
        state = self._load_state()

        # 어댑터/계좌 전환 감지 — 리스크 상태는 시장 단위 파일이라 어댑터가 바뀌면(페이퍼→
        # 실계좌 등) 이전 계좌의 equity 이력을 물려받아 MDD 서킷이 오발동한다. 지문이
        # 명시적으로 다를 때만 리셋(레거시=지문 없음 은 현재 지문 채택, 오리셋 방지).
        existing_key = state.get("account_key")
        if self.account_key and existing_key is not None and existing_key != self.account_key:
            state = {}

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
                    # 저널이 이 키로 미집행 원안을 반사실 기록에 남긴다 — 동결된 배분만
                    # 남기면 veto 된 패턴의 표본이 끊겨 재검증이 불가능해지기 때문.
                    "counterfactual_key": key,
                    "risk_violations": [f"forbidden_pattern key={key}"],
                    "circuit_open": False,
                    "equity": None,
                    "equity_error": None,  # 브로커 조회 전 동결 — venue 장애와 무관
                    "mdd": 0.0,
                }
                self._save_state({**state, "prev_weights": frozen, "account_key": self.account_key})
                return frozen

        # 평가액은 브로커 조회다 — 실패해도 결정 자체는 공개 시세만으로 성립하므로
        # 예외로 스텝을 죽이지 않고 사유만 남긴다. 호출부는 이 사유를 보고 실주문을
        # 건너뛴다(MDD 검증이 불가한 상태로는 집행하지 않는다).
        equity, equity_error = None, None
        if self.equity_fn:
            try:
                equity = await self.equity_fn()
            except Exception as e:
                equity_error = f"{type(e).__name__}: {str(e)[:200]}"
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
            "equity_error": equity_error,  # None 아니면 브로커 조회 실패 — 실주문 스킵 사유
            "mdd": round(mdd, 4),
        }
        self._save_state(
            {
                "prev_weights": decision.weights,
                "peak_equity": peak if peak is not None else equity,
                "account_key": self.account_key,
            }
        )
        return decision.weights
