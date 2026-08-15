"""RiskGuardedPolicy — 정책(LLM 포함)을 감싸 Risk Engine 통과를 강제.

LLM 은 이 레이어의 존재를 모른다 — 정책 출력이 무엇이든 enforce 후의 배분만 어댑터로
간다. 직전 목표 배분·평가액 고점은 시장별 상태 파일(JSON)에 영속 — turnover·MDD 입력.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path

from adapters.allocation import CASH
from adapters.base import Observation, Position
from adapters.retry import with_retry
from harness.policy import Policy
from risk.engine import RiskEngine

# 평가액 조회 재시도 예산. 어댑터가 자기 호출에 두는 예산보다 길게 잡는다 — 이 한 번의
# 실패는 값 하나가 비는 데서 끝나지 않고 **그날 집행 전체를 스킵**시키기 때문이다
# (아래 equity_error 경로). 장 개시 직후 브로커가 수십 초 막히는 일이 실제로 있었고
# (2026-08-13 10:01 KIS 잔고 조회 HTTP 500, 어댑터의 3초 예산으로는 부족했다), 이
# 시점에는 관측·결정이 이미 끝나 있어 기다리는 비용이 낮다. 대기 5+10+20 = 35초.
EQUITY_ATTEMPTS = 4
EQUITY_BASE_DELAY = 5.0

# 투자된 부분이 하루에 움직일 수 있는 최대 등락률. 국내 제한폭이 ±30% 라 분산된 보유가
# 하루에 그보다 크게 움직이기는 어렵다.
MAX_DAILY_MOVE = 0.35
# 손익으로 설명되지 않아도 입출금으로 보지 않는 하한(평가액 대비). 수수료·거래세·평가
# 반올림이 여기 들어간다 — 전액 현금이면 설명 가능한 변동이 0 이 되어, 이 하한이 없으면
# 몇 백 원짜리 잔돈이 매번 입출금으로 잡힌다.
MIN_FLOW_RATIO = 0.005


def detect_cash_flow(
    prev_equity: float | None,
    equity: float | None,
    invested: float = 1.0,
    days: int = 1,
) -> float:
    """평가액 변화 중 시장 변동으로 설명되지 않는 부분 → 외부 입출금 추정(없으면 0).

    상한을 평가액 전체가 아니라 **투자된 몫**에만 건다. 현금은 시장으로 움직이지 않으므로,
    평가액 전체에 상한을 걸면(≈35%) 서킷 임계(15%)보다 큰 상한이 되어 정작 서킷을 오발동
    시킬 크기의 출금이 전부 상한 안에 숨는다. invested 는 직전 목표 배분의 비현금 비중이라
    통화 단위가 없다 — 시장마다 보유 평가액과 평가액의 통화가 다를 수 있어(해외는 보유가
    USD, 평가액은 원화 총자산) 금액으로 비교하면 단위가 섞인다.

    목표 배분이라 실제 체결보다 클 수 있는데, 그쪽이 상한을 넓혀 놓치는 방향이라 안전하다.
    오차의 두 방향은 위험도가 다르다 — 입출금을 손익으로 놓치면 고점이 보정되지 않아 서킷이
    보수적으로(일찍) 작동할 뿐이지만, 반대로 손익을 입출금으로 읽으면 진짜 낙폭이 고점
    보정에 지워져 서킷이 침묵한다.

    days 는 직전 스텝과의 간격이다. 주말·휴장으로 며칠 벌어지면 시장이 움직일 여지도 그만큼
    커지므로, 하루치 상한을 그대로 쓰면 연휴 뒤의 큰 등락이 입출금으로 오인된다.
    """
    if prev_equity is None or equity is None:
        return 0.0
    delta = equity - prev_equity
    move = max(invested, 0.0) * MAX_DAILY_MOVE * max(days, 1)
    explainable = abs(prev_equity) * max(move, MIN_FLOW_RATIO)
    return delta if abs(delta) > explainable else 0.0


def adjust_peak(peak: float | None, prev_equity: float | None, equity: float) -> float:
    """외부 입출금이 있었던 날의 평가액 고점 재기준. 낙폭 **비율**을 보존한다.

    고점에 입금액을 더하는 방식(가산)은 입금이 기존 낙폭을 치유해버린다 — 1000만이 800만이
    된 상태(낙폭 20%)에서 1000만을 넣으면 낙폭이 11% 로 줄어 서킷이 침묵한다. 같은 배율로
    고점을 옮기면 낙폭 비율이 그대로 남아, 외부 자금 이동에 불변인 MDD 의 정의와 맞는다.

    직전 평가액이 0 이면 배율이 정의되지 않는다 — 고점 이력이 없는 첫 입금이므로 입금액을
    그대로 새 고점으로 삼는다.
    """
    if peak and prev_equity and prev_equity > 0:
        return peak * equity / prev_equity
    return equity


def _elapsed_days(prev_day: str | None, asof_day: date | None) -> int:
    """직전 스텝과의 간격(일). 기록이 없거나 읽을 수 없으면 1 일로 본다(가장 좁은 상한)."""
    try:
        return max((asof_day - date.fromisoformat(prev_day)).days, 1)
    except (TypeError, ValueError):
        return 1


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
                equity = await with_retry(
                    self.equity_fn,
                    attempts=EQUITY_ATTEMPTS,
                    base_delay=EQUITY_BASE_DELAY,
                )
            except Exception as e:
                equity_error = f"{type(e).__name__}: {str(e)[:200]}"
        # 외부 입출금은 손익이 아니다 — 고점을 그대로 두면 출금이 곧 낙폭으로 계상돼
        # 손실 없이도 서킷이 터지고, 입금은 고점을 부풀려 이후 진짜 하락에 서킷이 일찍
        # 터진다. 감지되면 고점을 재기준하고 그 사실을 결정 메타에 남긴다(사후 감사).
        peak = state.get("peak_equity")
        prev_equity = state.get("prev_equity")
        asof_day = getattr(obs, "asof_day", None)
        prev_cash = (state.get("prev_weights") or {}).get(CASH)
        mdd, cash_flow = 0.0, 0.0
        if equity is not None:
            cash_flow = detect_cash_flow(
                prev_equity,
                equity,
                1.0 - prev_cash if prev_cash is not None else 1.0,
                _elapsed_days(state.get("prev_day"), asof_day),
            )
            if cash_flow:
                peak = adjust_peak(peak, prev_equity, equity)
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
            "cash_flow": round(cash_flow, 2) or None,  # 감지된 외부 입출금(없으면 None)
        }
        self._save_state(
            {
                "prev_weights": decision.weights,
                "peak_equity": peak if peak is not None else equity,
                # 다음 스텝의 입출금 감지 기준. 평가액을 못 읽은 날은 직전 값을 유지해야
                # 한다 — None 으로 덮으면 브로커 장애 하루가 감지 기준선을 지워버린다.
                "prev_equity": equity if equity is not None else prev_equity,
                "prev_day": str(asof_day) if asof_day else state.get("prev_day"),
                "account_key": self.account_key,
            }
        )
        return decision.weights
