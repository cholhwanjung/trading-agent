"""국면 신호 채점 — 룰 기반 FSM 과 jump model 을 같은 잣대로 비교한다.

**왜 조건부 수익률인가**: 두 신호의 진짜 우열은 "전환을 제때 했는가"인데, 지속성이
높은 신호일수록 전환이 드물어(연 1회 미만) 라이브로는 수년이 걸린다. 반면 "그 상태로
서 있던 날의 익일 수익률"은 매일 한 관측씩 쌓인다. 그래서 risk_on/risk_off 두 무리의
평균 차이(spread)를 주지표로 삼고, 전환 빈도는 회전율 대리로 함께 본다.

**측정 대상은 지수 프록시가 아니라 B&H 가상 arm 수익률이다.** 국면 로그에 프록시 종가가
없어 사후 재구성이 불가하고, 국면 신호가 이 시스템에서 할 일이 '우리가 든 것'을 게이팅
하는 것이라 그쪽이 판정에 더 가깝다. 두 모델을 같은 대상으로 채점하므로 어느 쪽에도
유리하지 않다 — 대상이 바뀌어도 비교의 공정성은 유지된다.

**정렬**: 상태는 t−1 봉까지 보고 만들어지므로 관측일 D 의 상태에 대응하는 미실현 봉은
D 일 봉이다. 수익일마다 그 이전(≤) 최신 상태를 붙여 **수익일 1개당 관측 1개**만 쓴다 —
주말·휴장·재실행으로 상태가 여러 번 찍혀도 중복 계상되지 않고, 표본이 겹치지 않는다.

판정 기준은 데이터가 쌓이기 전에 고정했다: JM spread > FSM spread **이면서** 전환 횟수가
FSM 이하일 때만 승격 후보. 사후에 지표를 덧붙이지 않는다.
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from eval.meta import load_arm_history
from harness.jsonlog import iter_events
from regime.pulse import UPTREND

FSM_EVENT, JM_EVENT = "regime", "jm_regime"
FSM_RISK_ON = frozenset({UPTREND})  # UNDER_PRESSURE·CORRECTION 은 전부 risk_off
JM_RISK_ON = frozenset({"bull"})
MARKET_ARM = "bh"  # 시장 대리 — 결정에 영향받지 않는 고정 균등 arm
MIN_DAYS = 21  # 판정 최소 표본 (다른 승격 지표와 동일 하한)
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class RegimeScore:
    model: str
    n_days: int  # 상태·수익률이 모두 있는 거래일
    n_risk_on: int
    n_risk_off: int
    mean_on_pct: float  # risk_on 일의 익일 수익률 평균 (%)
    mean_off_pct: float
    spread_pct: float  # mean_on − mean_off — 주지표
    switches: int  # 거래일 기준 상태 전환 횟수
    switches_per_year: float
    insufficient: bool  # True 면 값은 참고용, 판정에 쓰지 않는다


def load_states(
    log_root: Path | str, market: str, event: str, risk_on: frozenset[str]
) -> dict[date, bool]:
    """국면 로그 → {관측일: risk_on}. 같은 날 여러 건이면 마지막 것만 남긴다.

    관측일은 레코드의 ts 날짜 — 국면은 실행 시각의 UTC 날짜를 기준일로 계산되므로
    둘이 같다. 재실행·중복 기록이 있어도 하루 1개로 접힌다.
    """
    out: dict[date, bool] = {}
    for record in iter_events(log_root, market, event):
        state, ts = record.get("state"), record.get("ts")
        if not state or not ts:
            continue
        out[date.fromisoformat(ts[:10])] = state in risk_on
    return out


def forward_returns(
    virtual_dir: Path | str, market: str, arm: str = MARKET_ARM
) -> dict[date, float]:
    """{수익일: 그날 봉의 수익률} — 가상 arm equity 에서 파생.

    equity 이력의 day 는 마킹에 쓴 t−1 종가일이라, 연속 두 항목의 비율이 나중 날짜
    봉의 수익률이 된다. virtual_dir 은 가상 상태 디렉토리(data/state/virtual).
    """
    history = load_arm_history(Path(virtual_dir), market, arm)
    out: dict[date, float] = {}
    for prev, cur in zip(history, history[1:]):
        if prev.get("equity", 0) > 0 and cur.get("equity") is not None:
            out[date.fromisoformat(cur["day"])] = cur["equity"] / prev["equity"] - 1
    return out


def score(model: str, states: dict[date, bool], fwd: dict[date, float]) -> RegimeScore | None:
    """수익일마다 직전 최신 상태를 붙여 조건부 평균·spread 계산. 겹칠 것이 없으면 None."""
    if not states or not fwd:
        return None

    state_days = sorted(states)
    pairs: list[tuple[bool, float]] = []
    for day in sorted(fwd):
        idx = bisect.bisect_right(state_days, day) - 1
        if idx >= 0:  # 그 봉 이전에 서 있던 상태가 없으면 채점 불가
            pairs.append((states[state_days[idx]], fwd[day]))
    if not pairs:
        return None

    on = [r for is_on, r in pairs if is_on]
    off = [r for is_on, r in pairs if not is_on]
    mean_on = sum(on) / len(on) if on else 0.0
    mean_off = sum(off) / len(off) if off else 0.0
    switches = sum(1 for a, b in zip(pairs, pairs[1:]) if a[0] != b[0])

    return RegimeScore(
        model=model,
        n_days=len(pairs),
        n_risk_on=len(on),
        n_risk_off=len(off),
        mean_on_pct=round(mean_on * 100, 4),
        mean_off_pct=round(mean_off * 100, 4),
        spread_pct=round((mean_on - mean_off) * 100, 4),
        switches=switches,
        switches_per_year=round(switches / len(pairs) * TRADING_DAYS_PER_YEAR, 2),
        # 한쪽 무리가 비면 spread 가 상대 평균이 아니라 0 과의 차이가 된다 — 판정 제외
        insufficient=len(pairs) < MIN_DAYS or not on or not off,
    )


def compare_regimes(
    log_root: Path | str, virtual_dir: Path | str, market: str, arm: str = MARKET_ARM
) -> dict:
    """시장 1곳의 FSM vs JM 채점 + 사전 확정 기준에 따른 판정.

    verdict: no_jm(모델 미배선) · insufficient(표본 부족) · jm_promotable · hold
    """
    fwd = forward_returns(virtual_dir, market, arm)
    fsm = score("fsm", load_states(log_root, market, FSM_EVENT, FSM_RISK_ON), fwd)
    jm = score("jm", load_states(log_root, market, JM_EVENT, JM_RISK_ON), fwd)

    reasons: list[str] = []
    if fsm is None:
        verdict = "insufficient"
        reasons.append("fsm 상태 또는 수익률 없음")
    elif jm is None:
        verdict = "no_jm"
        reasons.append(f"{JM_EVENT} 로그 없음 — 모델 미배선")
    elif fsm.insufficient or jm.insufficient:
        verdict = "insufficient"
        reasons.append(f"표본 부족 (fsm n={fsm.n_days} · jm n={jm.n_days} · 하한 {MIN_DAYS})")
    else:
        wins_spread = jm.spread_pct > fsm.spread_pct
        holds_turnover = jm.switches <= fsm.switches
        verdict = "jm_promotable" if wins_spread and holds_turnover else "hold"
        reasons.append(
            f"spread jm={jm.spread_pct} vs fsm={fsm.spread_pct} → {'통과' if wins_spread else '미달'}"
        )
        reasons.append(
            f"전환 jm={jm.switches} vs fsm={fsm.switches} → {'통과' if holds_turnover else '미달'}"
        )

    return {
        "market": market,
        "arm": arm,
        "n_return_days": len(fwd),
        "fsm": asdict(fsm) if fsm else None,
        "jm": asdict(jm) if jm else None,
        "verdict": verdict,
        "reasons": reasons,
    }
