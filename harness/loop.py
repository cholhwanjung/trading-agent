"""일일 페이퍼 루프 — 무인 운용의 심장. 에이전트 없이도 무인으로 돈다.

한 스텝 = observe(감사됨) → decide → validate(∑=1) → submit → 구조화 로그.
verifier가 에이전트보다 먼저라는 원칙의 실행 지점: 이 루프가 3개 시장에서
무인으로 돌면 기준선이 완성되고, 이후 Policy 자리에 Trader가 꽂힌다.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from adapters.base import MarketAdapter, Observation, OrderResult, bar_observation_window
from harness.jsonlog import JsonlLogger
from harness.policy import Policy

# 부동소수 합산 오차 허용치
_SUM_TOL = 1e-6


class AllocationError(ValueError):
    """배분비율 벡터가 계약(∑=1, long-only)을 위반했을 때."""


def validate_weights(weights: dict[str, float]) -> None:
    """∑=1 ± 오차, 모든 값 ≥ 0 검증. 위반 시 AllocationError."""

    if not weights:
        raise AllocationError("empty weights")
    negatives = {s: w for s, w in weights.items() if w < 0}
    if negatives:
        raise AllocationError(f"negative weights (long-only): {negatives}")
    total = sum(weights.values())
    if abs(total - 1.0) > _SUM_TOL:
        raise AllocationError(f"weights sum={total!r}, expected 1.0")


def _observation_snapshot(obs: Observation) -> dict:
    """Observation → 감사 가능한 직렬화 dict. 에이전트가 '그때 본' 원시 관측 그대로.

    feature·근거는 결정 로그에 이미 있으므로 여기 넣지 않는다(한 파일=한 개념).
    """

    start, end = bar_observation_window(obs.asof_day)
    return {
        "market": obs.market,
        "asof_day": obs.asof_day.isoformat(),
        "collected_at": obs.collected_at.isoformat(),
        "window": [start.isoformat(), end.isoformat()],
        "bars": {
            symbol: [
                {"day": b.day.isoformat(), "open": b.open, "high": b.high,
                 "low": b.low, "close": b.close, "volume": b.volume}
                for b in bars
            ]
            for symbol, bars in obs.bars.items()
        },
        "news": [
            {"published_at": n.published_at.isoformat(), "headline": n.headline,
             "source": n.source, "url": n.url}
            for n in obs.news
        ],
    }


def write_observation_snapshot(snapshot_dir: Path, obs: Observation) -> Path:
    """관측 스냅샷을 {snapshot_dir}/{market}/{asof_day}.json 로 영속화(감사·시각화용).

    순수 append 채널 — 결정·리스크 로직과 무관. 같은 날 재실행은 덮어쓴다(최신 관측).
    """

    market_dir = snapshot_dir / obs.market
    market_dir.mkdir(parents=True, exist_ok=True)
    path = market_dir / f"{obs.asof_day.isoformat()}.json"
    path.write_text(
        json.dumps(_observation_snapshot(obs), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


async def run_daily_step(
    adapter: MarketAdapter,
    policy: Policy,
    symbols: list[str],
    logger: JsonlLogger,
    asof_day: date | None = None,
    snapshot_dir: Path | None = None,
) -> OrderResult:
    """시장 1개의 하루치 스텝. 관측은 누출 감사를 통과해야만 정책에 전달된다.

    snapshot_dir 지정 시 감사 직후 관측 스냅샷을 기록한다(결정 실패해도 관측은 보존).

    **브로커 장애 격리**: 관측·결정은 공개 시세만으로 성립하는데, 계좌 조회(보유·평가액)는
    브로커에 의존한다. 브로커가 죽었다고 스텝 전체를 실패시키면 그날의 결정·가상 병행
    운용 기록까지 사라져 측정 시계열에 구멍이 남는다(승격 판정을 그만큼 늦춘다). 그래서
    계좌 조회 실패는 스텝을 중단시키지 않고 결정까지 진행한 뒤 **실주문만 건너뛴다**
    (`execution_mode=degraded`). 평가액을 못 읽으면 MDD 검증이 불가하므로, 검증 없는
    집행을 하지 않는다는 뜻이기도 하다. degraded 여부는 로그에 남아 사후에 '실주문까지
    검증된 날'과 구분할 수 있다 — 라벨 없이 섞이면 측정 하네스가 자기를 속이게 된다.
    """

    obs = await adapter.observe_and_audit(symbols, asof_day)
    if snapshot_dir is not None:
        write_observation_snapshot(snapshot_dir, obs)

    venue_error: str | None = None
    try:
        positions = await adapter.get_positions()
    except Exception as e:
        positions, venue_error = [], f"{type(e).__name__}: {str(e)[:200]}"

    weights = await policy.decide(obs, positions)
    validate_weights(weights)

    # 평가액 조달 실패(정책 내부의 리스크 게이트 입력)도 같은 degraded 사유로 취급
    meta = getattr(policy, "last_decision", None) or {}
    venue_error = venue_error or meta.get("equity_error")

    if venue_error:
        result = OrderResult(
            market=adapter.market,
            submitted_at=datetime.now(timezone.utc),
            accepted=False,
            error=f"execution_skipped venue_unavailable — {venue_error}",
        )
    else:
        result = await adapter.submit_allocation(weights)

    logger.log(
        adapter.market,
        "daily_step",
        {
            "policy": policy.name,
            # LLM 정책의 결정 메타(근거·인용 ID·시나리오) — 로그 감사용, 없으면 None
            "decision": getattr(policy, "last_decision", None),
            "execution_mode": "degraded" if venue_error else "live",
            "asof_day": obs.asof_day,
            "collected_at": obs.collected_at,
            "n_bars": {s: len(b) for s, b in obs.bars.items()},
            "n_news": len(obs.news),
            "positions": [p.symbol for p in positions],
            "weights": weights,
            "accepted": result.accepted,
            "orders": result.orders,
            "error": result.error,
        },
    )
    return result
