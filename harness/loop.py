"""일일 페이퍼 루프 — 무인 운용의 심장. 에이전트 없이도 무인으로 돈다.

한 스텝 = observe(감사됨) → decide → validate(∑=1) → submit → 구조화 로그.
verifier가 에이전트보다 먼저라는 원칙의 실행 지점: 이 루프가 3개 시장에서
무인으로 돌면 기준선이 완성되고, 이후 Policy 자리에 Trader가 꽂힌다.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from adapters.base import MarketAdapter, Observation, OrderResult, observation_window
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

    start, end = observation_window(obs.asof_day)
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
    """

    obs = await adapter.observe_and_audit(symbols, asof_day)
    if snapshot_dir is not None:
        write_observation_snapshot(snapshot_dir, obs)
    positions = await adapter.get_positions()
    weights = await policy.decide(obs, positions)
    validate_weights(weights)
    result = await adapter.submit_allocation(weights)
    logger.log(
        adapter.market,
        "daily_step",
        {
            "policy": policy.name,
            # LLM 정책의 결정 메타(근거·인용 ID·시나리오) — 로그 감사용, 없으면 None
            "decision": getattr(policy, "last_decision", None),
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
