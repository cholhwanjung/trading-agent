"""baseline 정책 — 에이전트 없이 페이퍼 루프를 돌리기 위한 기준선.

정책은 관측을 받아 배분비율 벡터(∑=1, 현금 포함, long-only)를 반환한다.
LLM 없음 — B&H와 랜덤은 이후 모든 에이전트 성과의 비교 기준선이다.
"""

from __future__ import annotations

import random
from typing import Protocol

from adapters.allocation import CASH
from adapters.base import Observation, Position


class Policy(Protocol):
    """배분 정책 계약. Trader 도 이 계약을 따른다 (LLM 정책 때문에 async)."""

    name: str

    async def decide(self, obs: Observation, positions: list[Position]) -> dict[str, float]:
        """현금 포함 배분비율 벡터 반환. ∑=1, 모든 값 ≥ 0 (long-only)."""
        ...


class BuyAndHold:
    """첫 결정에서 유니버스 균등 배분 후 유지. 시장 baseline."""

    name = "buy_and_hold"

    def __init__(self, universe: list[str], cash_weight: float = 0.0) -> None:
        self.universe = universe
        self.cash_weight = cash_weight
        self._initial: dict[str, float] | None = None

    async def decide(self, obs: Observation, positions: list[Position]) -> dict[str, float]:
        if self._initial is None:
            per_asset = (1.0 - self.cash_weight) / len(self.universe)
            self._initial = {sym: per_asset for sym in self.universe}
            self._initial[CASH] = self.cash_weight
        return dict(self._initial)


class RandomPolicy:
    """매 스텝 무작위 배분. 운/노이즈 분리용 하한 baseline."""

    name = "random"

    def __init__(self, universe: list[str], seed: int | None = None) -> None:
        self.universe = universe
        self._rng = random.Random(seed)

    async def decide(self, obs: Observation, positions: list[Position]) -> dict[str, float]:
        raw = {sym: self._rng.random() for sym in [*self.universe, CASH]}
        total = sum(raw.values())
        return {sym: w / total for sym, w in raw.items()}
