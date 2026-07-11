"""가상 포트폴리오 — 동일 관측·실가격 기반 병행 페이퍼 운용 (LiveTradeBench 방식).

백테스트가 아니다: 매일 라이브로 도착하는 t-1 종가에 목표 배분을 적용하는 forward
시뮬레이션(하드룰 2 위배 없음). 정책별 상태를 JSON 으로 영속해 equity 곡선을 누적 —
LLM vs B&H vs 랜덤의 델타 측정(Phase 1 완료 기준)이 목적.

거래비용: 리밸런싱 노셔널에 cost_bps 부과 (QuantaAlpha 식 민감도 리포트는 Phase 3+).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from adapters.allocation import CASH


class VirtualPortfolio:
    def __init__(self, state_path: Path | str, initial_cash: float = 100_000.0) -> None:
        self.state_path = Path(state_path)
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.cash: float = state["cash"]
            self.qty: dict[str, float] = state["qty"]
            self.history: list[dict] = state["history"]
        else:
            self.cash = initial_cash
            self.qty = {}
            self.history = []

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.qty.items() if q > 0)

    def step(
        self,
        day: date,
        prices: dict[str, float],
        weights: dict[str, float],
        cost_bps: float = 10.0,
    ) -> float:
        """목표 배분을 prices 로 리밸런싱하고 equity 기록. 같은 날 재실행은 멱등(스킵)."""

        day_key = day.isoformat()
        if self.history and self.history[-1]["day"] == day_key:
            return self.history[-1]["equity"]  # 이미 오늘 스텝 완료

        equity = self.equity(prices)
        traded = 0.0
        new_qty: dict[str, float] = {}
        for symbol, weight in weights.items():
            if symbol == CASH:
                continue
            target_value = weight * equity
            current_value = self.qty.get(symbol, 0.0) * prices[symbol]
            traded += abs(target_value - current_value)
            new_qty[symbol] = target_value / prices[symbol]
        # 배분에서 빠진 보유 자산은 전량 매도
        for symbol, q in self.qty.items():
            if symbol not in new_qty and q > 0:
                traded += q * prices[symbol]

        cost = traded * cost_bps / 1e4
        invested = sum(q * prices[s] for s, q in new_qty.items())
        self.qty = new_qty
        self.cash = equity - invested - cost

        after = self.equity(prices)
        self.history.append({"day": day_key, "equity": round(after, 2), "cost": round(cost, 4)})
        self._save()
        return after

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {"cash": self.cash, "qty": self.qty, "history": self.history},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
