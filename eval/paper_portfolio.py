"""가상 포트폴리오 — 동일 관측·실가격 기반 병행 페이퍼 운용 (LiveTradeBench 방식).

백테스트가 아니다: 매일 라이브로 도착하는 t-1 종가에 목표 배분을 적용하는 forward
시뮬레이션(미래 참조 없음). 정책별 상태를 JSON 으로 영속해 equity 곡선을 누적 —
LLM vs B&H vs 랜덤의 델타 측정이 목적.

거래비용: 리밸런싱 노셔널에 cost_bps 부과 (QuantaAlpha 식 민감도 리포트는 향후).
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
            # 마지막 관측가 — 유니버스에서 빠진 보유를 청산할 때 mark 로 사용 (구버전은 {})
            self.last_prices: dict[str, float] = state.get("last_prices", {})
        else:
            self.cash = initial_cash
            self.qty = {}
            self.history = []
            self.last_prices = {}

    def _mark(self, symbol: str, prices: dict[str, float]) -> float:
        """현재가 우선, 없으면(유니버스 이탈) 마지막 관측가, 그것도 없으면 0.0."""
        price = prices.get(symbol)
        return price if price is not None else self.last_prices.get(symbol, 0.0)

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * self._mark(s, prices) for s, q in self.qty.items() if q > 0)

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

        # 관측된 현재가를 누적 — 이후 유니버스에서 빠져도 마지막 관측가로 청산 가능
        self.last_prices.update(prices)
        equity = self.equity(prices)
        traded = 0.0
        new_qty: dict[str, float] = {}
        for symbol, weight in weights.items():
            if symbol == CASH:
                continue
            price = self._mark(symbol, prices)
            if price <= 0:
                continue  # 가격 불명 종목은 배분 불가 — 현금으로 남김
            target_value = weight * equity
            current_value = self.qty.get(symbol, 0.0) * price
            traded += abs(target_value - current_value)
            new_qty[symbol] = target_value / price
        # 배분에서 빠진 보유 자산은 전량 매도 (유니버스 이탈분은 마지막 관측가로 mark)
        for symbol, q in self.qty.items():
            if symbol not in new_qty and q > 0:
                traded += q * self._mark(symbol, prices)

        cost = traded * cost_bps / 1e4
        invested = sum(q * self._mark(s, prices) for s, q in new_qty.items())
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
                {
                    "cash": self.cash,
                    "qty": self.qty,
                    "history": self.history,
                    "last_prices": self.last_prices,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
