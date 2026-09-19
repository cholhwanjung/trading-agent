"""가상 포트폴리오 — 동일 관측·실가격 기반 병행 페이퍼 운용 (LiveTradeBench 방식).

백테스트가 아니다: 매일 라이브로 도착하는 봉에 목표 배분을 적용하는 forward
시뮬레이션(미래 참조 없음). 정책별 상태를 JSON 으로 영속해 equity 곡선을 누적 —
LLM vs B&H vs 랜덤의 델타 측정이 목적.

체결 규약: 결정은 **다음 봉의 시가**에 체결한다. 결정은 직전 봉이 닫힌 뒤의 정보(밤사이
뉴스·해외장)까지 보고 내려지므로, 직전 종가에 체결하면 이미 본 정보보다 앞선 가격으로
사는 셈이 된다 — 그 시간차의 수익이 정책의 공으로 잡히고, 정보를 쓰지 않는 기준선
(B&H)에는 생기지 않아 비교가 한쪽으로 기운다. 다음 봉 시가는 정보 상한 이후 처음
거래할 수 있는 가격이다. 그래서 오늘의 결정은 pending 으로 두었다가 다음 봉이 도착하면
그 시가로 체결하고 종가로 마킹한다.

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
            # 아직 체결되지 않은 최신 결정 — 다음 봉 시가에 체결된다
            self.pending: dict[str, float] | None = state.get("pending")
        else:
            self.cash = initial_cash
            self.qty = {}
            self.history = []
            self.last_prices = {}
            self.pending = None

    def _mark(self, symbol: str, prices: dict[str, float]) -> float:
        """현재가 우선, 없으면(유니버스 이탈) 마지막 관측가, 그것도 없으면 0.0."""
        price = prices.get(symbol)
        return price if price is not None else self.last_prices.get(symbol, 0.0)

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * self._mark(s, prices) for s, q in self.qty.items() if q > 0)

    def step(
        self,
        day: date,
        opens: dict[str, float],
        closes: dict[str, float],
        weights: dict[str, float] | None,
        cost_bps: float = 10.0,
    ) -> float:
        """봉 1개 진행 — 직전 결정을 이 봉의 시가에 체결, 종가로 마킹, 오늘 결정은 pending.

        weights=None 은 새 결정이 없다는 뜻이다(결정 실패일 · 매수 후 보유) — 보유를 그대로
        둔다. 같은 봉을 다시 받으면(휴장일 재실행) 체결·마킹은 건너뛰고 pending 만 최신
        결정으로 바꾼다: 다음 시가에 유효한 것은 그 전에 내린 마지막 결정이다.
        """

        day_key = day.isoformat()
        if self.history and self.history[-1]["day"] == day_key:
            if weights is not None:
                self.pending = weights
                self._save()
            return self.history[-1]["equity"]  # 이 봉은 이미 체결·마킹됐다

        cost = 0.0
        equity_open = self.equity(opens)
        if self.pending is not None:
            cost = self._rebalance(self.pending, opens, equity_open, cost_bps)
            self.pending = None

        # 관측된 종가를 누적 — 이후 유니버스에서 빠져도 마지막 관측가로 청산 가능
        self.last_prices.update(closes)
        after = self.equity(closes)
        self.history.append({
            "day": day_key,
            "equity": round(after, 2),
            # 체결 직전(시가) 평가액 — 밤사이 수익과 장중 수익을 가를 때 쓴다
            "equity_open": round(equity_open, 2),
            "cost": round(cost, 4),
        })
        if weights is not None:
            self.pending = weights
        self._save()
        return after

    def _rebalance(
        self, weights: dict[str, float], prices: dict[str, float], equity: float, cost_bps: float
    ) -> float:
        """목표 배분으로 리밸런싱하고 거래비용을 돌려준다. 가격이 없으면 마지막 관측가로 친다."""
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
        return cost

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "cash": self.cash,
                    "qty": self.qty,
                    "history": self.history,
                    "last_prices": self.last_prices,
                    "pending": self.pending,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
