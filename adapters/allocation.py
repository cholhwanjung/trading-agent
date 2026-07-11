"""배분비율 → 주문 의도(Δq) 변환 — 어댑터 공통 순수 로직 ([ADR-006]).

네트워크 없이 테스트 가능. 어댑터는 이 결과를 각 브로커의 주문 API로 번역만 한다.
CASH 는 주문 대상이 아니라 잔여 현금 목표 — 자산 주문이 체결되면 자연히 맞춰진다.
"""

from __future__ import annotations

from dataclasses import dataclass

CASH = "CASH"


@dataclass(frozen=True)
class OrderIntent:
    """주문 의도 1건. notional 은 quote 통화 절대값, qty 는 price 제공 시에만."""

    symbol: str
    side: str  # "buy" | "sell"
    notional: float
    qty: float | None


def compute_order_deltas(
    weights: dict[str, float],
    holdings: dict[str, float],  # symbol -> 현재 평가액 (quote 통화)
    cash: float,
    prices: dict[str, float] | None = None,
    min_notional: float = 10.0,
) -> list[OrderIntent]:
    """목표 배분과 현재 보유의 차이를 주문 의도 리스트로 변환.

    - weights 에 없는 보유 자산은 목표 0 으로 간주해 전량 매도 대상.
    - min_notional 미만의 차이는 무시(브로커 최소 주문 금액 + 잔주문 방지).
    - 매도가 매수보다 앞에 온다(현금 확보 후 매수).
    """

    total = cash + sum(holdings.values())
    intents: list[OrderIntent] = []
    for symbol in sorted(set(weights) | set(holdings)):
        if symbol == CASH:
            continue
        target = weights.get(symbol, 0.0) * total
        delta = target - holdings.get(symbol, 0.0)
        if abs(delta) < min_notional:
            continue
        side = "buy" if delta > 0 else "sell"
        price = (prices or {}).get(symbol)
        qty = abs(delta) / price if price else None
        intents.append(OrderIntent(symbol=symbol, side=side, notional=abs(delta), qty=qty))
    intents.sort(key=lambda i: i.side != "sell")
    return intents
