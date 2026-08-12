"""배분비율 → 주문 의도(Δq) 변환 — 어댑터 공통 순수 로직.

네트워크 없이 테스트 가능. 어댑터는 이 결과를 각 브로커의 주문 API로 번역만 한다.
CASH 는 주문 대상이 아니라 잔여 현금 목표 — 자산 주문이 체결되면 자연히 맞춰진다.

목표 배분이 언제나 그대로 체결되는 것은 아니다. 정수 주로만 거래되는 시장에서는
1주 가격이 목표 금액보다 비싸면 그 종목이 통째로 빠지고, 최소 주문 금액·1회 명목
상한도 같은 식으로 의도를 잘라낸다. project_to_executable 은 그 절삭을 **주문을
내기 전에** 계산해, 실제로 표현되는 배분이 무엇인지 함께 돌려준다 — 의도한 배분으로
학습하면 보유한 적 없는 포트폴리오에 성과를 귀속시키게 되기 때문이다.
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


@dataclass(frozen=True)
class BudgetSnapshot:
    """이 계좌에서 배분을 얼마나 잘게 표현할 수 있는지 — 결정 시점의 예산 제약.

    모든 값은 **비중**이다. 계좌마다 통화가 다르고 트레이더는 비중으로만 답하므로,
    금액을 그대로 주면 매번 나눗셈을 시키게 되고 그 계산은 자주 틀린다.
    """

    currency: str
    equity: float
    cash_weight: float  # 지금 현금 비중
    lot: float  # 최소 거래단위(0 = 분수 거래)
    min_order_weight: float  # 최소 주문 금액의 비중 환산
    max_order_weight: float | None  # 1회 주문 상한의 비중 환산
    min_step_weight: dict[str, float]  # symbol -> 1단위(1주)의 비중. 0 = 제약 없음


def bucket_cash(cash: float, share: float) -> float:
    """한 계좌를 여러 시장이 공유할 때, 그 시장이 쓸 수 있는 현금.

    시장마다 배분 벡터가 따로 있고 각각 합이 1 이라, 같은 현금을 두 시장이 각자
    자기 것으로 보면 합계 200% 를 주문하려 든다. 먼저 도는 쪽이 매수여력을 다 쓰고
    나머지는 통째로 스킵되므로 결과가 실행 순서에 좌우된다.

    나누는 것은 **현금뿐**이다 — 보유 종목은 어느 시장 것인지 이미 명확하다. 지분의
    합이 1 이면 시장별 순자산의 합이 계좌 총자산과 같아진다.
    """

    return cash * share


def build_budget(
    currency: str,
    equity: float,
    cash: float,
    unit_prices: dict[str, float],
    *,
    lot: float = 0.0,
    min_order: float = 0.0,
    max_order: float | None = None,
) -> BudgetSnapshot | None:
    """예산 제약을 비중 공간으로 환산. equity 가 0 이하면 판단할 근거가 없어 None.

    unit_prices 는 **계좌 통화 기준** 1단위 가격이다. 분수 거래(lot=0)면 쓰이지 않는다.
    """
    if equity <= 0:
        return None
    return BudgetSnapshot(
        currency=currency,
        equity=round(equity, 2),
        cash_weight=round(cash / equity, 4),
        lot=lot,
        min_order_weight=round(min_order / equity, 6),
        max_order_weight=round(max_order / equity, 6) if max_order is not None else None,
        min_step_weight={
            s: round(lot * p / equity, 6) if lot > 0 else 0.0
            for s, p in unit_prices.items()
            if p
        },
    )


@dataclass(frozen=True)
class Projection:
    """목표 배분을 그 계좌에서 실제로 낼 수 있는 주문으로 투영한 결과."""

    weights: dict[str, float]  # 이 계획대로 체결됐을 때 표현되는 배분 (CASH 포함, 합=1)
    intents: list[OrderIntent]  # 낼 수 있는 주문만. qty 는 거래단위 배수
    dropped: dict[str, str]  # symbol -> 의도가 표현되지 못한 사유


def _quantize(qty: float, lot: float) -> float:
    """거래단위 내림. lot=0 은 분수 거래(그대로), 음수 수량은 발생하지 않는다."""
    if lot <= 0:
        return max(qty, 0.0)
    return max(int(qty / lot) * lot, 0.0)


def weights_from_quantities(
    quantities: dict[str, float], prices: dict[str, float], total: float
) -> dict[str, float]:
    """보유 수량 → 배분비율. 잔여는 CASH. total 이 0 이면 전액 현금으로 본다."""
    if total <= 0:
        return {CASH: 1.0}
    weights = {
        s: round(q * prices[s] / total, 6)
        for s, q in quantities.items()
        if q > 0 and prices.get(s)
    }
    weights[CASH] = round(1.0 - sum(weights.values()), 6)
    return weights


def project_to_executable(
    weights: dict[str, float],
    holdings_qty: dict[str, float],  # symbol -> 현재 보유 수량
    cash: float,
    prices: dict[str, float],  # symbol -> 1주(1단위) 가격, 계좌 통화
    *,
    lot: float = 0.0,  # 최소 거래단위. 0 = 분수 거래 가능
    min_notional: float = 0.0,
    max_order_notional: float | None = None,  # 1회 주문 명목 상한(있으면)
) -> Projection:
    """목표 배분 → (실제 표현되는 배분, 낼 수 있는 주문, 못 낸 사유).

    **목표를 넘기는 방향으로는 절대 올리지 않는다.** 거래단위 때문에 남는 현금을
    한 주 더 사는 데 쓰면 그 종목이 리스크 레이어가 정한 종목당 상한을 넘어설 수
    있다 — 상한은 그 위 레이어에서 이미 강제된 값이므로 여기서 되돌리면 안 된다.
    그래서 수량은 항상 내림이고, 표현되지 못한 비중은 현금으로 남는다.

    dropped 에는 **의미 있는 크기의** 의도만 담는다(최소 주문 금액 이상 어긋난 경우).
    소수점 아래 잔량 정도의 어긋남까지 사유로 쌓으면 매일 잡음만 늘고 진짜 절삭이 묻힌다.
    """
    total = cash + sum(q * prices[s] for s, q in holdings_qty.items() if prices.get(s))
    symbols = sorted((set(weights) | set(holdings_qty)) - {CASH})
    final_qty: dict[str, float] = {}
    dropped: dict[str, str] = {}
    buys: list[tuple[float, str, float]] = []  # (목표비중, symbol, 매수 수량)
    proceeds = 0.0
    intents: list[OrderIntent] = []

    for symbol in symbols:
        held = holdings_qty.get(symbol, 0.0)
        price = prices.get(symbol)
        if not price:
            final_qty[symbol] = held
            dropped[symbol] = "no_price"
            continue
        target_value = weights.get(symbol, 0.0) * total
        qty = _quantize(target_value / price, lot)
        delta = qty - held
        notional = abs(delta) * price
        reason: str | None = None
        if delta != 0 and notional < min_notional:
            qty, reason = held, "below_min_notional"
        elif max_order_notional is not None and notional > max_order_notional:
            qty, reason = held, "over_order_cap"
        elif delta < 0:
            proceeds += notional
            intents.append(
                OrderIntent(symbol=symbol, side="sell", notional=notional, qty=abs(delta))
            )
        elif delta > 0:
            buys.append((weights.get(symbol, 0.0), symbol, delta))
        final_qty[symbol] = qty
        # 어긋난 금액이 최소 주문 미만이면 애초에 낼 수 없는 주문이라 사유로 남기지 않는다.
        gap = abs(target_value - qty * price)
        if gap >= min_notional and gap > 0:
            if reason:
                dropped[symbol] = reason
            elif lot > 0:
                dropped[symbol] = "lot_granularity"  # 1주 단위로 담기지 않아 남은 의도

    # 매수는 현금 한도 안에서만 — 확신이 큰(목표 비중이 높은) 종목부터 채운다.
    # 모자라면 살 수 있는 수량까지 줄이고, 한 단위도 못 사면 그 종목을 뺀다.
    available = cash + proceeds
    for _, symbol, delta in sorted(buys, key=lambda b: (-b[0], b[1])):
        price = prices[symbol]
        affordable = _quantize(min(delta, available / price), lot)
        if affordable <= 0 or affordable * price < min_notional:
            final_qty[symbol] = holdings_qty.get(symbol, 0.0)
            dropped[symbol] = "insufficient_cash"
            continue
        if affordable < delta:
            final_qty[symbol] = holdings_qty.get(symbol, 0.0) + affordable
            dropped[symbol] = "insufficient_cash"
        available -= affordable * price
        intents.append(
            OrderIntent(
                symbol=symbol, side="buy", notional=affordable * price, qty=affordable
            )
        )

    intents.sort(key=lambda i: i.side != "sell")  # 매도 먼저 — 현금 확보 후 매수
    return Projection(
        weights=weights_from_quantities(final_qty, prices, total),
        intents=intents,
        dropped=dropped,
    )
