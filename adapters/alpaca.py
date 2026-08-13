"""미국 주식 어댑터 — Alpaca Paper Trading.

주문은 notional(달러 금액) 방식 — 가격 조회 없이 Δ평가액을 그대로 제출할 수 있어
분할 주식(fractional) 포함 배분 정밀도가 높다. 데이터는 무료 IEX 피드 고정.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import httpx

from adapters.allocation import (
    CASH,
    build_budget,
    project_to_executable,
    weights_from_quantities,
)
from adapters.base import (
    Bar,
    MarketAdapter,
    NewsItem,
    OrderResult,
    Position,
    observation_window,
    off_session_weekday,
)
from adapters.financials import Financials
from adapters.retry import with_retry

TRADE_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


class AlpacaPaperAdapter(MarketAdapter):
    market = "US"

    def __init__(
        self,
        api_key: str,
        secret: str,
        universe: list[str],  # 예: ["SPY"]
        min_notional: float = 10.0,
        sec_user_agent: str | None = None,  # SEC 공시·재무 조회 식별자 (키 아님)
    ) -> None:
        self.universe = universe
        self.min_notional = min_notional
        self._key, self._secret = api_key, secret
        self._sec_user_agent = sec_user_agent
        self._client = httpx.AsyncClient(
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret},
            timeout=15.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, url: str, params: dict | None = None) -> dict | list:
        async def call():
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

        return await with_retry(call, exceptions=(httpx.HTTPError,))

    async def _fetch_bars(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        data = await self._get(
            f"{DATA_BASE}/v2/stocks/bars",
            params={
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": f"{start.isoformat()}T00:00:00Z",
                "end": f"{end.isoformat()}T23:59:59Z",
                "feed": "iex",  # 무료 피드 (유료 데이터 미사용)
                "limit": 1000,
            },
        )
        out: dict[str, list[Bar]] = {s: [] for s in symbols}
        for symbol, raw_bars in (data.get("bars") or {}).items():
            for rb in raw_bars:
                day = datetime.fromisoformat(rb["t"].replace("Z", "+00:00")).date()
                if start <= day <= end:  # 구간 재확인
                    out[symbol].append(
                        Bar(
                            day=day,
                            open=rb["o"],
                            high=rb["h"],
                            low=rb["l"],
                            close=rb["c"],
                            volume=rb["v"],
                        )
                    )
        return out

    async def get_news(self, symbols: list[str], asof_day: date) -> list[NewsItem]:
        from adapters.edgar import fetch_edgar_filings
        from adapters.news_us import fetch_us_news

        start, end = observation_window(asof_day)
        news = await fetch_us_news(self._key, self._secret, symbols, start, end)
        # 공시를 구조화 이벤트로 관측에 편입 (같은 뉴스 채널). 헤드라인은 의무공시의 대체재가
        # 아니다 — 8-K 는 4영업일 내 제출 의무라 적시성·완결성이 다르다.
        news += await fetch_edgar_filings(
            symbols, start, end, user_agent=self._sec_user_agent
        )
        return news

    async def get_financials(self, symbols: list[str], asof_day: date) -> dict[str, Financials]:
        from adapters.edgar import fetch_edgar_financials

        _, end = observation_window(asof_day)
        return await fetch_edgar_financials(symbols, end, user_agent=self._sec_user_agent)

    async def get_equity(self) -> float:
        account = await self._get(f"{TRADE_BASE}/v2/account")
        return float(account["equity"])

    async def get_budget(self, unit_prices: dict[str, float]):
        """notional 주문(분수 주)이라 1주 가격이 배분을 제약하지 않는다."""
        account = await self._get(f"{TRADE_BASE}/v2/account")
        return build_budget(
            "USD",
            float(account["equity"]),
            float(account["cash"]),
            {},
            min_order=self.min_notional,
        )

    async def get_positions(self) -> list[Position]:
        raw = await self._get(f"{TRADE_BASE}/v2/positions")
        return [
            Position(
                symbol=p["symbol"],
                quantity=float(p["qty"]),
                avg_price=float(p["avg_entry_price"]),
                market_value=float(p["market_value"]),
            )
            for p in raw
        ]

    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        now = datetime.now(timezone.utc)
        # 평일 정규장 밖이면 주문하지 않는다. 이 브로커는 장외 주문을 거부하는 대신
        # 다음 개장까지 대기열에 넣는데, 그러면 오늘의 결정이 내일 시가에 체결되어
        # 결정 시점과 체결 시점이 어긋난 채로 기록이 남는다.
        off_session = off_session_weekday(self.market, now)
        if off_session:
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=off_session
            )
        try:
            account = await self._get(f"{TRADE_BASE}/v2/account")
            cash = float(account["cash"])
            positions = await self.get_positions()
            # 분수 주 notional 주문이라 수량 대신 평가액을 '단가 1' 로 다뤄 투영한다 —
            # 이 계좌에서는 거래단위가 의도를 잘라내지 않으므로 환산이 곧 항등이다.
            values = {p.symbol: p.market_value for p in positions}
            unit = {s: 1.0 for s in set(weights) | set(values) if s != CASH}

            # 미체결(pending) 주문이 있는 종목은 제외 — 포지션에 안 잡혀 중복 주문이 나간다
            open_orders = await self._get(f"{TRADE_BASE}/v2/orders", params={"status": "open"})
            pending = {o["symbol"] for o in open_orders}

            plan = project_to_executable(
                weights, values, cash, unit, min_notional=self.min_notional
            )
            final_values = dict(values)
            orders = []
            for it in plan.intents:
                if it.symbol in pending:
                    orders.append({"symbol": it.symbol, "side": it.side, "skipped": "open_order"})
                    plan.dropped[it.symbol] = "open_order"
                    continue
                resp = await self._client.post(
                    f"{TRADE_BASE}/v2/orders",
                    json={
                        "symbol": it.symbol,
                        "notional": str(round(it.notional, 2)),
                        "side": it.side,
                        "type": "market",
                        "time_in_force": "day",
                    },
                )
                resp.raise_for_status()
                placed = resp.json()
                final_values[it.symbol] = final_values.get(it.symbol, 0.0) + (
                    it.notional if it.side == "buy" else -it.notional
                )
                orders.append(
                    {
                        "symbol": it.symbol,
                        "side": it.side,
                        "notional": round(it.notional, 2),
                        "order_id": placed.get("id"),
                        "status": placed.get("status"),
                    }
                )
            return OrderResult(
                market=self.market,
                submitted_at=now,
                accepted=True,
                orders=orders,
                executed_weights=weights_from_quantities(
                    final_values, unit, cash + sum(values.values())
                ),
                dropped=plan.dropped,
            )
        except Exception as e:
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=str(e)[:300]
            )
