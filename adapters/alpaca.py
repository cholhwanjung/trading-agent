"""미국 주식 어댑터 — Alpaca Paper Trading.

주문은 notional(달러 금액) 방식 — 가격 조회 없이 Δ평가액을 그대로 제출할 수 있어
분할 주식(fractional) 포함 배분 정밀도가 높다. 데이터는 무료 IEX 피드 고정.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import httpx

from adapters.allocation import compute_order_deltas
from adapters.base import (
    Bar,
    MarketAdapter,
    NewsItem,
    OrderResult,
    Position,
    observation_window,
)
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
    ) -> None:
        self.universe = universe
        self.min_notional = min_notional
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

    async def get_ohlcv(self, symbols: list[str], asof_day: date) -> dict[str, list[Bar]]:
        start, end = observation_window(asof_day)
        return await self._fetch_bars(symbols, start, end)

    async def get_ohlcv_history(
        self, symbols: list[str], asof_day: date, lookback_days: int = 90
    ) -> dict[str, list[Bar]]:
        from datetime import timedelta

        # 상한 t-1
        return await self._fetch_bars(
            symbols, asof_day - timedelta(days=lookback_days), asof_day - timedelta(days=1)
        )

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
        start, end = observation_window(asof_day)
        data = await self._get(
            f"{DATA_BASE}/v1beta1/news",
            params={
                "symbols": ",".join(symbols),
                "start": f"{start.isoformat()}T00:00:00Z",
                "end": f"{end.isoformat()}T23:59:59Z",
                "limit": 50,
            },
        )
        items = []
        for n in data.get("news") or []:
            published = datetime.fromisoformat(n["created_at"].replace("Z", "+00:00"))
            if start <= published.date() <= end:  # 윈도우 재확인
                items.append(
                    NewsItem(
                        published_at=published,
                        headline=n.get("headline", ""),
                        source=n.get("source", "alpaca"),
                        url=n.get("url"),
                    )
                )
        return items

    async def get_equity(self) -> float:
        account = await self._get(f"{TRADE_BASE}/v2/account")
        return float(account["equity"])

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
        try:
            account = await self._get(f"{TRADE_BASE}/v2/account")
            cash = float(account["cash"])
            holdings = {p.symbol: p.market_value for p in await self.get_positions()}

            # 미체결(pending) 주문이 있는 종목은 제외 — 포지션에 안 잡혀 중복 주문이 나간다
            open_orders = await self._get(f"{TRADE_BASE}/v2/orders", params={"status": "open"})
            pending = {o["symbol"] for o in open_orders}

            # notional 주문이라 가격 불필요 (qty=None)
            intents = compute_order_deltas(
                weights, holdings, cash, prices=None, min_notional=self.min_notional
            )
            orders = []
            for it in intents:
                if it.symbol in pending:
                    orders.append({"symbol": it.symbol, "side": it.side, "skipped": "open_order"})
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
                orders.append(
                    {
                        "symbol": it.symbol,
                        "side": it.side,
                        "notional": round(it.notional, 2),
                        "order_id": placed.get("id"),
                        "status": placed.get("status"),
                    }
                )
            return OrderResult(market=self.market, submitted_at=now, accepted=True, orders=orders)
        except Exception as e:
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=str(e)[:300]
            )
