"""크립토 어댑터 — ccxt + Binance Spot Testnet (Phase 0 선봉 시장).

24/7 시장이라 표본 축적이 가장 빠르다. universe 는 생성자에서 고정 —
잔고에 잡히는 무관한 testnet 자산 수백 종을 평가 대상에서 배제하기 위함.

시세/주문 분리: OHLCV·티커는 메인넷 공개 API(실제 시장 데이터, testnet 클라인은
주기 초기화로 이력 ~1개월뿐), 주문·잔고만 testnet. 관측 품질과 검증 격리를 동시에.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone

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


class BinanceTestnetAdapter(MarketAdapter):
    market = "CRYPTO"

    def __init__(
        self,
        api_key: str,
        secret: str,
        universe: list[str],  # 예: ["BTC/USDT", "ETH/USDT"]
        quote: str = "USDT",
        min_notional: float = 10.0,  # Binance spot 최소 주문 금액
    ) -> None:
        import ccxt.async_support as ccxt_async

        self.ex = ccxt_async.binance({"apiKey": api_key, "secret": secret})
        self.ex.set_sandbox_mode(True)  # testnet.binance.vision 라우팅 (주문·잔고)
        self.data = ccxt_async.binance()  # 메인넷 공개 API (시세 — 키 불필요)
        self.universe = universe
        self.quote = quote
        self.min_notional = min_notional

    async def close(self) -> None:
        """aiohttp 세션 정리. 사용 후 반드시 호출."""
        await self.ex.close()
        await self.data.close()

    async def _fetch_bars(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        """메인넷 일봉 조회 후 [start, end] 로 필터 — 진행 중인 당일 봉 차단 (R2)."""
        since = int(datetime.combine(start, time(), tzinfo=timezone.utc).timestamp() * 1000)
        limit = (end - start).days + 3
        out: dict[str, list[Bar]] = {}
        for symbol in symbols:
            raw = await with_retry(lambda s=symbol: self.data.fetch_ohlcv(s, "1d", since, limit))
            bars = []
            for ts, o, h, lo, c, v in raw:
                day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
                if start <= day <= end:
                    bars.append(Bar(day=day, open=o, high=h, low=lo, close=c, volume=v))
            out[symbol] = bars
        return out

    async def get_ohlcv(self, symbols: list[str], asof_day: date) -> dict[str, list[Bar]]:
        start, end = observation_window(asof_day)
        return await self._fetch_bars(symbols, start, end)

    async def get_news(self, symbols: list[str], asof_day: date) -> list[NewsItem]:
        # 무료 RSS(CoinDesk·Cointelegraph) — 시장 전반 헤드라인, 심볼 필터 없음
        from adapters.news_rss import fetch_rss_news

        start, end = observation_window(asof_day)
        return await fetch_rss_news(start, end)

    async def get_ohlcv_history(
        self, symbols: list[str], asof_day: date, lookback_days: int = 90
    ) -> dict[str, list[Bar]]:
        from datetime import timedelta

        # 상한 t-1 (ADR-013)
        return await self._fetch_bars(
            symbols, asof_day - timedelta(days=lookback_days), asof_day - timedelta(days=1)
        )

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """실시간 체결가 — 메인넷 공개 티커(당일, 행동 전용 · [ADR-021])."""
        out: dict[str, float] = {}
        for symbol in symbols:
            ticker = await with_retry(lambda s=symbol: self.data.fetch_ticker(s))
            out[symbol] = float(ticker["last"])
        return out

    async def get_equity(self) -> float:
        balance = await with_retry(self.ex.fetch_balance)
        equity = float(balance.get("total", {}).get(self.quote) or 0)
        for symbol in self.universe:
            base = symbol.split("/")[0]
            qty = float(balance.get("total", {}).get(base) or 0)
            if qty > 0:
                ticker = await with_retry(lambda s=symbol: self.data.fetch_ticker(s))
                equity += qty * float(ticker["last"])
        return equity

    async def get_positions(self) -> list[Position]:
        balance = await with_retry(self.ex.fetch_balance)
        positions = []
        for symbol in self.universe:
            base = symbol.split("/")[0]
            qty = float(balance.get("total", {}).get(base) or 0)
            if qty <= 0:
                continue
            ticker = await with_retry(lambda s=symbol: self.data.fetch_ticker(s))
            price = float(ticker["last"])
            # spot 잔고에는 취득단가가 없다 — avg_price=0.0 은 "미상" 표기
            positions.append(
                Position(symbol=symbol, quantity=qty, avg_price=0.0, market_value=qty * price)
            )
        return positions

    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        now = datetime.now(timezone.utc)
        try:
            balance = await with_retry(self.ex.fetch_balance)
            cash = float(balance.get("total", {}).get(self.quote) or 0)
            holdings: dict[str, float] = {}
            prices: dict[str, float] = {}
            for symbol in self.universe:
                base = symbol.split("/")[0]
                qty = float(balance.get("total", {}).get(base) or 0)
                ticker = await with_retry(lambda s=symbol: self.data.fetch_ticker(s))
                prices[symbol] = float(ticker["last"])
                if qty > 0:
                    holdings[symbol] = qty * prices[symbol]

            intents = compute_order_deltas(
                weights, holdings, cash, prices, min_notional=self.min_notional
            )
            if intents:
                await self.ex.load_markets()  # amount_to_precision 에 필요 (캐시됨)
            orders = []
            for it in intents:
                amount = float(self.ex.amount_to_precision(it.symbol, it.qty))
                placed = await self.ex.create_order(it.symbol, "market", it.side, amount)
                orders.append(
                    {
                        "symbol": it.symbol,
                        "side": it.side,
                        "qty": amount,
                        "notional": round(it.notional, 2),
                        "order_id": placed.get("id"),
                    }
                )
            return OrderResult(market=self.market, submitted_at=now, accepted=True, orders=orders)
        except Exception as e:  # 주문 실패는 예외가 아니라 결과로 — 러너가 로그로 남긴다
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=str(e)[:300]
            )
