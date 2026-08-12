"""크립토 어댑터 — ccxt + Binance Spot Testnet (선봉 시장).

24/7 시장이라 표본 축적이 가장 빠르다. universe 는 생성자에서 고정 —
잔고에 잡히는 무관한 testnet 자산 수백 종을 평가 대상에서 배제하기 위함.

시세/주문 분리: OHLCV·티커는 메인넷 공개 API(실제 시장 데이터, testnet 클라인은
주기 초기화로 이력 ~1개월뿐), 주문·잔고만 testnet. 관측 품질과 검증 격리를 동시에.
같은 분리를 체결 거래소가 바뀌어도 유지하려고 시세 채널을 BinanceDataFeed 로 떼어냈다
(Upbit 체결 어댑터가 관측을 USDT 기준 그대로 물려받는다 — 심볼 계보 보존).
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone

from adapters.allocation import build_budget, project_to_executable, weights_from_quantities
from adapters.base import (
    Bar,
    MarketAdapter,
    NewsItem,
    OrderResult,
    Position,
    observation_window,
)
from adapters.retry import with_retry

OHLCV_LIMIT = 1000  # Binance klines 1회 응답 행 수 상한 — 넘겨 요청해도 여기서 잘린다


async def _ensure_markets(client) -> None:
    """ccxt 마켓 메타를 1회만 프리로드(캐시). markets 가 이미 있으면 즉시 반환.

    exchangeInfo 응답은 드물게 느리거나 일시 불가(ExchangeNotAvailable/타임아웃)할 수 있고,
    그동안 첫 데이터 조회 안에서 lazily 터지면 크립토 스텝 전체가 죽는다. 일반 조회(3회)보다
    넉넉한 재시도(5회·백오프 2s→)로 감싸 더 긴 일시 장애 구간을 흡수한다.
    """
    if not client.markets:
        await with_retry(client.load_markets, attempts=5, base_delay=2.0)


class BinanceDataFeed:
    """Binance 메인넷 공개 API 시세·뉴스 채널 (키 불필요).

    체결 거래소와 분리된 관측 원천이다. 체결이 어디서 일어나든 관측 심볼·통화(USDT)를
    한 곳으로 고정해, 거래소를 바꿔도 봉·피처·메모리 pattern_key 의 시계열 계보가
    끊기지 않게 한다. 체결 venue 어댑터가 상속해 쓴다.
    """

    def __init__(self) -> None:
        import ccxt.async_support as ccxt_async

        # timeout(ms) 명시 — 무설정 시 라이브러리 기본에 의존하지 않도록 브로커 REST(15s)와 통일
        self.data = ccxt_async.binance({"timeout": 15000})

    async def close_data(self) -> None:
        await self.data.close()

    async def _fetch_bars(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        """메인넷 일봉 조회 후 [start, end] 로 필터 — 진행 중인 당일 봉 차단."""
        since = int(datetime.combine(start, time(), tzinfo=timezone.utc).timestamp() * 1000)
        limit = (end - start).days + 3
        await _ensure_markets(self.data)  # exchangeInfo 프리로드(넉넉한 재시도) — lazily 실패 방지
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

    async def _fetch_bars_history(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        """장기 창 전용 — since 커서로 1회 응답 상한(1000행)을 이어붙인다.

        limit 를 요청 일수만큼 키워도 거래소가 1000행에서 자르는데, 자를 때 남는 것은
        since 부터의 **가장 오래된** 1000봉이다 — 창을 길게 잡을수록 최근 구간이 통째로
        빠지고, 그게 봉 개수만 보면 정상으로 보인다. 마지막 봉 다음으로 커서를 밀어
        end 까지 채운다. 상한 t-1 은 [start, end] 필터가 유지한다.
        """
        since = int(datetime.combine(start, time(), tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.combine(end, time(), tzinfo=timezone.utc).timestamp() * 1000)
        await _ensure_markets(self.data)
        out: dict[str, list[Bar]] = {}
        for symbol in symbols:
            bars: dict[date, Bar] = {}
            cursor = since
            while cursor <= end_ms:
                raw = await with_retry(
                    lambda s=symbol, cur=cursor: self.data.fetch_ohlcv(s, "1d", cur, OHLCV_LIMIT)
                )
                if not raw:
                    break
                for ts, o, h, lo, c, v in raw:
                    day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
                    if start <= day <= end:
                        bars[day] = Bar(day=day, open=o, high=h, low=lo, close=c, volume=v)
                if len(raw) < OHLCV_LIMIT:  # 마지막 페이지
                    break
                cursor = int(raw[-1][0]) + 1
            out[symbol] = sorted(bars.values(), key=lambda b: b.day)
        return out

    async def get_news(self, symbols: list[str], asof_day: date) -> list[NewsItem]:
        # 무료 RSS(CoinDesk·Cointelegraph) — 시장 전반 헤드라인, 심볼 필터 없음
        from adapters.news_rss import fetch_rss_news

        start, end = observation_window(asof_day)
        return await fetch_rss_news(start, end)

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """실시간 체결가 — 메인넷 공개 티커(당일, 행동 전용)."""
        await _ensure_markets(self.data)
        out: dict[str, float] = {}
        for symbol in symbols:
            ticker = await with_retry(lambda s=symbol: self.data.fetch_ticker(s))
            out[symbol] = float(ticker["last"])
        return out


class BinanceTestnetAdapter(BinanceDataFeed, MarketAdapter):
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

        super().__init__()  # 시세 채널(메인넷 공개)
        self.ex = ccxt_async.binance({"apiKey": api_key, "secret": secret, "timeout": 15000})
        self.ex.set_sandbox_mode(True)  # testnet.binance.vision 라우팅 (주문·잔고)
        self.universe = universe
        self.quote = quote
        self.min_notional = min_notional

    async def close(self) -> None:
        """aiohttp 세션 정리. 사용 후 반드시 호출."""
        await self.ex.close()
        await self.close_data()

    async def _snapshot(self) -> tuple[float, dict[str, float], dict[str, float]]:
        """(quote 현금, 보유수량, 현재가) — 잔고 1회 + 유니버스 전 종목 티커 조회."""
        await _ensure_markets(self.data)  # 티커 조회 전 시세 클라이언트 마켓 프리로드
        balance = await with_retry(self.ex.fetch_balance)
        cash = float(balance.get("total", {}).get(self.quote) or 0)
        qty: dict[str, float] = {}
        prices: dict[str, float] = {}
        for symbol in self.universe:
            base = symbol.split("/")[0]
            held = float(balance.get("total", {}).get(base) or 0)
            if held > 0:
                qty[symbol] = held
            ticker = await with_retry(lambda s=symbol: self.data.fetch_ticker(s))
            prices[symbol] = float(ticker["last"])
        return cash, qty, prices

    async def get_equity(self) -> float:
        cash, qty, prices = await self._snapshot()
        return cash + sum(q * prices[s] for s, q in qty.items())

    async def get_budget(self, unit_prices: dict[str, float]):
        """분수 거래 — 거래단위 제약 없음. 최소 주문 금액만 비중으로 환산한다."""
        cash, qty, prices = await self._snapshot()
        equity = cash + sum(q * prices[s] for s, q in qty.items())
        return build_budget(self.quote, equity, cash, {}, min_order=self.min_notional)

    async def get_positions(self) -> list[Position]:
        _, qty, prices = await self._snapshot()
        # spot 잔고에는 취득단가가 없다 — avg_price=0.0 은 "미상" 표기
        return [
            Position(symbol=s, quantity=q, avg_price=0.0, market_value=q * prices[s])
            for s, q in qty.items()
        ]

    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        now = datetime.now(timezone.utc)
        try:
            cash, qty, prices = await self._snapshot()
            # 분수 거래 — 의도를 잘라내는 건 최소 주문 금액뿐이다.
            plan = project_to_executable(
                weights, qty, cash, prices, min_notional=self.min_notional
            )
            if plan.intents:
                await _ensure_markets(self.ex)  # amount_to_precision 에 필요 (넉넉한 재시도·캐시됨)
            final_qty = dict(qty)
            orders = []
            for it in plan.intents:
                amount = float(self.ex.amount_to_precision(it.symbol, it.qty))
                placed = await self.ex.create_order(it.symbol, "market", it.side, amount)
                final_qty[it.symbol] = final_qty.get(it.symbol, 0.0) + (
                    amount if it.side == "buy" else -amount
                )
                orders.append(
                    {
                        "symbol": it.symbol,
                        "side": it.side,
                        "qty": amount,
                        "notional": round(it.notional, 2),
                        "order_id": placed.get("id"),
                    }
                )
            total = cash + sum(q * prices[s] for s, q in qty.items())
            return OrderResult(
                market=self.market,
                submitted_at=now,
                accepted=True,
                orders=orders,
                executed_weights=weights_from_quantities(final_qty, prices, total),
                dropped=plan.dropped,
            )
        except Exception as e:  # 주문 실패는 예외가 아니라 결과로 — 러너가 로그로 남긴다
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=str(e)[:300]
            )
