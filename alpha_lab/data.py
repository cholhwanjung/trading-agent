"""연구 유니버스 데이터 — ccxt 공개 일봉 → 정렬 패널 (무료·일간).

연구 유니버스(횡단면 IC 용)는 매매 유니버스보다 넓다 — 크립토
상위 유동성 10종. 상한 t-1: 진행 중인 당일 봉 제외.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np

# 연구 유니버스 — Binance 상위 유동성 USDT 페어 (정적 목록, 분기 리뷰)
CRYPTO_RESEARCH_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT",
]

_PANEL_FEATURES = ("open", "high", "low", "close", "volume")


def _assemble_panel(
    per_symbol: dict[str, dict[date, tuple]], symbols: list[str]
) -> tuple[dict[str, np.ndarray], list[str], list[date]]:
    """종목별 {날짜: (o,h,lo,c,v)} → (panel, symbols, dates). 공통 거래일만 inner-join.

    데이터 없는 종목은 제외(횡단면에서 자동 탈락). returns 는 close 로부터 파생.
    """
    symbols = [s for s in symbols if per_symbol.get(s)]
    if len(symbols) < 2:
        raise ValueError("패널 구성 가능한 종목 < 2 — 데이터/유니버스 확인")
    common = sorted(set.intersection(*(set(per_symbol[s]) for s in symbols)))
    if not common:
        raise ValueError("공통 거래일 없음 — 유니버스 확인")

    T, N = len(common), len(symbols)
    panel = {f: np.full((T, N), np.nan) for f in _PANEL_FEATURES}
    for j, symbol in enumerate(symbols):
        for i, day in enumerate(common):
            o, h, lo, c, v = per_symbol[symbol][day]
            panel["open"][i, j] = o
            panel["high"][i, j] = h
            panel["low"][i, j] = lo
            panel["close"][i, j] = c
            panel["volume"][i, j] = v
    close = panel["close"]
    returns = np.full_like(close, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns[1:] = close[1:] / close[:-1] - 1.0
    panel["returns"] = returns
    return panel, symbols, common


async def fetch_crypto_panel(
    symbols: list[str] | None = None,
    lookback_days: int = 730,
    asof_day: date | None = None,
) -> tuple[dict[str, np.ndarray], list[str], list[date]]:
    """메인넷 공개 OHLCV → (panel, symbols, dates). 공통 날짜만 정렬(inner join)."""
    import ccxt.async_support as ccxt_async

    symbols = symbols or CRYPTO_RESEARCH_UNIVERSE
    asof_day = asof_day or datetime.now(timezone.utc).date()
    end = asof_day - timedelta(days=1)
    since = int(
        datetime.combine(end - timedelta(days=lookback_days), datetime.min.time(),
                         tzinfo=timezone.utc).timestamp() * 1000
    )

    ex = ccxt_async.binance()
    per_symbol: dict[str, dict[date, tuple]] = {}
    try:
        for symbol in symbols:
            rows: dict[date, tuple] = {}
            cursor = since
            while True:
                raw = await ex.fetch_ohlcv(symbol, "1d", cursor, 1000)
                if not raw:
                    break
                for ts, o, h, lo, c, v in raw:
                    day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
                    if day <= end:
                        rows[day] = (o, h, lo, c, v)
                if len(raw) < 1000:
                    break
                cursor = raw[-1][0] + 1
            per_symbol[symbol] = rows
    finally:
        await ex.close()

    return _assemble_panel(per_symbol, symbols)


# 연구 유니버스 — 매매 메가캡(5) + 유동성 상위 확장(7). 횡단면 IC 추정 분산을 낮춘다.
US_RESEARCH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "TSLA", "JPM", "V", "JNJ", "WMT",
]

ALPACA_DATA_BARS = "https://data.alpaca.markets/v2/stocks/bars"


async def fetch_us_panel(
    api_key: str,
    secret: str,
    symbols: list[str] | None = None,
    lookback_days: int = 730,
    asof_day: date | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, np.ndarray], list[str], list[date]]:
    """Alpaca 무료 IEX 일봉 → (panel, symbols, dates). 상한 t-1(진행 중 당일 봉 제외).

    2년 일봉은 종목당 ~500봉(< limit)이라 페이지네이션 불필요 — 종목별 1회 조회.
    """
    symbols = symbols or US_RESEARCH_UNIVERSE
    asof_day = asof_day or datetime.now(timezone.utc).date()
    end = asof_day - timedelta(days=1)
    start = end - timedelta(days=lookback_days)

    own_client = client is None
    client = client or httpx.AsyncClient(
        headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}, timeout=30.0
    )
    per_symbol: dict[str, dict[date, tuple]] = {}
    try:
        for symbol in symbols:
            resp = await client.get(
                ALPACA_DATA_BARS,
                params={
                    "symbols": symbol,
                    "timeframe": "1Day",
                    "start": f"{start.isoformat()}T00:00:00Z",
                    "end": f"{end.isoformat()}T23:59:59Z",
                    "feed": "iex",  # 무료 피드 (유료 데이터 미사용)
                    "limit": 10000,
                },
            )
            resp.raise_for_status()
            rows: dict[date, tuple] = {}
            for rb in (resp.json().get("bars") or {}).get(symbol, []):
                day = datetime.fromisoformat(rb["t"].replace("Z", "+00:00")).date()
                if day <= end:  # 상한 재확인
                    rows[day] = (rb["o"], rb["h"], rb["l"], rb["c"], rb["v"])
            per_symbol[symbol] = rows
    finally:
        if own_client:
            await client.aclose()

    return _assemble_panel(per_symbol, symbols)


def make_us_panel_fn(env: dict[str, str]):
    """env 의 Alpaca 키로 US 패널 fetch 클로저 생성. 키 없으면 None(US alpha 스킵)."""
    key, secret = env.get("ALPACA_PAPER_API_KEY"), env.get("ALPACA_PAPER_SECRET")
    if not (key and secret):
        return None

    async def _fn(symbols=None, lookback_days=730, asof_day=None):
        return await fetch_us_panel(
            key, secret, symbols=symbols, lookback_days=lookback_days, asof_day=asof_day
        )

    return _fn
