"""연구 유니버스 데이터 — ccxt 공개 일봉 → 정렬 패널 (무료·일간).

연구 유니버스(횡단면 IC 용)는 매매 유니버스보다 넓다 — 크립토
상위 유동성 10종. 상한 t-1: 진행 중인 당일 봉 제외.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np

# 연구 유니버스 — Binance 상위 유동성 USDT 페어 (정적 목록, 분기 리뷰)
CRYPTO_RESEARCH_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT",
]


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

    common = sorted(set.intersection(*(set(r) for r in per_symbol.values())))
    if not common:
        raise ValueError("공통 거래일 없음 — 유니버스 확인")

    T, N = len(common), len(symbols)
    panel = {f: np.full((T, N), np.nan) for f in ("open", "high", "low", "close", "volume")}
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
