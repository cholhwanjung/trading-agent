"""연구 유니버스 데이터 — 공개 일봉 → 정렬 패널 (무료·일간).

연구 유니버스(횡단면 IC 용)는 매매 유니버스보다 넓다 — 크립토
상위 유동성 10종. 상한 t-1: 진행 중인 당일 봉 제외.
패널 필드: OHLCV + vwap(거래량가중평균가) + dollar_volume(거래대금) + trade_count(체결수).
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

_PANEL_FEATURES = ("open", "high", "low", "close", "volume", "vwap", "dollar_volume", "trade_count")


def _assemble_panel(
    per_symbol: dict[str, dict[date, tuple]], symbols: list[str]
) -> tuple[dict[str, np.ndarray], list[str], list[date]]:
    """종목별 {날짜: (o,h,lo,c,v,vwap,거래대금,체결수)} → (panel, symbols, dates). 공통 거래일 inner-join.

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
            o, h, lo, c, v, vw, dv, nt = per_symbol[symbol][day]
            panel["open"][i, j] = o
            panel["high"][i, j] = h
            panel["low"][i, j] = lo
            panel["close"][i, j] = c
            panel["volume"][i, j] = v
            panel["vwap"][i, j] = vw
            panel["dollar_volume"][i, j] = dv
            panel["trade_count"][i, j] = nt
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
    """메인넷 공개 klines → (panel, symbols, dates). 공통 날짜만 정렬(inner join).

    fetch_ohlcv 는 6컬럼(OHLCV)만 돌려줘 거래대금·체결수를 버린다. 유동성 신호를
    위해 원시 klines(12컬럼)를 직접 조회 — quote(USDT) 거래대금[7]·체결 건수[8]를
    확보한다. VWAP 은 거래대금/거래량으로 근사(klines 는 VWAP 직접 미제공).
    """
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
            market_id = symbol.replace("/", "")  # BTC/USDT → BTCUSDT (원시 klines 심볼)
            rows: dict[date, tuple] = {}
            cursor = since
            while True:
                raw = await ex.publicGetKlines(
                    {"symbol": market_id, "interval": "1d", "startTime": cursor, "limit": 1000}
                )
                if not raw:
                    break
                for k in raw:
                    day = datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).date()
                    if day <= end:
                        o, h, lo, c, v = (float(k[i]) for i in range(1, 6))
                        quote_vol, n_trades = float(k[7]), float(k[8])
                        vwap = quote_vol / v if v > 0 else c
                        rows[day] = (o, h, lo, c, v, vwap, quote_vol, n_trades)
                if len(raw) < 1000:
                    break
                cursor = int(raw[-1][0]) + 1
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
    """Alpaca 통합 피드 일봉(분할조정) → (panel, symbols, dates). 상한 t-1(당일 봉 제외).

    봉 스키마 t/o/h/l/c/v/n/vw 중 vw(VWAP)·n(체결수)까지 캡처 — 유동성 신호용.
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
                    # 단일 거래소(iex) 거래량은 통합의 3% 남짓이라 유동성 신호가
                    # 표본 잡음 위에 올라간다. 통합 피드도 end 가 15분 이상 과거면
                    # 무료 플랜 권한 안이고, 이 창은 상한이 t-1 이라 항상 해당한다.
                    "feed": "sip",
                    # 2년 창에는 분할이 들어올 수 있다. 기본값(raw)이면 그 자리에
                    # -90% 짜리 가짜 수익률이 생겨 IC 가 통째로 오염된다.
                    "adjustment": "split",
                    "limit": 10000,
                },
            )
            resp.raise_for_status()
            rows: dict[date, tuple] = {}
            for rb in (resp.json().get("bars") or {}).get(symbol, []):
                day = datetime.fromisoformat(rb["t"].replace("Z", "+00:00")).date()
                if day <= end:  # 상한 재확인
                    o, h, lo, c, v = rb["o"], rb["h"], rb["l"], rb["c"], rb["v"]
                    vwap = rb.get("vw", c)  # Alpaca 제공 VWAP (없으면 종가 대체)
                    # Alpaca 는 quote 거래대금 미제공 → VWAP×거래량으로 근사
                    rows[day] = (o, h, lo, c, v, vwap, vwap * v, float(rb.get("n", 0)))
            per_symbol[symbol] = rows
    finally:
        if own_client:
            await client.aclose()

    return _assemble_panel(per_symbol, symbols)


#: 시장 → 연구 유니버스. 팩터가 **무엇을 근거로 승격됐는지**를 정의하는 집합이라,
#: 스코어링 횡단면을 넓히더라도 이 값은 그대로 둔다(승격 근거와 주입 값의 출처 구분).
RESEARCH_UNIVERSE: dict[str, list[str]] = {
    "CRYPTO": CRYPTO_RESEARCH_UNIVERSE,
    "US": US_RESEARCH_UNIVERSE,
}


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
