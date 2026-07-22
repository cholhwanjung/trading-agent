"""라이브 국면 계산 — 어댑터로 지수 프록시 봉을 조회해 classify_regime 에 넘긴다.

별도 지수 API 없이 기존 어댑터의 심볼 조회만 사용(`get_ohlcv_history`, 상한 t−1 =
[ADR-013] 누출 통제 유지). 지수 프록시:
    CRYPTO = BTC/USDT (대장), US = SPY, KR = 069500 (KODEX 200).
"""

from __future__ import annotations

from datetime import date

from regime.pulse import RegimeResult, classify_regime

INDEX_PROXY = {"CRYPTO": "BTC/USDT", "US": "SPY", "KR": "069500"}
LOOKBACK_DAYS = 300  # DD 25세션 윈도우 + CORRECTION 추적 여유


async def compute_regime(adapter, market: str, asof_day: date) -> RegimeResult | None:
    """지수 프록시 국면. 프록시 없음/조회 실패 → None (fail-open — 관측 보조일 뿐)."""
    proxy = INDEX_PROXY.get(market)
    if not proxy:
        return None
    try:
        bars = await adapter.get_ohlcv_history([proxy], asof_day, lookback_days=LOOKBACK_DAYS)
        return classify_regime(bars.get(proxy, []))
    except Exception:
        return None
