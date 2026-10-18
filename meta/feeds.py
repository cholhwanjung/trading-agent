"""지수 프록시 일봉 피드 — 메타 배분의 시장별 집계 신호 입력.

라이브 국면 계산과 **같은 프록시**(`regime.live.INDEX_PROXY`)를 같은 어댑터 경로
(`_fetch_bars_history`, 페이지네이션 포함)로 읽는다. 프록시가 다르면 메타 층이 트레이더와
다른 시장을 보는 셈이 된다. KIS 시세 조회에는 모의 키를 우선 쓴다 — 실전 토큰 캐시를
건드리지 않기 위해서다. 키가 없는 시장은 빈 결과로 빠지고 호출부가 그 시장을 제외한다.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from adapters.base import Bar
from regime.live import INDEX_PROXY

LOOKBACK_DAYS = 420  # 12개월 모멘텀(252거래일) + 국면 창(300캘린더일) 여유


def kis_credentials(env: dict[str, str], state_dir: Path) -> tuple[str, str, str, Path, str] | None:
    """시세 조회용 KIS 자격 (app_key, app_secret, account, token_cache, mode). 없으면 None."""
    if env.get("KIS_PAPER_APP_KEY") and "-" in env.get("KIS_PAPER_ACCOUNT", ""):
        return (
            env["KIS_PAPER_APP_KEY"],
            env["KIS_PAPER_APP_SECRET"],
            env["KIS_PAPER_ACCOUNT"],
            state_dir / "kis_token.json",
            "demo",
        )
    if env.get("KIS_REAL_APP_KEY") and "-" in env.get("KIS_REAL_ACCOUNT", ""):
        return (
            env["KIS_REAL_APP_KEY"],
            env["KIS_REAL_APP_SECRET"],
            env["KIS_REAL_ACCOUNT"],
            state_dir / "kis_real_token.json",
            "real",
        )
    return None


async def fetch_proxy_bars(
    env: dict[str, str],
    state_dir: Path,
    asof_day: date,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, list[Bar]]:
    """시장 → 프록시 일봉 [asof−lookback, asof−1]. 상한 t−1 은 end 로 강제한다."""
    end = asof_day - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    out: dict[str, list[Bar]] = {}

    from adapters.ccxt_adapter import BinanceDataFeed

    feed = BinanceDataFeed()
    try:
        proxy = INDEX_PROXY["CRYPTO"]
        out["CRYPTO"] = (await feed._fetch_bars_history([proxy], start, end)).get(proxy, [])
    finally:
        await feed.close_data()

    creds = kis_credentials(env, state_dir)
    if creds is None:
        return out
    app_key, app_secret, account, token_cache, mode = creds
    from adapters.kis import KISDomesticAdapter
    from adapters.kis_overseas import KISOverseasAdapter

    for market, cls in (("KR", KISDomesticAdapter), ("US", KISOverseasAdapter)):
        proxy = INDEX_PROXY[market]
        adapter = cls(app_key, app_secret, account, [proxy], token_cache=token_cache, mode=mode)
        try:
            out[market] = (await adapter._fetch_bars_history([proxy], start, end)).get(proxy, [])
        finally:
            await adapter.close()
    return out
