"""시장 간 배분 백테스트용 지수 프록시 장기 일봉 수집 — `data/research/meta_hist/`.

라이브 국면 계산과 **같은 프록시·같은 어댑터 경로**(`_fetch_bars_history`)를 쓴다.
프록시가 다르면 백테스트가 라이브 shadow 와 다른 것을 재는 셈이 된다.
    CRYPTO = BTC/USDT (Binance 공개 시세)  US = QQQ (KIS 해외)  KR = 069500 (KIS 국내)

KIS 는 1회 응답 ~100행이라 페이지네이션 상한이 운영 경로 기준(~1,200거래일)으로 잡혀
있다. 여기서는 연구 1회 수집이므로 상한만 넉넉히 올린다 — 운영 코드는 건드리지 않는다.

사용: .venv/bin/python scripts/fetch_meta_history.py [--start 2010-01-01]
출력: data/research/meta_hist/{market}.json  {"market","symbol","source","fetched_at","bars":[...]}
      verify 줄: market=… n=… first=… last=… max_gap_days=…
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import adapters.kis as kis_module  # noqa: E402
from harness.env import load_env  # noqa: E402
from meta.feeds import kis_credentials  # noqa: E402
from regime.live import INDEX_PROXY  # noqa: E402

STATE_DIR = ROOT / "data" / "state"
OUT_DIR = ROOT / "data" / "research" / "meta_hist"
RESEARCH_MAX_PAGES = 80  # ~8,000거래일 — 2010년 이후 전 구간을 덮고도 남는다


def _max_gap_days(days: list[date]) -> int:
    return max(((b - a).days for a, b in zip(days, days[1:])), default=0)


def _write(market: str, symbol: str, source: str, bars: list) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(b), "day": b.day.isoformat()} for b in bars]
    (OUT_DIR / f"{market}.json").write_text(
        json.dumps(
            {
                "market": market,
                "symbol": symbol,
                "source": source,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "bars": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    days = [b.day for b in bars]
    return {
        "market": market,
        "symbol": symbol,
        "n": len(bars),
        "first": days[0].isoformat() if days else None,
        "last": days[-1].isoformat() if days else None,
        "max_gap_days": _max_gap_days(days),
    }


async def main(start: date, end: date) -> int:
    env = load_env(ROOT / ".env")
    kis_module.HISTORY_MAX_PAGES = RESEARCH_MAX_PAGES  # 연구 1회 수집에만 적용
    results: list[dict] = []

    from adapters.ccxt_adapter import BinanceDataFeed

    feed = BinanceDataFeed()
    try:
        bars = await feed._fetch_bars_history([INDEX_PROXY["CRYPTO"]], start, end)
        results.append(
            _write("CRYPTO", INDEX_PROXY["CRYPTO"], "binance", bars[INDEX_PROXY["CRYPTO"]])
        )
    finally:
        await feed.close_data()

    creds = kis_credentials(env, STATE_DIR)
    if creds is None:
        print("status=skip market=KR,US reason=no_kis_credentials")
    else:
        app_key, app_secret, account, token_cache, mode = creds
        from adapters.kis import KISDomesticAdapter
        from adapters.kis_overseas import KISOverseasAdapter

        kr = KISDomesticAdapter(
            app_key, app_secret, account, [INDEX_PROXY["KR"]], token_cache=token_cache, mode=mode
        )
        try:
            bars = await kr._fetch_bars_history([INDEX_PROXY["KR"]], start, end)
            results.append(_write("KR", INDEX_PROXY["KR"], f"kis:{mode}", bars[INDEX_PROXY["KR"]]))
        finally:
            await kr.close()
        us = KISOverseasAdapter(
            app_key, app_secret, account, [INDEX_PROXY["US"]], token_cache=token_cache, mode=mode
        )
        try:
            bars = await us._fetch_bars_history([INDEX_PROXY["US"]], start, end)
            results.append(_write("US", INDEX_PROXY["US"], f"kis:{mode}", bars[INDEX_PROXY["US"]]))
        finally:
            await us.close()

    for r in results:
        print(" ".join(f"{k}={v}" for k, v in r.items()))
    return 0 if all(r["n"] > 0 for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    args = parser.parse_args()
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    raise SystemExit(asyncio.run(main(args.start, yesterday)))
