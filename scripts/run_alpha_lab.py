"""Alpha Lab 1사이클 — writer→judge→백테스트→admission (시장별).

사용법:
    uv run python scripts/run_alpha_lab.py [--n 5] [--markets CRYPTO,US]

- 시장: CRYPTO(연구 유니버스 10종) + US(Alpaca 키 있을 때, 12종). 시장별 격리 라이브러리.
- 라이브러리: data/state/alpha_library_<MARKET>.json (경험 메모리 포함)
- 백테스트는 스크리닝 전용 — 라이브 기여는 Trader 인용→reflection 으로 계측.
- 개선 곡선 포화(QuantaAlpha ~11-12 iteration decay) 모니터링을 위해 사이클 이벤트를
  data/logs/ALPHA/ 에 구조화 기록한다.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alpha_lab import FactorLibrary, generate_candidates  # noqa: E402
from alpha_lab.data import fetch_crypto_panel, make_us_panel_fn  # noqa: E402
from harness import (  # noqa: E402
    JsonlLogger,
    load_env,
    make_usage_sink,
    single_instance,
    wait_for_network,
    with_deadline,
)
from llm import LLMRouter  # noqa: E402

STATE_DIR = ROOT / "data" / "state"


def _markets(env: dict[str, str]) -> dict[str, tuple]:
    """시장 → (panel_fn, asset_label). US 는 Alpaca 키 있을 때만 편입."""
    out: dict[str, tuple] = {"CRYPTO": (fetch_crypto_panel, "크립토")}
    us_panel_fn = make_us_panel_fn(env)
    if us_panel_fn is not None:
        out["US"] = (us_panel_fn, "미국 주식")
    return out


async def run_cycle(
    market: str, panel_fn, asset_label: str,
    router: LLMRouter, logger: JsonlLogger, n: int, today: date,
) -> int:
    """한 시장의 1사이클. 승격 팩터 수 반환."""
    library = FactorLibrary(STATE_DIR / f"alpha_library_{market}.json")
    print(f"[{market}] library: active={len(library.active())} total={len(library.factors)}")
    panel, symbols, dates = await panel_fn()
    print(f"[{market}] panel: {len(dates)}일 × {len(symbols)}종 ({dates[0]} ~ {dates[-1]})")

    # 라이브 감쇠 퇴출 먼저 — writer 가 살아있는 라이브러리만 보게
    decayed = library.review_decay(panel, dates, today)
    for e in decayed:
        print(f"  [{market}] DECAYED name={e['name']} live_ic={e['live_ic']}"
              f" (admission oos_ic={e['admission_oos_ic']})")
        logger.log("ALPHA", e.pop("event"), e | {"market": market, "cycle_day": str(today)})
    print(f"[{market}] decay_review retired={len(decayed)} active_remaining={len(library.active())}")

    candidates = await generate_candidates(router, library, n=n, asset_label=asset_label)
    print(f"[{market}] writer: {len(candidates)}개 생성, judge/dsl 기각 "
          f"{sum(1 for c in candidates if c.rejected)}개")
    for c in candidates:
        if c.rejected:
            print(f"  [{market}] pre-reject name={c.name} reason={c.rejected[:80]}")

    events = library.admit(candidates, panel, today)
    admitted = [e for e in events if e["event"] == "factor_admitted"]
    for e in events:
        logger.log("ALPHA", e.pop("event"), e | {"market": market, "cycle_day": str(today)})
    for e in admitted:
        print(f"  [{market}] ADMITTED name={e['name']} train_ic={e['train_ic']}"
              f" oos_ic={e['oos_ic']} oos_icir={e['oos_icir']}")
    for e in events:
        if "reason" in e:
            print(f"  [{market}] rejected name={e['name']} reason={e['reason'][:80]}")

    print(f"[{market}] cycle_done admitted={len(admitted)} library_active={len(library.active())}")
    return len(admitted)


_VALID_MARKETS = {"CRYPTO", "US"}


def _market_set(value: str) -> set[str]:
    """쉼표 구분 시장 목록 → 대문자 집합. 알 수 없는 시장은 즉시 거부."""
    markets = {m.strip().upper() for m in value.split(",") if m.strip()}
    if not markets:
        raise argparse.ArgumentTypeError("빈 시장 목록")
    invalid = markets - _VALID_MARKETS
    if invalid:
        raise argparse.ArgumentTypeError(
            f"알 수 없는 시장: {sorted(invalid)} (가능: {sorted(_VALID_MARKETS)})"
        )
    return markets


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha Lab 1사이클 — writer→judge→백테스트→admission")
    p.add_argument("--n", type=int, default=5, help="사이클당 후보 팩터 수 (기본 5)")
    p.add_argument("--markets", type=_market_set, help="쉼표 구분 시장 (예: CRYPTO,US)")
    return p.parse_args()


async def main() -> int:
    args = _parse_args()
    n = args.n
    env = load_env(ROOT / ".env")

    markets = _markets(env)
    if args.markets:
        markets = {k: v for k, v in markets.items() if k in args.markets}
    if not markets:
        print("status=fail detail=실행할 시장 없음(키/인자 확인)")
        return 1

    # 단일 인스턴스 락 — 주간 잡이라 겹칠 일은 드물지만, 락 파일이 실행중 표식을 겸해
    # 대시보드가 "지금 alpha 사이클이 돌고 있음"을 알 수 있게 한다(수 분~수십 분 걸리는 잡).
    lock = single_instance(STATE_DIR / "run_alpha_lab.lock", label="alpha 사이클")
    if lock is None:
        print("status=skip detail=이미 실행 중 — 중복 실행 차단")
        return 0

    # wake/부팅 직후 실행 시 네트워크(DNS+TCP) 준비 대기 — LLM·패널 수집 조기 실패 방지.
    if not await wait_for_network():
        print("status=fail event=network_unavailable detail=네트워크 게이트 타임아웃(10분)")
        return 1

    router = LLMRouter(env, usage_sink=make_usage_sink(ROOT))
    logger = JsonlLogger(ROOT / "data" / "logs")
    today = datetime.now(timezone.utc).date()
    try:
        for market, (panel_fn, asset_label) in markets.items():
            try:
                await run_cycle(market, panel_fn, asset_label, router, logger, n, today)
            except Exception as e:  # 시장별 실패 격리 — 한 시장 실패가 다른 시장을 막지 않음
                print(f"[{market}] status=error error={type(e).__name__}: {str(e)[:200]}")
        return 0
    finally:
        await router.close()


if __name__ == "__main__":
    # 런 전체 데드라인 — 느린 reasoning 응답 누적(07-20 ReadTimeout×3 유형) 총량 차단.
    sys.exit(asyncio.run(with_deadline(main(), label="alpha_lab")))
