"""실시간 이벤트 트리거 워커 — 스케줄 밖 급변 감지·재결정 (단계 1, [ADR-021]).

주기 check-once: launchd StartInterval 이 이 스크립트를 15분마다 실행한다(상주 데몬
아님 — 무상태·재시작안전). 현재가를 조회해 직전 참조가 대비 급변이면 트리거를 발동,
기존 RiskGuardedPolicy(LLMTrader) 결정 경로를 당일 컨텍스트와 함께 호출하고 주문한다.

**학습 제외**: 트리거 결정은 메모리 파이프라인(record/promote/probation/outcome/
reflection)을 호출하지 않는다 — 당일 정보 기반 결정을 admission 에 넣으면 leakage 오염
(하드룰 7 · [ADR-013]). 승격 교훈이 0인 v1 은 memory_fn 도 생략(방어 반응 우선).

유니버스·리스크 한도·어댑터 구성은 run_paper_step 에서 import — 단일 출처 유지.

사용법:
    uv run python scripts/run_watcher.py                 # CRYPTO 1회 점검(+발동 시 주문)
    uv run python scripts/run_watcher.py --market CRYPTO
    uv run python scripts/run_watcher.py --dry-run       # 조회·판정만(주문·상태저장 없음)
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import JsonlLogger, load_env  # noqa: E402
from llm import LLMRouter  # noqa: E402
from risk import RiskEngine, RiskGuardedPolicy  # noqa: E402
from scripts.run_paper_step import (  # noqa: E402
    LIMITS,
    STATE_DIR,
    build_adapters,
    load_prev_weights,
)
from trader import LLMTrader  # noqa: E402
from watcher import config_for, evaluate, max_drift  # noqa: E402


def _arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def _load_watch_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_watch_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


async def _close(adapter) -> None:
    close = getattr(adapter, "close", None)
    if close:
        await close()


async def main() -> int:
    market = _arg("--market", "CRYPTO").upper()
    dry_run = "--dry-run" in sys.argv
    config = config_for(market)  # 미지원 시장은 KeyError (v1 은 CRYPTO 전용)

    env = load_env(ROOT / ".env")
    adapters = build_adapters(env)
    if market not in adapters:
        for a, _ in adapters.values():
            await _close(a)
        print(f"status=skip detail={market} 어댑터 키 없음/형식 오류")
        return 0
    adapter, symbols = adapters[market]
    for m, (a, _) in adapters.items():
        if m != market:
            await _close(a)

    logger = JsonlLogger(ROOT / "data" / "logs")
    watch_path = STATE_DIR / f"watch_{market}.json"
    router = LLMRouter(env)
    try:
        now = datetime.now(timezone.utc)
        current = await adapter.get_current_prices(symbols)
        state = _load_watch_state(watch_path)
        trigger, new_state = evaluate(state, current, now, config)

        if trigger is None:
            drift = max_drift(current, state.get("ref") or current)
            print(
                f"market={market} status=no_trigger drift={drift:.4f}"
                f" threshold={config.move_threshold} prices={current}"
            )
            if not dry_run:
                _save_watch_state(watch_path, new_state)
            return 0

        logger.log(market, "realtime_trigger", trigger)
        print(
            f"market={market} status=triggered worst={trigger['worst_symbol']}"
            f" move={trigger['worst_move']} dry_run={int(dry_run)}"
        )
        if dry_run:
            print("dry_run=1 detail=주문·상태저장 생략")
            return 0

        # 결정 경로 재사용 — 당일 급변을 trigger 채널로 주입, Risk Engine 동일 게이팅.
        # memory_fn 생략(v1): 승격 교훈 0 + 방어 반응 우선. signals_fn 은 일간과 동일.
        signals_fn = None
        if market == "CRYPTO":
            from alpha_lab.signals import compute_alpha_signals

            async def signals_fn(obs, _syms=symbols):
                return await compute_alpha_signals(
                    STATE_DIR / "alpha_library_CRYPTO.json", _syms, obs.asof_day
                )

        risk_path = STATE_DIR / f"risk_{market}.json"
        trader = LLMTrader(
            router, market, symbols, adapter.get_ohlcv_history,
            signals_fn=signals_fn,
            prev_weights_fn=lambda p=risk_path: load_prev_weights(p),
        )
        guard = RiskGuardedPolicy(
            trader, RiskEngine(LIMITS[market]), risk_path, equity_fn=adapter.get_equity
        )
        obs = await adapter.observe_and_audit(symbols)  # [t-3,t-1] 누출 감사 (행동 컨텍스트)
        positions = await adapter.get_positions()
        weights = await guard.decide(obs, positions, trigger=trigger)
        result = await adapter.submit_allocation(weights)

        meta = guard.last_decision or {}
        logger.log(
            market,
            "realtime_action",
            {
                "weights": weights,
                "accepted": result.accepted,
                "n_orders": len(result.orders),
                "orders": result.orders,
                "risk_violations": meta.get("risk_violations", []),
                "rationale": meta.get("rationale", ""),
                "error": result.error,
            },
        )
        print(
            f"market={market} status={'ok' if result.accepted else 'rejected'}"
            f" n_orders={len(result.orders)} weights={weights}"
        )
        _save_watch_state(watch_path, new_state)
        return 0 if result.accepted else 1
    finally:
        await _close(adapter)
        await router.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
