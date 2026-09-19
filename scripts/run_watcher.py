"""실시간 이벤트 트리거 워커 — 스케줄 밖 급변 감지·재결정 (단계 1).

주기 check-once: launchd StartInterval 이 이 스크립트를 15분마다 실행한다(상주 데몬
아님 — 무상태·재시작안전). 현재가를 조회해 직전 참조가 대비 급변이면 트리거를 발동,
기존 RiskGuardedPolicy(LLMTrader) 결정 경로를 당일 컨텍스트와 함께 호출하고 주문한다.

**학습 제외**: 트리거 결정은 메모리 파이프라인(record/promote/probation/outcome/
reflection)을 호출하지 않는다 — 당일 정보 기반 결정을 admission 에 넣으면 leakage 오염.
교훈 주입도 하지 않는다(일간 결정에서 배운 것을 당일 급변이라는 다른 조건에 적용할
근거가 없다).

유니버스·어댑터·정책 조립·집행 게이트는 일간 스텝의 것을 그대로 쓴다 — 주문으로 가는
길이 둘이면 한쪽에만 게이트가 붙는다.

사용법:
    uv run python scripts/run_watcher.py                 # CRYPTO 1회 점검(+발동 시 주문)
    uv run python scripts/run_watcher.py --market KR     # KR 은 장중(09:00~15:30 KST)만 동작
    uv run python scripts/run_watcher.py --dry-run       # 조회·판정만(주문·상태저장 없음)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters import configure_observation  # noqa: E402
from harness import (  # noqa: E402
    JsonlLogger,
    decide_and_submit,
    load_env,
    make_usage_sink,
    market_locks,
    notify,
)
from llm import LLMRouter  # noqa: E402
from scripts.run_paper_step import (  # noqa: E402
    STATE_DIR,
    TRADABLE,
    build_adapters,
    build_market_policy,
    close_unselected,
    log_ledger_activity,
)
from watcher import config_for, evaluate, in_session, max_drift  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실시간 이벤트 트리거 워커 — 급변 감지·재결정")
    p.add_argument("--market", default="CRYPTO", type=str.upper,
                   choices=["KR", "US", "CRYPTO"], help="점검 시장 (기본 CRYPTO)")
    p.add_argument("--dry-run", action="store_true", help="조회·판정만 (주문·상태저장 없음)")
    return p.parse_args()


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
    args = _parse_args()
    market = args.market
    dry_run = args.dry_run
    config = config_for(market)  # 미지원 시장은 KeyError (지원 목록은 watcher.DEFAULTS)

    # 장외엔 아무것도 하지 않는다 — 시장가 미체결 시간대의 트리거·주문 차단(15분 틱 조기 종료).
    if not in_session(config, datetime.now(timezone.utc)):
        print(f"market={market} status=closed detail=장외 게이팅 스킵")
        return 0

    env = load_env(ROOT / ".env")
    configure_observation(env)  # 관측 윈도우 길이 .env 오버라이드(실험 변수, 미설정 시 기본)

    # 계좌 락 — 15분 인터벌 틱이 이전(느린) 틱과 겹치거나, **일일 스텝과 겹쳐** 같은 계좌에
    # 중복 주문하고 risk_{market}.json 을 레이스로 덮어쓰는 것을 차단. 키가 시장이라
    # 페이퍼 스텝과 같은 락을 두고 경합한다(먼저 잡은 쪽이 끝날 때까지 다른 쪽은 스킵).
    locks = market_locks(STATE_DIR, [market], label="워처")
    if locks is None:
        print(f"status=skip detail=이미 실행 중(market={market}) — 중복 실행 차단")
        return 0

    adapters = build_adapters(env)
    if market not in adapters:
        for a, _ in adapters.values():
            await _close(a)
        print(f"status=skip detail={market} 어댑터 키 없음/형식 오류")
        return 0
    adapter, symbols = adapters[market]
    # 계좌를 나눠 쓰는 상대 시장은 닫지 않는다 — 총자산 계산에 그쪽 잔고가 필요하다.
    peer_adapters = await close_unselected(adapters, {market})

    logger = JsonlLogger(ROOT / "data" / "logs")
    watch_path = STATE_DIR / f"watch_{market}.json"
    router = LLMRouter(env, usage_sink=make_usage_sink(ROOT))
    try:
        now = datetime.now(timezone.utc)
        # 급변은 **매매할 수 있는 종목**으로만 판정한다. 관측 전용 종목까지 보면 계좌가
        # 들 수 없는 종목 하나의 변동성이 재결정과 주문을 끌고 다닌다 — 움직인 건 그 종목인데
        # 사고파는 건 지수 ETF 다. 관측 전용 종목은 결정의 근거로만 남는다(t-1 관측).
        current = await adapter.get_current_prices(TRADABLE[market])
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

        # 일간 스텝과 같은 정책 조립·같은 집행 게이트 — 당일 급변만 trigger 채널로 더한다.
        guard = build_market_policy(market, adapter, symbols, router, None, env, "auto")
        obs = await adapter.observe_and_audit(symbols)  # 상한 t-1 누출 감사 (행동 컨텍스트)
        weights, _, result, venue_error = await decide_and_submit(
            adapter, guard, obs, trigger=trigger
        )

        meta = guard.last_decision or {}
        logger.log(
            market,
            "realtime_action",
            {
                "weights": weights,
                "execution_mode": "degraded" if venue_error else "live",
                "accepted": result.accepted,
                "n_orders": len(result.orders),
                "orders": result.orders,
                "risk_violations": meta.get("risk_violations", []),
                "circuit_open": meta.get("circuit_open"),
                "mdd": meta.get("mdd"),
                "rationale": meta.get("rationale", ""),
                "error": result.error,
            },
        )
        status = "ok" if result.accepted else ("degraded" if venue_error else "rejected")
        print(
            f"market={market} status={status} n_orders={len(result.orders)}"
            f" mdd={meta.get('mdd')} weights={weights}"
            + (f" error={result.error}" if result.error else "")
        )
        # 실자금 계좌만 통지 — 일간 스텝과 같은 기준(주문 거부·집행 스킵·낙폭 서킷).
        if getattr(adapter, "mode", None) == "real":
            if not result.accepted:
                await notify(env, f"{market} 워처 주문 실패", result.error or "accepted=False")
            elif meta.get("circuit_open"):
                await notify(env, f"{market} 실계좌 MDD 서킷", f"mdd={meta.get('mdd')} (워처)")
        log_ledger_activity(logger, market, adapter)
        _save_watch_state(watch_path, new_state)
        return 0 if result.accepted else 1
    finally:
        for a in [adapter, *peer_adapters]:
            await _close(a)
        await router.close()
        for lock in locks:  # 락 해제 (프로세스 종료로도 커널이 해제하나 즉시 반납)
            lock.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
