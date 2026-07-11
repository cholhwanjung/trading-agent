"""일일 페이퍼 스텝 — 실계좌(Risk-guarded LLM) + 가상 포트폴리오 병행 운용 (Phase 1).

사용법:
    uv run python scripts/run_paper_step.py            # 실주문 + 가상 A/B
    uv run python scripts/run_paper_step.py --dry-run  # 관측·포지션 조회까지만

구성 (시장별):
- 실계좌: RiskGuardedPolicy(LLMTrader) — 라이브 페이퍼가 진실 verifier ([ADR-001]).
- 가상 3종(llm/bh/random): 동일 관측·t-1 종가 forward 시뮬레이션 → 델타 측정 기준선.
  가상 llm 은 실계좌와 같은 목표 배분을 사용(추가 LLM 호출 없음).

로그: data/logs/{market}/{date}.jsonl · 가상 상태: data/state/virtual/*.json
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import VirtualPortfolio  # noqa: E402
from harness import BuyAndHold, JsonlLogger, MarketRun, RandomPolicy, run_all_markets  # noqa: E402
from llm import LLMRouter  # noqa: E402
from memory import (  # noqa: E402
    MemoryStore,
    fill_pending_outcomes,
    lessons_payload,
    promote_candidates,
    record_decision,
    retrieve,
    review_probation,
    review_retention,
)
from risk import RiskEngine, RiskGuardedPolicy, RiskLimits  # noqa: E402
from trader import LLMTrader  # noqa: E402

# 시장별 유니버스 + Risk limits 초기 캘리브레이션 (좁은 유니버스라 종목당 상한 완화)
CRYPTO_UNIVERSE = ["BTC/USDT", "ETH/USDT"]
US_UNIVERSE = ["SPY"]
LIMITS = {
    "CRYPTO": RiskLimits(max_weight_per_asset=0.50, min_cash=0.10, max_daily_turnover=0.50, mdd_circuit=0.20),
    "US": RiskLimits(max_weight_per_asset=0.85, min_cash=0.10, max_daily_turnover=0.50, mdd_circuit=0.15),
}
STATE_DIR = ROOT / "data" / "state"
COST_BPS = {"CRYPTO": 10.0, "US": 1.0}  # 가상 포트폴리오 거래비용


def load_env(path: Path) -> dict[str, str]:
    """의존성 없는 .env 파서 (scripts/check_credentials.py 와 동일 규칙)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def build_adapters(env: dict[str, str]) -> dict[str, tuple[object, list[str]]]:
    out: dict[str, tuple[object, list[str]]] = {}
    if env.get("BINANCE_TESTNET_API_KEY") and env.get("BINANCE_TESTNET_SECRET"):
        from adapters.ccxt_adapter import BinanceTestnetAdapter

        out["CRYPTO"] = (
            BinanceTestnetAdapter(
                env["BINANCE_TESTNET_API_KEY"], env["BINANCE_TESTNET_SECRET"], CRYPTO_UNIVERSE
            ),
            CRYPTO_UNIVERSE,
        )
    if env.get("ALPACA_PAPER_API_KEY") and env.get("ALPACA_PAPER_SECRET"):
        from adapters.alpaca import AlpacaPaperAdapter

        out["US"] = (
            AlpacaPaperAdapter(env["ALPACA_PAPER_API_KEY"], env["ALPACA_PAPER_SECRET"], US_UNIVERSE),
            US_UNIVERSE,
        )
    return out


async def fetch_prices(adapter, symbols: list[str]):
    """t-1 종가 + 기준일 — 가상 마킹과 메모리 outcome 계측이 공유."""
    today = datetime.now(timezone.utc).date()
    bars = await adapter.get_ohlcv(symbols, today)
    prices = {s: b[-1].close for s, b in bars.items() if b}
    day = max((b[-1].day for b in bars.values() if b), default=None)
    return prices, day


async def run_virtual(
    market: str,
    symbols: list[str],
    prices: dict,
    day,
    llm_weights: dict | None,
    llm_base_weights: dict | None,
    logger: JsonlLogger,
) -> None:
    """가상 4종 스텝 — t-1 종가로 마킹.

    llm = 메모리 블렌딩 최종 배분(실계좌와 동일) / llm_base = 무메모리 base 배분.
    두 arm 의 equity 델타가 Phase 2 ablation 의 측정치다 (No-Memory vs +Memory).
    """
    if len(prices) < len(symbols) or day is None:
        logger.log(market, "virtual_skip", {"reason": "missing_prices", "have": list(prices)})
        return

    obs = None  # baseline 정책은 관측을 쓰지 않는다 (B&H 고정 / 랜덤)
    policies: dict[str, dict | None] = {
        "llm": llm_weights,
        "llm_base": llm_base_weights,
        "bh": await BuyAndHold(symbols).decide(obs, []),
        "random": await RandomPolicy(symbols, seed=int(day.strftime("%Y%m%d"))).decide(obs, []),
    }
    for name, weights in policies.items():
        if weights is None:
            continue
        portfolio = VirtualPortfolio(STATE_DIR / "virtual" / f"{market}_{name}.json")
        equity = portfolio.step(day, prices, weights, cost_bps=COST_BPS[market])
        logger.log(
            market,
            "virtual_step",
            {"portfolio": name, "day": str(day), "equity": round(equity, 2), "weights": weights},
        )
        print(f"market={market} virtual={name} equity={equity:,.2f}")


async def main() -> int:
    env = load_env(ROOT / ".env")
    adapters = build_adapters(env)
    if not adapters:
        print("status=fail detail=.env에 사용 가능한 브로커 키 없음 (docs/CREDENTIALS.md)")
        return 1

    router = LLMRouter(env)
    logger = JsonlLogger(ROOT / "data" / "logs")
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if "--dry-run" in sys.argv:
            for market, (adapter, symbols) in adapters.items():
                obs = await adapter.observe_and_audit(symbols)
                positions = await adapter.get_positions()
                print(
                    f"market={market} status=observed asof={obs.asof_day}"
                    f" n_bars={ {s: len(b) for s, b in obs.bars.items()} } n_news={len(obs.news)}"
                    f" positions={[(p.symbol, round(p.market_value, 2)) for p in positions]}"
                )
            return 0

        memory = MemoryStore(ROOT / "data" / "memory.sqlite")

        # 실계좌: Risk-guarded LLM (+ 검증 통과 교훈만 주입 — 하드룰 3)
        runs = []
        guards: dict[str, RiskGuardedPolicy] = {}
        prev_weights_by_market: dict[str, dict | None] = {}
        for market, (adapter, symbols) in adapters.items():
            prev_weights_by_market[market] = json_safe_weights_path(
                STATE_DIR / f"risk_{market}.json"
            )

            def make_memory_fn(m: str):
                async def memory_fn(obs) -> list[dict]:
                    hits = retrieve(memory, m, obs.asof_day, k=5)
                    return lessons_payload(hits)  # confidence 포함 (R9 블렌딩 입력)

                return memory_fn

            def make_forbidden_fn(m: str):
                def forbidden_patterns() -> set[str]:
                    return {
                        e.pattern_key
                        for e in memory.query(m, store="procedural", status="active")
                        if e.pattern_key and e.data.get("kind") == "forbidden"
                    }

                return forbidden_patterns

            signals_fn = None
            if market == "CRYPTO":  # 연구 유니버스가 크립토뿐 (ADR-017), US 는 확장 후
                from alpha_lab.signals import compute_alpha_signals

                async def signals_fn(obs, _syms=symbols):
                    return await compute_alpha_signals(
                        STATE_DIR / "alpha_library_CRYPTO.json", _syms, obs.asof_day
                    )

            trader = LLMTrader(
                router, market, symbols, adapter.get_ohlcv_history,
                memory_fn=make_memory_fn(market),
                signals_fn=signals_fn,
            )
            guard = RiskGuardedPolicy(
                trader,
                RiskEngine(LIMITS[market]),
                STATE_DIR / f"risk_{market}.json",
                equity_fn=adapter.get_equity,
                forbidden_patterns_fn=make_forbidden_fn(market),
            )
            guards[market] = guard
            runs.append(MarketRun(adapter, guard, symbols))

        results = await run_all_markets(runs, logger)
        exit_code = 0
        for market, outcome in results.items():
            if isinstance(outcome, Exception):
                print(f"market={market} status=error error={type(outcome).__name__}: {outcome}")
                exit_code = 1
            else:
                meta = guards[market].last_decision or {}
                print(
                    f"market={market} status={'ok' if outcome.accepted else 'rejected'}"
                    f" n_orders={len(outcome.orders)}"
                    f" violations={meta.get('risk_violations', [])}"
                    f" mdd={meta.get('mdd')}"
                    + (f" error={outcome.error}" if outcome.error else "")
                )
                if not outcome.accepted:
                    exit_code = 1

        # 가상 병행 운용 + 메모리 파이프라인 (실스텝 성공 여부와 무관하게 baseline 은 쌓인다)
        today = datetime.now(timezone.utc).date()
        for market, (adapter, symbols) in adapters.items():
            guard = guards[market]
            llm_weights = None
            llm_base_weights = None
            if not isinstance(results.get(market), Exception) and guard.last_decision:
                llm_weights = json_safe_weights_path(guard.state_path)
                llm_base_weights = getattr(guard.inner, "last_base_weights", None)

            prices, day = await fetch_prices(adapter, symbols)
            await run_virtual(market, symbols, prices, day, llm_weights, llm_base_weights, logger)

            # ── 메모리: 결과 소급 기입 → 오늘 결정 기록 → admission → probation ──
            try:
                for entry_id, outcome in fill_pending_outcomes(memory, market, prices, today):
                    logger.log(market, "memory_outcome", {"id": entry_id, "outcome": round(outcome, 5)})
                if llm_weights and guard.last_decision:
                    meta = guard.last_decision
                    embedding = None
                    try:
                        rationale = meta.get("rationale", "")
                        if rationale:
                            embedding = (await router.embed([rationale]))[0]
                    except Exception:
                        pass  # 임베딩 실패는 비치명 — pattern_key 폴백
                    entry_id = record_decision(
                        memory, market, today, llm_weights,
                        prev_weights_by_market[market],
                        meta.get("features", {}), meta, prices, embedding=embedding,
                    )
                    if entry_id:
                        logger.log(market, "memory_record", {"id": entry_id})
                for event in promote_candidates(memory, market, today):
                    logger.log(market, "memory_admission", event)
                for event in review_probation(memory, market, today):
                    logger.log(market, "memory_probation", event)
                for event in review_retention(memory, market, today):
                    logger.log(market, "memory_retention", event)
                if today.weekday() == 6:  # 일요일: 주간 reflection
                    from reflection import run_weekly

                    report, ref_events = await run_weekly(memory, market, today, router)
                    if report:
                        logger.log(market, "weekly_reflection", report)
                    for event in ref_events:
                        logger.log(market, "weekly_credit", event)
            except Exception as e:  # 메모리 실패가 매매 루프를 죽이면 안 된다
                logger.log(market, "memory_error", {"error_type": type(e).__name__, "error": str(e)[:200]})
        memory.close()

        # 일일 브리핑 생성 (결정론 — 게이트웨이 /briefing 이 서빙)
        try:
            from interaction import write_briefing

            print(f"briefing={write_briefing(ROOT)}")
        except Exception as e:
            print(f"briefing_error={type(e).__name__}: {str(e)[:120]}")
        return exit_code
    finally:
        for adapter, _ in adapters.values():
            close = getattr(adapter, "close", None)
            if close:
                await close()
        await router.close()


def json_safe_weights_path(state_path: Path) -> dict[str, float] | None:
    if state_path.exists():
        import json

        return json.loads(state_path.read_text(encoding="utf-8")).get("prev_weights")
    return None


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
