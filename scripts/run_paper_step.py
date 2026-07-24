"""일일 페이퍼 스텝 — 실계좌(Risk-guarded LLM) + 가상 포트폴리오 병행 운용.

사용법:
    uv run python scripts/run_paper_step.py            # 실주문 + 가상 A/B (키 있는 전 시장)
    uv run python scripts/run_paper_step.py --dry-run  # 관측·포지션 조회까지만
    uv run python scripts/run_paper_step.py --markets CRYPTO,US   # 시장 선택 (KR 은 10:00 KST 별도 잡)
    uv run python scripts/run_paper_step.py --debate              # debate 강제 소집 (사용자 요청)

구성 (시장별):
- 실계좌: RiskGuardedPolicy(LLMTrader) — 라이브 페이퍼가 진실 verifier.
- 가상 3종(llm/bh/random): 동일 관측·t-1 종가 forward 시뮬레이션 → 델타 측정 기준선.
  가상 llm 은 실계좌와 같은 목표 배분을 사용(추가 LLM 호출 없음).

로그: data/logs/{market}/{date}.jsonl · 가상 상태: data/state/virtual/*.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import VirtualPortfolio, record_meta_shadow  # noqa: E402
from harness import (  # noqa: E402
    BuyAndHold,
    JsonlLogger,
    MarketRun,
    RandomPolicy,
    load_env,
    make_usage_sink,
    notify,
    run_all_markets,
    single_instance,
    wait_for_network,
    with_deadline,
)
from llm import LLMRouter  # noqa: E402
from memory import (  # noqa: E402
    MemoryStore,
    build_query_text,
    fill_pending_outcomes,
    lessons_payload,
    promote_candidates,
    record_decision,
    retrieve,
    review_probation,
    review_retention,
)
from regime import (  # noqa: E402
    INDEX_PROXY,
    compute_regime,
    load_market_signals,
    propose_meta_weights,
    update_regime_signal,
)
from risk import RiskEngine, RiskGuardedPolicy, RiskLimits  # noqa: E402
from trader import LLMTrader  # noqa: E402

# 시장별 유니버스 — 설명 가능한 메이저 집중(2026-07-20, 사용자 승인): 뉴스·데이터
# 커버리지가 좋은 대형 종목만. 버핏식 원칙(사업 이해·경영진)이 문자 그대로 작동하는 대상.
CRYPTO_UNIVERSE = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # 메이저 3종 (전부 연구 유니버스 소속)
US_UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]  # 나스닥 메가캡 5
KR_UNIVERSE = ["005930", "000660", "005380", "035420"]  # 삼성전자·SK하이닉스·현대차·NAVER
LIMITS = {
    # 개별 종목은 지수 ETF 보다 변동성이 크다 — 종목당 상한을 유니버스 크기에 맞춰 강화
    "CRYPTO": RiskLimits(max_weight_per_asset=0.40, min_cash=0.10, max_daily_turnover=0.50, mdd_circuit=0.20),
    "US": RiskLimits(max_weight_per_asset=0.35, min_cash=0.10, max_daily_turnover=0.50, mdd_circuit=0.15),
    "KR": RiskLimits(max_weight_per_asset=0.40, min_cash=0.10, max_daily_turnover=0.50, mdd_circuit=0.15),
}
STATE_DIR = ROOT / "data" / "state"
# 시장별 최신 regime 의 cross-job 공유 — 장 시간 분리로 시장이 별도 잡이어도 메타 제안이 전 시장을 본다.
REGIME_STATE_PATH = STATE_DIR / "regime_latest.json"
COST_BPS = {"CRYPTO": 10.0, "US": 1.0, "KR": 3.0}  # 가상 포트폴리오 거래비용


def load_prev_weights(state_path: Path) -> dict[str, float] | None:
    """리스크 상태 파일에서 직전 목표 배분을 읽는다. 파일 없으면 None."""
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8")).get("prev_weights")
    return None


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
    # US: 실전 KIS 해외주식 키가 있으면 실계좌(실자금)로, 없으면 Alpaca 페이퍼로.
    # 실전은 비율 Risk Engine 이 못 막는 절대 금액을 LiveGuard(1회/일일 상한 + kill switch)로
    # 보완한다. 실전 토큰은 전용 캐시 파일로 재사용(발급 분당 1회·잦은 발급 제한 회피).
    if (
        env.get("KIS_REAL_APP_KEY")
        and env.get("KIS_REAL_APP_SECRET")
        and "-" in env.get("KIS_REAL_ACCOUNT", "")
    ):
        from adapters.kis_overseas import KISOverseasAdapter
        from risk import LiveCaps, LiveGuard

        guard = LiveGuard(
            LiveCaps(
                max_order_notional=float(env.get("LIVE_MAX_ORDER_USD") or 200),
                max_daily_notional=float(env.get("LIVE_MAX_DAILY_USD") or 500),
                kill_switch_path=STATE_DIR / "KILL_SWITCH",
                state_path=STATE_DIR / "live_notional_US.json",
            )
        )
        out["US"] = (
            KISOverseasAdapter(
                env["KIS_REAL_APP_KEY"],
                env["KIS_REAL_APP_SECRET"],
                env["KIS_REAL_ACCOUNT"],
                US_UNIVERSE,
                token_cache=STATE_DIR / "kis_real_token.json",
                mode="real",
                live_guard=guard,
            ),
            US_UNIVERSE,
        )
    elif env.get("ALPACA_PAPER_API_KEY") and env.get("ALPACA_PAPER_SECRET"):
        from adapters.alpaca import AlpacaPaperAdapter

        out["US"] = (
            AlpacaPaperAdapter(env["ALPACA_PAPER_API_KEY"], env["ALPACA_PAPER_SECRET"], US_UNIVERSE),
            US_UNIVERSE,
        )
    # 계좌 형식(8자리-상품코드 2자리)이 맞을 때만 — 아니면 잔고·주문이 전부 실패한다
    if (
        env.get("KIS_PAPER_APP_KEY")
        and env.get("KIS_PAPER_APP_SECRET")
        and "-" in env.get("KIS_PAPER_ACCOUNT", "")
    ):
        from adapters.kis import KISPaperAdapter

        out["KR"] = (
            KISPaperAdapter(
                env["KIS_PAPER_APP_KEY"],
                env["KIS_PAPER_APP_SECRET"],
                env["KIS_PAPER_ACCOUNT"],
                KR_UNIVERSE,
                token_cache=STATE_DIR / "kis_token.json",
            ),
            KR_UNIVERSE,
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
    두 arm 의 equity 델타가 ablation 의 측정치다 (No-Memory vs +Memory).
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


def make_memory_fn(memory: MemoryStore, router: LLMRouter, market: str):
    """검증 통과(active) 교훈 조회 클로저 — LLMTrader.memory_fn 주입용."""

    async def memory_fn(obs, features) -> list[dict]:
        # active 교훈이 없으면 임베딩 호출 자체를 생략 (승격 전 매일 낭비 방지)
        if not (
            memory.query(market, store="semantic", status="active")
            or memory.query(market, store="procedural", status="active")
        ):
            return []
        embedding = None
        try:
            query = build_query_text(market, features)
            embedding = (await router.embed([query]))[0]
        except Exception:
            pass  # relevancy 없이 recency+importance 로 진행 (비치명)
        hits = retrieve(memory, market, obs.asof_day, query_embedding=embedding, k=5)
        return lessons_payload(hits)  # confidence 포함 (블렌딩 입력)

    return memory_fn


def make_forbidden_fn(memory: MemoryStore, market: str):
    """active Forbidden 패턴 집합 클로저 — Risk 가드 하드 veto 입력."""

    def forbidden_patterns() -> set[str]:
        return {
            e.pattern_key
            for e in memory.query(market, store="procedural", status="active")
            if e.pattern_key and e.data.get("kind") == "forbidden"
        }

    return forbidden_patterns


def make_signals_fn(env: dict[str, str], market: str, symbols: list[str]):
    """OOS 검증 팩터 스코어 클로저. 연구 패널 소스가 없는 시장은 None (신호 생략)."""
    from alpha_lab.data import fetch_crypto_panel, make_us_panel_fn

    # 시장별 연구 패널로 active 팩터 top-3 z-score 주입 (US 는 Alpaca 키 필요)
    panel_fn = {"CRYPTO": fetch_crypto_panel, "US": make_us_panel_fn(env)}.get(market)
    if panel_fn is None:
        return None
    from alpha_lab.signals import compute_alpha_signals

    library_path = STATE_DIR / f"alpha_library_{market}.json"

    async def signals_fn(obs) -> dict:
        return await compute_alpha_signals(library_path, symbols, obs.asof_day, panel_fn=panel_fn)

    return signals_fn


def build_market_policy(
    market: str,
    adapter,
    symbols: list[str],
    router: LLMRouter,
    memory: MemoryStore,
    env: dict[str, str],
    debate: str,
) -> RiskGuardedPolicy:
    """시장 1곳의 실계좌 정책 조립 — LLMTrader(교훈·신호 주입) + Risk 가드."""
    risk_path = STATE_DIR / f"risk_{market}.json"
    trader = LLMTrader(
        router, market, symbols, adapter.get_ohlcv_history,
        memory_fn=make_memory_fn(memory, router, market),
        signals_fn=make_signals_fn(env, market, symbols),
        # debate 트리거 입력: 직전 배분(대형 변경 감지) + 사용자 강제 소집
        prev_weights_fn=lambda: load_prev_weights(risk_path),
        debate=debate,
    )
    return RiskGuardedPolicy(
        trader,
        RiskEngine(LIMITS[market]),
        risk_path,
        equity_fn=adapter.get_equity,
        forbidden_patterns_fn=make_forbidden_fn(memory, market),
    )


async def run_memory_pipeline(
    memory: MemoryStore,
    market: str,
    today: date,
    llm_weights: dict | None,
    prev_weights: dict | None,
    decision_meta: dict | None,
    prices: dict,
    router: LLMRouter,
    logger: JsonlLogger,
) -> None:
    """결과 소급 기입 → 오늘 결정 기록 → admission → probation → retention → 주간 reflection."""
    for entry_id, outcome in fill_pending_outcomes(memory, market, prices, today):
        logger.log(market, "memory_outcome", {"id": entry_id, "outcome": round(outcome, 5)})
    if llm_weights and decision_meta:
        embedding = None
        try:
            rationale = decision_meta.get("rationale", "")
            if rationale:
                embedding = (await router.embed([rationale]))[0]
        except Exception:
            pass  # 임베딩 실패는 비치명 — pattern_key 폴백
        entry_id = record_decision(
            memory, market, today, llm_weights, prev_weights,
            decision_meta.get("features", {}), decision_meta, prices, embedding=embedding,
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


async def main() -> int:
    env = load_env(ROOT / ".env")

    # 단일 인스턴스 락 — 같은 시장셋의 catch-up/중복 런이 같은 계좌에 이중 주문하거나
    # 상태 파일(risk_*·live_notional_*)을 레이스로 덮어쓰지 않게 한다(실계좌 경로 필수).
    # 시장셋별 키라 장 시간이 다른 잡(KR 10:00 vs CRYPTO,US 23:00)은 서로 막지 않는다.
    markets_key = "all"
    if "--markets" in sys.argv:
        markets_key = "-".join(sorted(sys.argv[sys.argv.index("--markets") + 1].upper().split(",")))
    lock = single_instance(STATE_DIR / f"run_paper_step_{markets_key}.lock")
    if lock is None:
        print(f"status=skip detail=이미 실행 중(markets={markets_key}) — 중복 실행 차단")
        return 0

    adapters = build_adapters(env)
    # --markets KR / --markets CRYPTO,US — 장 시간이 다른 시장을 별도 잡으로 분리
    if "--markets" in sys.argv:
        wanted = set(sys.argv[sys.argv.index("--markets") + 1].upper().split(","))
        dropped = wanted - set(adapters)
        for adapter, _ in (v for k, v in adapters.items() if k not in wanted):
            close = getattr(adapter, "close", None)
            if close:
                await close()
        adapters = {k: v for k, v in adapters.items() if k in wanted}
        if dropped:
            print(f"status=warn detail=요청 시장 키 없음/형식 오류: {sorted(dropped)}")
    if not adapters:
        print("status=fail detail=.env에 사용 가능한 브로커 키 없음")
        return 1

    # wake/부팅 직후(launchd 캘린더 catch-up) 네트워크 스택이 올라오기 전 조기 실행이면
    # 브로커 호출이 DNS 실패로 죽는다 — 준비될 때까지 대기. 같은 거래일 지연이라 누출 무관.
    if not await wait_for_network():
        print("status=fail event=network_unavailable detail=네트워크 게이트 타임아웃(10분)")
        return 1

    router = LLMRouter(env, usage_sink=make_usage_sink(ROOT))
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

        # 실계좌: Risk-guarded LLM (+ 검증 통과 교훈만 주입)
        debate = "always" if "--debate" in sys.argv else "auto"  # --debate = 사용자 강제 소집
        runs = []
        guards: dict[str, RiskGuardedPolicy] = {}
        prev_weights_by_market: dict[str, dict | None] = {}
        for market, (adapter, symbols) in adapters.items():
            prev_weights_by_market[market] = load_prev_weights(STATE_DIR / f"risk_{market}.json")
            guard = build_market_policy(market, adapter, symbols, router, memory, env, debate)
            guards[market] = guard
            runs.append(MarketRun(adapter, guard, symbols))

        # 관측 스냅샷 영속화(시각화·감사) — 순수 append, 결정/리스크 미개입
        results = await run_all_markets(runs, logger, snapshot_dir=STATE_DIR / "observations")
        exit_code = 0
        for market, outcome in results.items():
            # 실자금 시장(mode=real)만 push 대상 — 페이퍼 실패는 로그로 족하다.
            is_live = getattr(adapters[market][0], "mode", None) == "real"
            if isinstance(outcome, Exception):
                print(f"market={market} status=error error={type(outcome).__name__}: {outcome}")
                exit_code = 1
                if is_live:
                    await notify(env, f"{market} 실계좌 오류", f"{type(outcome).__name__}: {outcome}")
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
                    if is_live:
                        await notify(env, f"{market} 실계좌 주문 거부", outcome.error or "accepted=False")
                elif is_live and meta.get("circuit_open"):
                    # MDD 서킷 발동 — 주문은 동결됐지만 실자금 드로다운이라 즉시 통지.
                    await notify(env, f"{market} 실계좌 MDD 서킷", f"mdd={meta.get('mdd')}")

        # 가상 병행 운용 + 메모리 파이프라인 (실스텝 성공 여부와 무관하게 baseline 은 쌓인다)
        today = datetime.now(timezone.utc).date()
        for market, (adapter, symbols) in adapters.items():
            guard = guards[market]
            llm_weights = None
            llm_base_weights = None
            if not isinstance(results.get(market), Exception) and guard.last_decision:
                llm_weights = load_prev_weights(guard.state_path)
                llm_base_weights = getattr(guard.inner, "last_base_weights", None)

            prices, day = await fetch_prices(adapter, symbols)
            await run_virtual(market, symbols, prices, day, llm_weights, llm_base_weights, logger)

            # 시장 국면 (shadow) — 계산·로깅만, 결정/리스크 미개입. 검증 후 승격.
            regime = await compute_regime(adapter, market, today)
            if regime is not None:
                logger.log(market, "regime", {
                    "state": regime.state, "distribution_days": regime.distribution_days,
                    "drawdown": regime.drawdown, "proxy": INDEX_PROXY.get(market),
                })
                print(f"market={market} regime={regime.state} dd={regime.distribution_days}"
                      f" drawdown={regime.drawdown}")
            # 최신 regime 을 cross-job 공유 상태에 병합 — 이 잡이 자기 시장만 봐도, 메타 제안은
            # 공유 상태에서 전 시장을 읽어 완전해진다(장 시간 분리로 인한 부분 제안·시장 탈락 방지).
            update_regime_signal(
                REGIME_STATE_PATH, market,
                regime.state if regime else None,
                regime.drawdown if regime else 0.0,
                today,
            )

            # ── 메모리 파이프라인 — 실패가 매매 루프를 죽이면 안 된다 ──
            try:
                await run_memory_pipeline(
                    memory, market, today, llm_weights, prev_weights_by_market[market],
                    guard.last_decision, prices, router, logger,
                )
            except Exception as e:
                logger.log(market, "memory_error", {"error_type": type(e).__name__, "error": str(e)[:200]})
        memory.close()

        # 시장 간 shadow 메타 배분 — 공유 regime 상태에서 전 시장을 읽어 제안(부분 잡이어도 완전).
        # 제안·로깅·누적만, 집행/Risk 미개입. 검증 후 승격.
        signals = load_market_signals(REGIME_STATE_PATH)
        if any(s.regime_state is not None for s in signals):
            proposal = propose_meta_weights(signals, asof_day=today)
            if record_meta_shadow(STATE_DIR / "meta_shadow.json", proposal):
                logger.log("META", "meta_shadow", {
                    "day": today.isoformat(), "weights": proposal.weights,
                    "deviation_l1": proposal.deviation_l1, "cited": proposal.cited_signals,
                    "note": proposal.note,
                })
                print(f"market=META meta_shadow weights={proposal.weights}"
                      f" dev={proposal.deviation_l1} note={proposal.note}")

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
        lock.close()  # 락 해제 (프로세스 종료로도 커널이 해제하나 즉시 반납)


if __name__ == "__main__":
    # 런 전체 데드라인 — per-call timeout 이 못 막는 총량 지연(무한대기·느린 LLM 누적)을
    # 상한으로 차단. 초과 시 취소→finally 정리→exit 1 로 락을 확실히 해제한다.
    sys.exit(asyncio.run(with_deadline(main(), label="paper_step")))
