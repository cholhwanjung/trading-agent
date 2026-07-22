"""LiveTradeBench 플러그인 스캐폴드 — 우리 Trader 를 벤치 에이전트로 노출.

live-trade-bench 는 프로젝트 의존성이 아니다(벤치 실행 시에만):
    uv run --with live-trade-bench python -c "from eval.ltb_agent import make_ltb_agent; ..."

통합 지점: BaseAgent.generate_allocation 오버라이드 — 벤치의 프롬프트/LLM 경로를
우회하고 우리 파이프라인(feature → LLMTrader → RiskEngine)으로 배분을 산출한다.
벤치 market_data 는 종가 이력만 제공하므로 OHLV 기반 feature(atr14_ratio,
vol_ratio_20d)는 None 처리 — 프롬프트에 '계산 불가'로 명시된다.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

from adapters.allocation import CASH
from adapters.base import Bar, Observation
from llm import LLMRouter
from risk import RiskEngine, RiskLimits
from trader.agent import SYSTEM_PROMPT, build_user_prompt
from trader.features import InsufficientHistoryError, compute_features
from trader.schema import parse_decision

UNAVAILABLE_FEATURES = ("atr14_ratio", "vol_ratio_20d")  # 종가 이력만으론 계산 불가


def closes_to_bars(closes: list[float], asof: date) -> list[Bar]:
    """벤치의 종가 이력 → 합성 일봉 (end=t-1). high/low/volume 은 종가로 채운다."""
    end = asof - timedelta(days=1)
    start = end - timedelta(days=len(closes) - 1)
    return [
        Bar(day=start + timedelta(days=i), open=c, high=c, low=c, close=c, volume=0.0)
        for i, c in enumerate(closes)
    ]


def compute_bench_features(closes: list[float], symbol: str, asof: date) -> dict | None:
    """종가 전용 feature — OHLV 필요 항목은 None 으로 명시."""
    try:
        fs = compute_features(symbol, closes_to_bars(closes, asof), asof)
    except InsufficientHistoryError:
        return None
    features = dict(fs.features)
    for name in UNAVAILABLE_FEATURES:
        features[name] = None
    return features


def make_ltb_agent(
    name: str,
    router: LLMRouter,
    limits: RiskLimits | None = None,
    tier: str = "smart",
):
    """live-trade-bench BaseAgent 서브클래스 인스턴스 생성 (지연 import)."""
    from live_trade_bench.agents.base_agent import BaseAgent

    engine = RiskEngine(limits or RiskLimits())

    class TradingAgentPlugin(BaseAgent):
        def __init__(self) -> None:
            super().__init__(name, model_name="external")  # 벤치 내부 LLM 경로 미사용
            self._prev_weights: dict[str, float] | None = None

        # 추상 메서드 — generate_allocation 오버라이드로 미사용이지만 계약상 구현
        def _prepare_market_analysis(self, market_data):
            return ""

        def _get_portfolio_prompt(self, analysis, market_data, date=None):
            return ""

        def generate_allocation(self, market_data, account_data, date=None, news_data=None):
            asof = (
                datetime.strptime(date, "%Y-%m-%d").date()
                if date
                else datetime.now(timezone.utc).date()
            )
            universe = list(market_data.keys())
            features = {
                s: compute_bench_features(d.get("price_history") or [], s, asof)
                for s, d in market_data.items()
            }
            bars = {
                s: closes_to_bars((d.get("price_history") or [])[-3:], asof)
                for s, d in market_data.items()
            }
            obs = Observation(
                market="LTB",
                asof_day=asof,
                collected_at=datetime.now(timezone.utc),
                bars=bars,
                news=[],
            )

            async def _decide() -> dict[str, float]:
                resp = await router.complete(
                    tier,
                    system=SYSTEM_PROMPT.format(market="LTB", universe=universe),
                    messages=[
                        {"role": "user", "content": build_user_prompt(obs, [], features)}
                    ],
                    max_tokens=4096,
                    json_mode=True,
                )
                return parse_decision(resp.text, universe).allocation

            raw = asyncio.run(_decide())
            decision = engine.enforce(raw, prev_weights=self._prev_weights)
            weights = dict(decision.weights)
            weights.setdefault(CASH, 0.0)
            self._prev_weights = weights
            return weights

    return TradingAgentPlugin()
