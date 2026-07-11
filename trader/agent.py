"""LLM Trader — Phase 1 stateless 단일 에이전트 (R3, [ADR-009] 메모리 없음 기준선).

관측(feature 7종 + [t-3,t-1] 봉·뉴스 + 포지션)을 구조화 프롬프트로 만들어
smart tier LLM 1회 호출 → 결정 스키마(R5) 검증 → 배분 벡터 반환.

실패 시 예외를 올린다 — 러너가 스텝을 격리 실패시키므로 주문 없는 안전 no-op 이 된다.
last_decision 에 근거·인용 ID·시나리오가 남아 loop 가 감사 로그에 포함한다.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import date

from adapters.allocation import CASH
from adapters.base import Bar, Observation, Position
from llm import LLMRouter
from trader.features import InsufficientHistoryError, compute_features
from trader.schema import parse_decision

HistoryFn = Callable[[list[str], date], Awaitable[dict[str, list[Bar]]]]

SYSTEM_PROMPT = """\
너는 {market} 시장의 포트폴리오 매니저다. 매일 1회, 자산 배분비율만으로 의사를 표현한다.

제약 (위반 시 결정 전체가 거부된다):
- 배분 대상은 유니버스 {universe} 와 "CASH" 뿐이다. 그 외 심볼 금지.
- 모든 비중 ≥ 0 (long-only), 합계 = 1.0. 현금도 포지션이다 — 확신이 없으면 현금 비중을 높여라.
- 관측 데이터는 전일까지다. 오늘의 가격은 알 수 없다.
- verified_lessons 는 반복 검증을 통과한 과거 교훈이다. 참고했다면 해당 id 를
  cited_memory_ids 에 넣어라. 없으면 빈 리스트.
- alpha_signals 는 OOS 검증된 팩터의 당일 스코어다(양수 = 익일 상대 우위 기대,
  oos_ic 가 신뢰 크기). 참고했다면 해당 키를 cited_signal_ids 에 넣어라.

반드시 아래 JSON 만 출력한다 (설명 문장 금지):
{{
  "allocation": {{"<symbol>": <float>, ..., "CASH": <float>}},
  "rationale": "<핵심 근거 2~3문장, 한국어>",
  "cited_signal_ids": ["<참고한 feature 이름>", ...],
  "cited_memory_ids": [],
  "scenario": {{
    "expected": "<예상 시나리오 1문장>",
    "invalidation": "<이 결정이 틀렸다고 판정할 구체적 조건 1문장>"
  }}
}}"""


def build_user_prompt(
    obs: Observation,
    positions: list[Position],
    features: dict[str, dict[str, float] | None],
    lessons: list[dict] | None = None,
    alpha_signals: dict | None = None,
    max_news: int = 10,
) -> str:
    """관측을 구조화 텍스트로 — 자유 산문 없이 JSON 블록 나열 (하드룰 10).

    lessons 는 admission+probation 을 통과한 active 교훈만 (하드룰 3 — 검증 전 개입 금지).
    """

    recent = {
        s: [{"day": str(b.day), "close": b.close, "volume": b.volume} for b in bars]
        for s, bars in obs.bars.items()
    }
    payload = {
        "asof_day": str(obs.asof_day),
        "features": features,  # None = 이력 부족으로 계산 불가
        "recent_bars_t3_t1": recent,
        "news_headlines": [
            {"day": str(n.published_at.date()), "headline": n.headline}
            for n in obs.news[:max_news]
        ],
        "current_positions": [
            {"symbol": p.symbol, "market_value": p.market_value} for p in positions
        ],
        "verified_lessons": lessons or [],
        "alpha_signals": alpha_signals or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


class LLMTrader:
    """Policy 계약 구현체. history_fn 은 어댑터의 get_ohlcv_history 를 바인딩해 주입."""

    def __init__(
        self,
        router: LLMRouter,
        market: str,
        universe: list[str],
        history_fn: HistoryFn,
        tier: str = "smart",
        memory_fn: Callable[[Observation], Awaitable[list[dict]]] | None = None,
        signals_fn: Callable[[Observation], Awaitable[dict]] | None = None,
    ) -> None:
        provider, model = router.spec(tier)  # 로그 표기용
        self.name = f"llm_trader:{provider}:{model}"
        self.router = router
        self.market = market
        self.universe = universe
        self.history_fn = history_fn
        self.tier = tier
        self.memory_fn = memory_fn  # active 교훈만 반환해야 한다 (memory.retrieval)
        self.signals_fn = signals_fn  # OOS 검증 팩터 스코어 (alpha_lab.signals)
        self.last_decision: dict | None = None

    async def _decide_once(self, obs, positions, features, lessons: list[dict], signals: dict):
        resp = await self.router.complete(
            self.tier,
            system=SYSTEM_PROMPT.format(market=self.market, universe=self.universe),
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(obs, positions, features, lessons, signals),
                }
            ],
            max_tokens=4096,
            json_mode=True,
        )
        decision = parse_decision(resp.text, self.universe)
        allocation = dict(decision.allocation)
        allocation.setdefault(CASH, 0.0)
        return allocation, decision, resp

    async def decide(self, obs: Observation, positions: list[Position]) -> dict[str, float]:
        """2-pass residual (R9): base(교훈 없음) 대비 mem(교훈 주입)의 편차만
        confidence·bounded 로 반영. active 교훈이 없으면 base 1회 호출로 끝."""
        history = await self.history_fn(self.universe, obs.asof_day)
        features: dict[str, dict[str, float] | None] = {}
        for symbol in self.universe:
            try:
                features[symbol] = compute_features(
                    symbol, history.get(symbol, []), obs.asof_day
                ).features
            except InsufficientHistoryError:
                features[symbol] = None  # 계산 불가를 명시 — LLM 이 불확실성으로 취급

        lessons = await self.memory_fn(obs) if self.memory_fn else []
        signals = {}
        if self.signals_fn:
            try:
                signals = (await self.signals_fn(obs)).get("signals", {})
            except Exception:
                signals = {}  # 신호는 관측 보조 — 실패해도 결정은 진행
        base_alloc, base_decision, base_resp = await self._decide_once(
            obs, positions, features, [], signals
        )

        tokens = {"in": base_resp.input_tokens or 0, "out": base_resp.output_tokens or 0}
        influence: dict = {"applied": False, "n_lessons": len(lessons)}
        final, decision = base_alloc, base_decision

        if lessons:
            from memory.influence import blend_allocations

            mem_alloc, mem_decision, mem_resp = await self._decide_once(
                obs, positions, features, lessons, signals
            )
            tokens["in"] += mem_resp.input_tokens or 0
            tokens["out"] += mem_resp.output_tokens or 0
            cited = [le for le in lessons if le["id"] in mem_decision.cited_memory_ids]
            blend = blend_allocations(base_alloc, mem_alloc, cited)
            influence = {
                "applied": blend.applied,
                "n_lessons": len(lessons),
                "confidence": round(blend.confidence, 4),
                "deviation_l1": round(blend.deviation_l1, 4),
                "scale": round(blend.scale, 4),
                "base_weights": base_alloc,
                "mem_weights": mem_alloc,
            }
            if blend.applied:
                final, decision = blend.weights, mem_decision

        self.last_base_weights = base_alloc  # ablation 병행용 (무메모리 arm)
        self.last_decision = {
            "features": features,  # 감사·pattern_key 계산용
            "alpha_signals_provided": sorted(signals),
            "retrieved_memory_ids": [le["id"] for le in lessons],
            "influence": influence,
            "rationale": decision.rationale,
            "cited_signal_ids": decision.cited_signal_ids,
            "cited_memory_ids": decision.cited_memory_ids,
            "scenario_expected": decision.scenario_expected,
            "scenario_invalidation": decision.scenario_invalidation,
            "model": f"{base_resp.provider}:{base_resp.model}",
            "tokens": tokens,
        }
        return final
