"""LLM Trader — stateless 단일 에이전트 (메모리 없음 기준선).

관측(feature 7종 + [t-3,t-1] 봉·뉴스 + 포지션)을 구조화 프롬프트로 만들어
smart tier LLM 1회 호출 → 결정 스키마 검증 → 배분 벡터 반환.

실패 시 예외를 올린다 — 러너가 스텝을 격리 실패시키므로 주문 없는 안전 no-op 이 된다.
last_decision 에 근거·인용 ID·시나리오가 남아 loop 가 감사 로그에 포함한다.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path

from adapters.allocation import CASH
from adapters.base import Bar, Observation, Position
from llm import LLMRouter
from trader.features import InsufficientHistoryError, compute_features
from trader.schema import parse_decision

HistoryFn = Callable[[list[str], date], Awaitable[dict[str, list[Bar]]]]

# 베이스 지식 — 검증 교훈(memory)과 분리된 사전 원칙. 수정은 승인 게이트.
PLAYBOOK_PATH = Path(__file__).parent / "playbook.md"


def load_playbook() -> str:
    return PLAYBOOK_PATH.read_text(encoding="utf-8") if PLAYBOOK_PATH.exists() else ""

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

# 실시간 트리거 소집 시에만 시스템 프롬프트에 덧붙는다 — 일간 경로는 불변.
# "오늘의 가격은 알 수 없다"는 기본 전제를 이 결정에 한해 예외 처리하고, 하방 방어를
# 기본값으로 지시. 당일 정보는 행동 판단에만, 학습·재해석에 쓰지 않는다.
TRIGGER_SYSTEM_CLAUSE = (
    "[실시간 트리거 모드] 이번 결정은 장중 급변 이벤트로 소집됐다. user 메시지의 "
    "realtime_trigger 에 당일 가격·변동이 제공된다 — '오늘의 가격은 알 수 없다'는 기본 "
    "전제의 예외다. 이 정보로 현재 배분이 여전히 유효한지 재판단하라. 하방 방어(현금 확대)가 "
    "기본 선택지이며, 확신 없는 추격 매수는 금지한다. 이 당일 정보로 과거 관측·feature 를 "
    "재해석하지 말 것."
)


def build_user_prompt(
    obs: Observation,
    positions: list[Position],
    features: dict[str, dict[str, float] | None],
    lessons: list[dict] | None = None,
    alpha_signals: dict | None = None,
    debate: dict | None = None,
    trigger: dict | None = None,
    max_news: int = 10,
) -> str:
    """관측을 구조화 텍스트로 — 자유 산문 없이 JSON 블록 나열.

    lessons 는 admission+probation 을 통과한 active 교훈만 (검증 전 개입 금지).
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
    if debate:
        payload["debate_review"] = {
            "instruction": "아래 Bull/Bear 토론을 검토해 배분을 재결정하라. 논거가 기존 제안을 지지하면 유지해도 된다.",
            **debate,
        }
    if trigger:
        # 관측 윈도우 밖(당일) 신호 — 분리된 라벨 채널(당일 정보 격리)
        payload["realtime_trigger"] = {
            "note": "장중 실시간 이벤트로 소집됨. 아래는 관측 윈도우 밖(당일) 정보 — 현재 "
            "배분의 유효성 재판단에만 쓰고, 이 정보로 과거 관측·feature 를 재해석하지 말 것.",
            **trigger,
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
        memory_fn: Callable[[Observation, dict], Awaitable[list[dict]]] | None = None,
        signals_fn: Callable[[Observation], Awaitable[dict]] | None = None,
        prev_weights_fn: Callable[[], dict[str, float] | None] | None = None,
        debate: str = "auto",  # "auto"(트리거 시만) | "always"(사용자 요청) | "off"
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
        self.prev_weights_fn = prev_weights_fn  # 대형 포지션 변경 트리거 기준
        self.debate = debate
        self.playbook = load_playbook()
        self.last_decision: dict | None = None

    async def _decide_once(
        self, obs, positions, features, lessons: list[dict], signals: dict,
        debate: dict | None = None, trigger: dict | None = None,
    ):
        system = SYSTEM_PROMPT.format(market=self.market, universe=self.universe)
        if self.playbook:
            system += "\n\n" + self.playbook
        if trigger:
            system += "\n\n" + TRIGGER_SYSTEM_CLAUSE
        resp = await self.router.complete(
            self.tier,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(
                        obs, positions, features, lessons, signals, debate, trigger
                    ),
                }
            ],
            max_tokens=4096,
            json_mode=True,
        )
        decision = parse_decision(resp.text, self.universe)
        allocation = dict(decision.allocation)
        allocation.setdefault(CASH, 0.0)
        return allocation, decision, resp

    async def decide(
        self, obs: Observation, positions: list[Position], trigger: dict | None = None
    ) -> dict[str, float]:
        """2-pass residual: base(교훈 없음) 대비 mem(교훈 주입)의 편차만
        confidence·bounded 로 반영. active 교훈이 없으면 base 1회 호출로 끝.

        trigger 가 있으면 실시간 이벤트 소집 — 당일 급변 컨텍스트를 모든
        결정 pass 에 라벨 채널로 주입. 학습 파이프라인은 호출부(watcher)가 생략한다.
        """
        history = await self.history_fn(self.universe, obs.asof_day)
        features: dict[str, dict[str, float] | None] = {}
        for symbol in self.universe:
            try:
                features[symbol] = compute_features(
                    symbol, history.get(symbol, []), obs.asof_day
                ).features
            except InsufficientHistoryError:
                features[symbol] = None  # 계산 불가를 명시 — LLM 이 불확실성으로 취급

        # features 를 함께 넘긴다 — retrieval 이 관측 상태로 relevancy 질의를 만들 수 있게
        lessons = await self.memory_fn(obs, features) if self.memory_fn else []
        signals = {}
        if self.signals_fn:
            try:
                signals = (await self.signals_fn(obs)).get("signals", {})
            except Exception:
                signals = {}  # 신호는 관측 보조 — 실패해도 결정은 진행
        base_alloc, base_decision, base_resp = await self._decide_once(
            obs, positions, features, [], signals, trigger=trigger
        )

        tokens = {"in": base_resp.input_tokens or 0, "out": base_resp.output_tokens or 0}
        influence: dict = {"applied": False, "n_lessons": len(lessons)}
        final, decision = base_alloc, base_decision

        if lessons:
            from memory.influence import blend_allocations

            mem_alloc, mem_decision, mem_resp = await self._decide_once(
                obs, positions, features, lessons, signals, trigger=trigger
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

        # 조건부 debate — 트리거 밖에서는 절대 소집되지 않는다
        debate_meta = None
        if self.debate != "off":
            from trader.debate import debate_trigger, run_debate

            prev = self.prev_weights_fn() if self.prev_weights_fn else None
            # 실시간 trigger 파라미터와 이름 충돌 방지 — debate 소집 사유는 debate_reason
            debate_reason = debate_trigger(final, prev, signals, forced=(self.debate == "always"))
            if debate_reason:
                payload = build_user_prompt(obs, positions, features, lessons, signals, trigger=trigger)
                debate_meta, d_tokens = await run_debate(
                    self.router, self.market, payload, final, debate_reason
                )
                tokens["in"] += d_tokens["in"]
                tokens["out"] += d_tokens["out"]
                re_alloc, re_decision, re_resp = await self._decide_once(
                    obs, positions, features, lessons, signals, debate=debate_meta, trigger=trigger
                )
                tokens["in"] += re_resp.input_tokens or 0
                tokens["out"] += re_resp.output_tokens or 0
                debate_meta["allocation_after"] = re_alloc
                final, decision = re_alloc, re_decision

        self.last_base_weights = base_alloc  # ablation 병행용 (무메모리 arm)
        self.last_decision = {
            "features": features,  # 감사·pattern_key 계산용
            "alpha_signals_provided": sorted(signals),
            "retrieved_memory_ids": [le["id"] for le in lessons],
            "influence": influence,
            "debate": debate_meta,  # None = 미소집 (verify 로그)
            "realtime_trigger": trigger,  # None = 일간 스텝 / dict = 이벤트 소집
            "rationale": decision.rationale,
            "cited_signal_ids": decision.cited_signal_ids,
            "cited_memory_ids": decision.cited_memory_ids,
            "scenario_expected": decision.scenario_expected,
            "scenario_invalidation": decision.scenario_invalidation,
            "model": f"{base_resp.provider}:{base_resp.model}",
            "tokens": tokens,
        }
        return final
