"""주간 메타 배분 shadow 스텝 — 결과 기입 → 메모리 소급·admission → 입력 패널 → 뉴스 다이제스트
→ 제안(교훈 있으면 base/mem 두 번 → 잔차 블렌드) → 감사 → 원장·episodic.

집행은 없다. 제안은 anchor ± max_tilt 로 clamp 되어 `data/state/meta_shadow_llm.json` 에
주 1회 기록되고, 채점은 `scripts/report_ablation.py` 가 결정론 국면 틸트·균등 baseline 과
같은 방법으로 한다. 같은 ISO 주에 다시 돌면 건너뛴다(--force 로 재제안).

사용법:
    uv run python scripts/run_meta_step.py              # 이번 주 제안(이미 있으면 skip)
    uv run python scripts/run_meta_step.py --dry-run    # 입력 패널·헤드라인만 출력, LLM 호출·기록 없음
    uv run python scripts/run_meta_step.py --force      # 이번 주 레코드 교체
스케줄: 월요일 11:00 KST — 세 시장의 일간 스텝이 전부 끝나 국면·뉴스가 t−1 로 갱신된 뒤.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import (  # noqa: E402
    JsonlLogger,
    load_env,
    make_usage_sink,
    single_instance,
    wait_for_network,
    with_deadline,
)
from llm import LLMRouter, extract_json  # noqa: E402
from memory import MemoryStore  # noqa: E402
from meta import (  # noqa: E402
    MARKETS,
    MetaProposalError,
    assign_ids,
    build_panel,
    deterministic_audit,
    digest_news,
    fill_outcomes,
    last_record,
    llm_audit,
    load_llm_ledger,
    load_recent_news,
    market_panel,
    propose,
    record_llm_proposal,
    reliability_lines,
    select_news,
    signal_reliability,
    week_key,
)
from meta.audit import llm_flags  # noqa: E402
from meta.feeds import fetch_proxy_bars  # noqa: E402
from meta.memory import (  # noqa: E402
    admit_meta,
    blend_meta,
    fill_meta_outcomes,
    meta_lessons,
    meta_pattern_key,
    record_meta_episode,
)
from meta.outcomes import save_reliability  # noqa: E402
from regime.live import compute_macro_regime  # noqa: E402
from trader.prompt_store import assemble_section, record_revision  # noqa: E402

STATE_DIR = ROOT / "data" / "state"
VIRTUAL_DIR = STATE_DIR / "virtual"
OBS_DIR = STATE_DIR / "observations"
LEDGER = STATE_DIR / "meta_shadow_llm.json"
RELIABILITY = STATE_DIR / "meta_signal_reliability.json"
PROMPT_REV_DIR = STATE_DIR / "prompt_revs"
MEMORY_PATH = ROOT / "data" / "memory.sqlite"

ANCHOR = {m: 1.0 / len(MARKETS) for m in MARKETS}  # 전략 배분 — 실험 중 변경 금지
MAX_TILT = 0.15


def _valuation(market: str) -> dict | None:
    path = STATE_DIR / f"index_valuation_{market}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("value") or {}
    except json.JSONDecodeError:
        return None
    picked = {
        k: value.get(k) for k in ("pe", "pb", "dividend_yield", "asof") if value.get(k) is not None
    }
    return picked or None


def _actual_split() -> dict[str, float] | None:
    """실계좌 시장별 평가액(원화 환산, 직전 스텝) → 비율. 하나라도 없으면 None."""
    equity: dict[str, float] = {}
    for m in MARKETS:
        path = STATE_DIR / f"risk_{m}.json"
        if not path.exists():
            return None
        try:
            v = json.loads(path.read_text(encoding="utf-8")).get("prev_equity")
        except json.JSONDecodeError:
            return None
        if not v or v <= 0:
            return None
        equity[m] = float(v)
    total = sum(equity.values())
    return {m: round(v / total, 4) for m, v in equity.items()}


async def _reflect(router: LLMRouter, spec, record: dict) -> dict | None:
    """지평 뒤 회고(fast tier) — 제안을 고쳐 쓰지 않고 인용 id 의 지지/반박만 적는다."""
    payload = {
        "proposal": {k: record.get(k) for k in ("day", "weights", "anchor", "cited", "rationale")},
        "outcome": record.get("outcome"),
    }
    resp = await router.complete(
        spec.tier,
        purpose="meta_reflection",
        system=spec.text,
        messages=[
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}
        ],
        json_mode=True,
        temperature=0.0,
    )
    data = extract_json(resp.text)
    if not isinstance(data, dict) or not str(data.get("lesson", "")).strip():
        return None
    cited = set(record.get("cited") or [])
    return {
        "lesson": str(data["lesson"])[:300],
        "right": [str(x) for x in data.get("right") or [] if str(x) in cited],
        "wrong": [str(x) for x in data.get("wrong") or [] if str(x) in cited],
    }


async def run(today: date, force: bool, dry_run: bool) -> int:
    env = load_env(ROOT / ".env")
    week = week_key(today)
    last = last_record(LEDGER)
    if last and last.get("week") == week and not force and not dry_run:
        print(
            f"status=skip week={week} detail=이번 주 제안 있음(day={last['day']}) — --force 로 교체"
        )
        return 0

    logger = JsonlLogger(ROOT / "data" / "logs")
    spec_meta = assemble_section("meta")
    spec_digest = assemble_section("meta_digest")
    spec_audit = assemble_section("meta_audit")
    spec_reflect = assemble_section("meta_reflect")

    # 2. 입력 패널 — 프록시 일봉은 라이브 국면과 같은 창, 상한 t−1 (dry-run 도 여기까지)
    bars = await fetch_proxy_bars(env, STATE_DIR, today)
    panels = {m: market_panel(m, bars.get(m, []), today, _valuation(m)) for m in MARKETS}
    macro = await compute_macro_regime(env, today)
    macro_d = (
        {"state": macro.state, "vix": macro.vix, "yield_spread": macro.yield_spread}
        if macro
        else None
    )
    panel = build_panel(
        panels,
        today,
        ANCHOR,
        MAX_TILT,
        macro=macro_d,
        prev_proposal={"day": last["day"], "weights": last["weights"]} if last else None,
        actual_split=_actual_split(),
    )
    for m, p in panels.items():
        print(
            f"panel market={m} bars={len(bars.get(m, []))} regime={p.regime} dd={p.drawdown}"
            f" mom12m={p.trailing.get('12m')} vol60={p.vol.get('60')}"
        )
    pattern = meta_pattern_key(
        {m: p.regime for m, p in panels.items()},
        macro_d["state"] if macro_d else None,
        {m: p.vol.get("60") for m, p in panels.items()},
    )
    by_market = assign_ids({m: select_news(load_recent_news(OBS_DIR, m, today)) for m in MARKETS})
    if dry_run:
        print(
            json.dumps(
                {
                    "panel": panel,
                    "pattern": pattern,
                    "news_counts": {m: len(v) for m, v in by_market.items()},
                },
                ensure_ascii=False,
                indent=1,
                default=str,
            )
        )
        print("status=dry_run detail=LLM 호출·기록 없음")
        return 0

    router = LLMRouter(env, usage_sink=make_usage_sink(ROOT))
    memory = MemoryStore(MEMORY_PATH)
    try:
        # 1. 지난 제안의 결과 기입 → episodic 소급 → 회고 → admission. 제안보다 먼저 —
        #    신뢰도 표·승격 교훈이 이번 제안에 들어가도록.
        for e in fill_outcomes(LEDGER, VIRTUAL_DIR):
            logger.log("META", "meta_llm_outcome", e)
            print(
                f"outcome week={e['week']} delta_vs_anchor={e['delta_vs_anchor']:+.4f} hits={e['direction_hits']}"
            )
        history = load_llm_ledger(LEDGER)
        filled = fill_meta_outcomes(memory, history)
        by_day = {h["day"]: h for h in history}
        for entry_id, _ in filled:
            entry = memory.get(entry_id)
            rec = by_day.get(entry.day.isoformat()) if entry else None
            if rec is None:
                continue
            try:
                reflection = await _reflect(router, spec_reflect, rec)
            except Exception as ex:  # 회고 실패는 제안을 막지 않는다
                reflection = None
                print(f"reflection_error week={rec.get('week')} error={type(ex).__name__}")
            if reflection:
                memory.update(entry_id, data={**entry.data, "reflection": reflection})
                logger.log(
                    "META",
                    "meta_llm_reflection",
                    {"day": rec["day"], "week": rec.get("week"), **reflection},
                )
        for ev in admit_meta(memory, today):
            logger.log("META", ev.pop("event"), ev)
            print(f"admission {ev}")
        reliability = signal_reliability(LEDGER)
        save_reliability(RELIABILITY, reliability)

        if record_revision(PROMPT_REV_DIR, spec_meta, today):
            print(f"prompt_rev={spec_meta.rev} 신규 판본 기록 blocks={','.join(spec_meta.blocks)}")

        # 3. 뉴스 다이제스트
        digest = await digest_news(router, spec_digest, by_market)
        news_ids = {m: sorted({i for ln in digest[m] for i in ln["ids"]}) for m in MARKETS}
        logger.log(
            "META",
            "meta_digest",
            {
                "day": today.isoformat(),
                "week": week,
                "headlines": {m: len(v) for m, v in by_market.items()},
                "lines": {m: len(v) for m, v in digest.items()},
                "news_ids_injected": news_ids,
            },
        )
        for m in MARKETS:
            if not news_ids[m]:
                print(f"news_injected=0 market={m} headlines={len(by_market[m])}")

        # 4. 제안 — 승격 교훈이 있으면 base(교훈 없음)·mem(교훈 포함) 두 번 → confidence 잔차 블렌드
        lessons = meta_lessons(memory, today)
        valid_ids = (
            set(panel["signal_ids"])
            | {it["id"] for v in by_market.values() for it in v}
            | {le["id"] for le in lessons}
        )
        rel_lines = reliability_lines(reliability)
        record: dict = {
            "day": today.isoformat(),
            "anchor": ANCHOR,
            "max_tilt": MAX_TILT,
            "prompt_rev": spec_meta.rev,
            "news_ids_injected": news_ids,
            "retrieved_memory_ids": [le["id"] for le in lessons],
            "pattern_key": pattern,
            "inputs": {
                "regime": {m: p.regime for m, p in panels.items()},
                "macro": macro_d["state"] if macro_d else None,
            },
        }
        try:
            base = await propose(
                router, spec_meta, panel, digest, lessons=[], reliability=rel_lines
            )
            proposal = base
            if lessons:
                mem = await propose(
                    router, spec_meta, panel, digest, lessons=lessons, reliability=rel_lines
                )
                cited_lessons = [le for le in lessons if le["id"] in mem.cited]
                blended, blend = blend_meta(base.weights, mem.weights, cited_lessons)
                proposal = mem
                record.update(
                    {
                        "weights_base": base.weights,
                        "weights_mem": mem.weights,
                        "memory_scale": round(blend.scale, 4),
                        "memory_applied": blend.applied,
                    }
                )
                proposal = type(mem)(**{**mem.__dict__, "weights": blended})
        except MetaProposalError as e:
            record.update(
                {
                    "weights": ANCHOR,
                    "error": f"parse: {str(e)[:200]}",
                    "audit": {"flags": ["parse_error"], "reject": True},
                }
            )
            record_llm_proposal(LEDGER, record)
            logger.log("META", "meta_llm_proposal", record)
            print(f"status=parse_error week={week} detail={str(e)[:120]}")
            return 1

        audit = deterministic_audit(proposal, valid_ids, panel)
        if not audit.reject:
            audit.llm = await llm_audit(router, spec_audit, proposal, panel, digest)
            audit.flags += llm_flags(audit.llm)
        record.update(
            {
                "weights": ANCHOR if audit.reject else proposal.weights,
                "weights_raw": proposal.weights_raw,
                "confidence": proposal.confidence,
                "cited": proposal.cited,
                "rationale": proposal.rationale,
                "horizon": proposal.horizon,
                "audit": {
                    "flags": audit.flags,
                    "fabricated": audit.fabricated,
                    "reject": audit.reject,
                    "llm": audit.llm,
                },
            }
        )
        record_llm_proposal(LEDGER, record)
        record["week"] = week
        record_meta_episode(memory, record, pattern, today)
        logger.log("META", "meta_llm_proposal", record)
        dev = 0.5 * sum(abs(record["weights"][m] - ANCHOR[m]) for m in MARKETS)
        print(
            f"market=META meta_llm_proposal week={week} weights={record['weights']}"
            f" dev={dev:.4f} confidence={proposal.confidence} flags={audit.flags}"
            f" reject={audit.reject} lessons={len(lessons)} prompt_rev={spec_meta.rev}"
        )
        return 0
    finally:
        memory.close()
        await router.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="주간 메타 배분 shadow 스텝")
    p.add_argument("--force", action="store_true", help="이번 주 레코드가 있어도 다시 제안")
    p.add_argument("--dry-run", action="store_true", help="입력만 출력 — LLM 호출·기록 없음")
    p.add_argument("--asof", type=date.fromisoformat, default=None, help="기준일(기본 오늘 UTC)")
    return p.parse_args()


async def main() -> int:
    args = _parse_args()
    today = args.asof or datetime.now(timezone.utc).date()
    lock = single_instance(STATE_DIR / "run_meta_step.lock", label="메타 배분 스텝")
    if lock is None:
        print("status=skip detail=이미 실행 중")
        return 0
    if not args.dry_run and not await wait_for_network():
        print("status=fail event=network_unavailable")
        return 1
    return await run(today, args.force, args.dry_run)


if __name__ == "__main__":
    sys.exit(asyncio.run(with_deadline(main(), label="meta_step")))
