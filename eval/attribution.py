"""결정 파이프라인 단계별 기여 — 1차 결정 → 교훈 블렌딩 → 토론 재결정 → 리스크 적용.

`llm − llm_base` 한 숫자는 뒤 세 단계의 합이다. 교훈이 하나도 없던 기간에도 그 값은 0 이
아니었다 — 토론과 리스크 엔진이 배분을 바꿨기 때문이다. 그래서 메모리가 실제로 더한 몫은
따로 센다: 단계 사이의 배분 차이에, 그 배분이 유효했던 구간의 수익률을 곱한다.

    기여_k(t) = Σ_s (w_k − w_{k−1})_s · r_s(t)

r_s(t) 는 가상 포트폴리오와 같은 규약이다 — t 봉을 보고 내린 결정은 다음 봉 시가에 체결돼
그다음 봉 시가까지 유효하므로 r = open_{t+2} / open_{t+1} − 1. 1차 근사(비용·복리 없음)라
arm 곡선의 차이와 정확히 같지는 않다.

로그와 관측 스냅샷만 읽는다 — 상태 파일을 새로 만들지 않으므로 과거 전 구간에 소급된다.
"""

from __future__ import annotations

import json
from pathlib import Path

from adapters.allocation import CASH
from harness.jsonlog import iter_events

STAGES = ("memory", "debate", "risk")


def load_opens(obs_dir: Path | str, market: str) -> dict[str, dict[str, float]]:
    """관측 스냅샷 → {봉 날짜: {종목: 시가}}. 스냅샷마다 창이 겹치므로 합친다."""
    out: dict[str, dict[str, float]] = {}
    for path in sorted((Path(obs_dir) / market).glob("*.json")):
        bars = json.loads(path.read_text(encoding="utf-8")).get("bars") or {}
        for symbol, rows in bars.items():
            for bar in rows:
                out.setdefault(str(bar["day"])[:10], {})[symbol] = bar["open"]
    return out


def stage_weights(decision: dict, base: dict, final: dict) -> list[dict]:
    """결정 1건 → [1차, 교훈 블렌딩 후, 토론 후, 리스크 적용 후] 배분."""
    influence = decision.get("influence") or {}
    blended = base
    if influence.get("applied"):
        # 블렌딩은 두 배분의 볼록 결합이다 — 배율이 1 이하라 long-only 클램프가 걸리지 않는다
        b, m, scale = influence["base_weights"], influence["mem_weights"], influence["scale"]
        blended = {s: (1 - scale) * b.get(s, 0.0) + scale * m.get(s, 0.0) for s in {*b, *m}}
    # 리스크 엔진이 받은 배분 = 토론이 있었으면 재결정, 없었으면 블렌딩 결과
    debated = decision.get("weights_pre_risk") or blended
    return [base, blended, debated, final]


def _decisions_by_bar(log_dir: Path, market: str) -> dict[str, tuple[dict, dict, dict]]:
    """{봉 날짜: (결정 메타, 1차 배분, 최종 배분)}. 같은 봉의 재실행은 마지막 결정이 유효하다."""
    steps = [s for s in iter_events(log_dir, market, "daily_step") if s.get("decision")]
    latest: dict[str, dict[str, dict]] = {"llm": {}, "llm_base": {}}
    for rec in iter_events(log_dir, market, "virtual_step"):
        arm, day = rec.get("portfolio"), str(rec.get("day", ""))[:10]
        if arm in latest and day and rec.get("weights"):
            latest[arm][day] = rec
    out: dict[str, tuple[dict, dict, dict]] = {}
    for day, rec in latest["llm"].items():
        # 이 가상 스텝을 낳은 결정 = 그 직전에 기록된 일간 결정
        prior = [s for s in steps if s["ts"] <= rec["ts"]]
        if prior and day in latest["llm_base"]:
            out[day] = (prior[-1]["decision"], latest["llm_base"][day]["weights"], rec["weights"])
    return out


def stage_attribution(log_dir: Path | str, obs_dir: Path | str, market: str) -> dict | None:
    """시장 1곳의 단계별 누적 기여(%p)와 개입 일수. 계산할 결정이 없으면 None."""
    opens = load_opens(obs_dir, market)
    days = sorted(opens)
    following = {d: (days[i + 1], days[i + 2]) for i, d in enumerate(days[:-2])}
    total = dict.fromkeys(STAGES, 0.0)
    active = dict.fromkeys(STAGES, 0)
    n = 0
    for day, (decision, base, final) in sorted(_decisions_by_bar(Path(log_dir), market).items()):
        if day not in following:
            continue  # 체결 봉과 그다음 봉이 아직 없다 — 수익률이 확정되지 않았다
        fill, exit_ = (opens[d] for d in following[day])
        returns = {s: exit_[s] / fill[s] - 1.0 for s in fill if s in exit_ and fill[s]}
        stages = stage_weights(decision, base, final)
        for name, before, after in zip(STAGES, stages, stages[1:]):
            moved = {s: after.get(s, 0.0) - before.get(s, 0.0) for s in {*before, *after} - {CASH}}
            total[name] += sum(dw * returns.get(s, 0.0) for s, dw in moved.items())
            active[name] += any(abs(dw) > 1e-9 for dw in moved.values())
        n += 1
    if not n:
        return None
    return {
        "market": market,
        "n": n,
        "stages": {k: {"pct": total[k] * 100, "days": active[k]} for k in STAGES},
        "total_pct": sum(total.values()) * 100,
    }
