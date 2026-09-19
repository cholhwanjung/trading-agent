"""리스크 한도의 로컬 조정 — 검증 · 미리보기 · 적용 · 기준값 복원.

    uv run python scripts/limits_cli.py show
    uv run python scripts/limits_cli.py set KR min_cash 0.2
    uv run python scripts/limits_cli.py reset KR [min_cash]
    uv run python scripts/limits_cli.py history

적용은 이 명령을 **대화형 터미널에서 실행한 사람**만 한다. 표준 입력이 터미널이 아니면(스크립트나
다른 프로그램이 부른 경우) 검증과 미리보기까지만 하고 끝난다 — 확인 없이 적용하는 옵션은 없다.
한도는 어떤 모델도 쓰지 못하는 값이어야 하고, 그 보장은 "쓰는 길이 여기 하나이고 사람 손을
거친다"는 데서 나온다.

덮어쓰기는 다음 런부터 적용된다 — 일일 스텝과 워처는 프로세스가 뜰 때 한도를 읽는다. 바뀐
한도는 그 뒤 결정 기록의 config_rev 로 드러나고, 지문이 가리키는 값은 data/state/config_revs 에
남는다. 변경 이력은 data/logs/CONFIG 의 config_change 이벤트다.

미리보기는 최근 결정들의 리스크 엔진 통과 전 배분을 옛 한도와 새 한도로 각각 다시 통과시켜
달라지는 결정을 센다. 결정마다 따로 통과시킨 것이라 누적 효과(바뀐 배분이 다음 날의 회전율
계산에 주는 영향)는 담지 않는다. 0건이면 그 한도는 지금 걸리지 않는다는 뜻이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.jsonlog import JsonlLogger, iter_events  # noqa: E402
from risk.engine import RiskEngine, RiskLimits, limits_rev  # noqa: E402
from risk.limits_config import (  # noqa: E402
    BOUNDS,
    LimitsConfig,
    check_market,
    direction,
    load_limits,
    record_limits_rev,
    write_overrides,
)

CONFIG_NS = "CONFIG"  # 시장 횡단 기록이라 사용량·챗 로그처럼 한 곳에 모은다
PREVIEW_DECISIONS = 30


def plan_change(
    config: LimitsConfig,
    tradable: dict[str, list[str]],
    market: str,
    key: str | None,
    value: float | None,
) -> dict:
    """변경 1건의 계획 — 검증 결과와 적용 전후 한도. value=None 은 기준값 복원(key=None 이면
    그 시장의 덮어쓰기 전부). 아무것도 쓰지 않는다."""
    if market not in config.limits:
        return {"errors": [f"unknown_market market={market} known={','.join(config.limits)}"]}
    current = dict(config.overrides.get(market) or {})
    proposed = dict(current)
    keys = [key] if key else sorted(current)
    if value is None:
        if key and key not in BOUNDS:
            return {"errors": [f"not_overridable key={key} allowed={','.join(sorted(BOUNDS))}"]}
        for k in keys:
            proposed.pop(k, None)
    else:
        proposed[key] = value
    after, errors = check_market(config.baseline[market], proposed, tradable[market])
    before = config.limits[market]
    changes = []
    if after is not None:
        for k in keys:
            old, new = getattr(before, k), getattr(after, k)
            changes.append(
                {
                    "key": k,
                    "old": old,
                    "new": new,
                    "baseline": config.baseline[market][k],
                    "direction": direction(k, old, new),
                }
            )
    return {
        "market": market,
        "errors": errors,
        "changes": changes,
        "before": before,
        "after": after,
        "overrides_after": {**config.overrides, market: proposed},
    }


def preview(log_dir: Path | str, market: str, before: RiskLimits, after: RiskLimits) -> dict:
    """최근 결정들을 옛 한도와 새 한도로 각각 다시 통과시켜 달라지는 결정을 센다."""
    steps = list(iter_events(log_dir, market, "daily_step"))
    rows, prev = [], None
    for step in steps:
        decision = step.get("decision") or {}
        raw = decision.get("weights_pre_risk")
        if raw:
            rows.append((step.get("asof_day"), raw, prev, float(decision.get("mdd") or 0.0)))
        if step.get("weights"):
            prev = step["weights"]
    rows = rows[-PREVIEW_DECISIONS:]
    old_engine, new_engine = RiskEngine(before), RiskEngine(after)
    changed = []
    for day, raw, prev_weights, mdd in rows:
        old = old_engine.enforce(raw, prev_weights=prev_weights, mdd=mdd)
        new = new_engine.enforce(raw, prev_weights=prev_weights, mdd=mdd)
        symbols = set(old.weights) | set(new.weights)
        if any(abs(old.weights.get(s, 0.0) - new.weights.get(s, 0.0)) > 1e-9 for s in symbols):
            changed.append(
                {
                    "day": day,
                    "cash_before": round(old.weights.get("CASH", 0.0), 4),
                    "cash_after": round(new.weights.get("CASH", 0.0), 4),
                    "violations_after": [v.split(" ")[0] for v in new.violations],
                }
            )
    return {"decisions": len(rows), "changed": changed}


def apply_change(root: Path | str, plan: dict, logger: JsonlLogger) -> str:
    """계획을 적용한다 — 덮어쓰기 저장 · 새 한도 판본 기록 · 변경 이벤트. 새 config_rev 반환."""
    market, after = plan["market"], plan["after"]
    write_overrides(root, plan["overrides_after"])
    record_limits_rev(root, market, after, plan["overrides_after"].get(market))
    rev = limits_rev(after)
    logger.log(
        CONFIG_NS,
        "config_change",
        {
            "for_market": market,
            "changes": plan["changes"],
            "config_rev_before": limits_rev(plan["before"]),
            "config_rev_after": rev,
            "source": "local_cli",
        },
    )
    return rev


def confirm(expected: str, stdin=None) -> bool:
    """대화형 터미널에서 같은 문구를 다시 입력해야 참. 터미널이 아니면 묻지 않고 거짓."""
    stdin = stdin or sys.stdin
    if not stdin.isatty():
        return False
    print(f"적용하려면 다음을 그대로 입력: {expected}")
    return stdin.readline().strip() == expected


def _limits_line(market: str, limits: RiskLimits) -> str:
    caps = ",".join(f"{s}:{c}" for s, c in sorted(limits.asset_caps.items())) or "-"
    return (
        f"market={market} config_rev={limits_rev(limits)} min_cash={limits.min_cash} "
        f"max_weight_per_asset={limits.max_weight_per_asset} asset_caps={caps} "
        f"max_daily_turnover={limits.max_daily_turnover} mdd_circuit={limits.mdd_circuit}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="리스크 한도의 로컬 조정")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="현재 한도와 기준값 대비 덮어쓰기")
    sub.add_parser("history", help="변경 이력")
    p_set = sub.add_parser("set", help="한도 1건 변경 (검증 → 미리보기 → 확인 → 적용)")
    p_set.add_argument("market")
    p_set.add_argument("key", choices=sorted(BOUNDS))
    p_set.add_argument("value", type=float, help="비율 (20%% 는 0.2)")
    p_reset = sub.add_parser("reset", help="기준값 복원 — 항목을 생략하면 그 시장 전부")
    p_reset.add_argument("market")
    p_reset.add_argument("key", nargs="?", choices=sorted(BOUNDS))
    args = parser.parse_args(argv)

    # 정의역은 일일 스텝의 것을 그대로 쓴다 — 따로 적으면 한쪽만 바뀐다
    from scripts.run_paper_step import TRADABLE

    config = load_limits(ROOT, TRADABLE)
    log_dir = ROOT / "data" / "logs"
    for error in config.errors:
        print(f"config_error {error}")

    if args.cmd == "show":
        for market, limits in config.limits.items():
            print(_limits_line(market, limits))
            for key, value in sorted((config.overrides.get(market) or {}).items()):
                print(
                    f"  override market={market} key={key} value={value} "
                    f"baseline={config.baseline[market][key]}"
                )
        print("bounds " + " ".join(f"{k}={lo}..{hi}" for k, (lo, hi) in sorted(BOUNDS.items())))
        return 0
    if args.cmd == "history":
        for event in iter_events(log_dir, CONFIG_NS, "config_change"):
            for c in event["changes"]:
                print(
                    f"ts={event['ts']} market={event['for_market']} key={c['key']} "
                    f"old={c['old']} new={c['new']} baseline={c['baseline']} "
                    f"direction={c['direction']} config_rev={event['config_rev_after']}"
                )
        return 0

    if config.errors:
        print("warning 버려진 덮어쓰기는 이번 저장에서 파일에서도 빠진다")
    market = args.market.upper()
    value = args.value if args.cmd == "set" else None
    plan = plan_change(config, TRADABLE, market, args.key, value)
    if plan["errors"]:
        for error in plan["errors"]:
            print(f"rejected {error}")
        return 2
    moved = [c for c in plan["changes"] if c["direction"] != "same"]
    if not moved:
        print(f"no_change market={market} — 이미 그 값이다")
        return 0
    print("before " + _limits_line(market, plan["before"]))
    print("after  " + _limits_line(market, plan["after"]))
    for c in moved:
        print(
            f"change market={market} key={c['key']} old={c['old']} new={c['new']} "
            f"baseline={c['baseline']} direction={c['direction']}"
        )
    seen = preview(log_dir, market, plan["before"], plan["after"])
    print(f"preview market={market} decisions={seen['decisions']} changed={len(seen['changed'])}")
    for row in seen["changed"][-5:]:
        print(
            f"  day={row['day']} cash_before={row['cash_before']} cash_after={row['cash_after']} "
            f"violations_after={','.join(row['violations_after']) or '-'}"
        )
    expected = " ".join([market, *(f"{c['key']}={c['new']}" for c in moved)])
    if not confirm(expected):
        print("not_applied — 대화형 터미널에서 확인 문구를 그대로 입력해야 적용된다")
        return 1
    rev = apply_change(ROOT, plan, JsonlLogger(log_dir))
    print(f"applied market={market} config_rev={rev} effective=next_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
