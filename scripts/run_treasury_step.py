"""주간/월간 자금 이체 스텝 — 메타 제안 → 이체 계획 → 결정론 가드 판정 (dry-run 기본).

사용법:
    uv run python scripts/run_treasury_step.py            # dry-run: 계획·판정 로깅, 집행 X
    uv run python scripts/run_treasury_step.py --arm bh   # 잔고 기준 가상 arm (기본 llm)

dry-run 은 실자금을 움직이지 않는다: 최신 meta_shadow 제안 → 버킷 매핑(A2) →
plan_transfers → enforce_transfer 까지 순수 결정론 파이프라인을 **로깅만** 한다.
실집행(UpbitTreasury.withdraw_krw · ManualLedger.emit)과 상태 갱신은 없다 —
메타 검증 + ALLOWLIST 하드코딩 후 별도 활성화.

버킷(A2): UPBIT=CRYPTO / KIS=US+KR(통합증거금). 금액 단위는 가상 포트폴리오 nominal
(전 시장 10만 시작 = 공통 단위) — 실계좌 KRW 환산은 활성화 시.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import load_meta_shadow  # noqa: E402
from harness import JsonlLogger  # noqa: E402
from treasury import ALLOWLIST, TreasuryLimits, enforce_transfer, plan_transfers  # noqa: E402
from treasury.manual import MANUAL_ROUTES  # noqa: E402

STATE_DIR = ROOT / "data" / "state"
VIRTUAL = STATE_DIR / "virtual"
MARKET_TO_BUCKET = {"CRYPTO": "UPBIT", "US": "KIS", "KR": "KIS"}
# dry-run 판정 표시용 — 실 ALLOWLIST(집행 게이트)가 아니다. 활성화 시에만 guard.ALLOWLIST 하드코딩.
PREVIEW_ALLOWLIST = frozenset({"UPBIT", "KIS"})


def _last_equity(virtual_dir: Path, market: str, arm: str) -> float | None:
    path = virtual_dir / f"{market}_{arm}.json"
    if not path.exists():
        return None
    hist = json.loads(path.read_text(encoding="utf-8")).get("history") or []
    return hist[-1]["equity"] if hist else None


def _latest_market_weights(ledger_path: Path) -> dict[str, float] | None:
    """meta_shadow 원장에서 최신 제안일의 시장 가중치. 없으면 None."""
    wbd = load_meta_shadow(ledger_path)
    if not wbd:
        return None
    return wbd[sorted(wbd)[-1]]


def _to_buckets(
    market_weights: dict[str, float], virtual_dir: Path, arm: str
) -> tuple[dict[str, float], dict[str, float]]:
    """시장 가중치·가상 잔고를 A2 버킷으로 집계. 공통 시장(가중치∧잔고)만 포함,
    버킷 target 은 포함분으로 재정규화(∑=1) — target·equity 를 같은 venue 집합에 정렬."""
    bucket_equity: dict[str, float] = {}
    bucket_target_raw: dict[str, float] = {}
    for market, bucket in MARKET_TO_BUCKET.items():
        eq = _last_equity(virtual_dir, market, arm)
        w = market_weights.get(market)
        if eq is None or w is None:
            continue
        bucket_equity[bucket] = bucket_equity.get(bucket, 0.0) + eq
        bucket_target_raw[bucket] = bucket_target_raw.get(bucket, 0.0) + w
    tw = sum(bucket_target_raw.values())
    bucket_target = {b: (w / tw if tw > 0 else 0.0) for b, w in bucket_target_raw.items()}
    return bucket_equity, bucket_target


def main() -> int:
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "llm"
    logger = JsonlLogger(ROOT / "data" / "logs")
    now = datetime.now(timezone.utc)

    market_weights = _latest_market_weights(STATE_DIR / "meta_shadow.json")
    if not market_weights:
        print("status=empty detail=meta_shadow 제안 없음 — run_paper_step 이 먼저 돌아야 한다")
        return 1

    bucket_equity, bucket_target = _to_buckets(market_weights, VIRTUAL, arm)
    if len(bucket_equity) < 2:
        print(f"status=skip detail=버킷 2개 미만(공통 시장 부족) buckets={list(bucket_equity)}")
        return 0

    total = sum(bucket_equity.values())
    # dry-run 예시 캘리브레이션 — 활성화 시 실 KRW 로 재설정. min_drift<max_tilt(0.10)라 메타 틸트 반영.
    limits = TreasuryLimits(
        per_transfer_cap=round(0.20 * total, 2),
        daily_cap=round(0.30 * total, 2),
        min_drift_to_fire=0.05,
    )
    intents = plan_transfers(bucket_target, bucket_equity, limits, reason="meta_rebalance")

    # 이체 상태(쿨다운·일일누적) — 있으면 읽되 dry-run 은 갱신하지 않는다.
    state_path = STATE_DIR / "treasury_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"last_transfer_at": None, "daily_moved": 0.0}
    )

    current_split = {v: round(eq / total, 4) for v, eq in bucket_equity.items()}
    target_view = {v: round(w, 4) for v, w in bucket_target.items()}
    logger.log("TREASURY", "treasury_dryrun_plan", {
        "arm": arm, "market_weights": market_weights, "bucket_target": target_view,
        "bucket_equity": {v: round(eq, 2) for v, eq in bucket_equity.items()},
        "current_split": current_split, "n_intents": len(intents),
        "limits": {"per_transfer_cap": limits.per_transfer_cap, "daily_cap": limits.daily_cap,
                   "min_drift_to_fire": limits.min_drift_to_fire},
    })
    print(f"market=TREASURY dryrun arm={arm} total={total:,.2f} target={target_view}"
          f" current={current_split} n_intents={len(intents)}")

    if not intents:
        print("status=ok detail=무이체 — 드리프트 임계 미달(churn 회피)")
        return 0

    for it in intents:
        decision = enforce_transfer(
            it, limits, state, live_balance=bucket_equity.get(it.from_venue, 0.0),
            now=now, allowlist=PREVIEW_ALLOWLIST,  # 판정 표시용 — 실집행 아님
        )
        auto_leg = it.from_venue == "UPBIT"  # Upbit KRW 출금만 API 자동, 그 외 레그 수동
        steps = MANUAL_ROUTES.get((it.from_venue, it.to_venue), [])
        logger.log("TREASURY", "treasury_dryrun_intent", {
            "from": it.from_venue, "to": it.to_venue, "amount": it.amount, "reason": it.reason,
            "would_allow": decision.allow, "violations": decision.violations,
            "auto_leg": "UPBIT_withdraw_krw" if auto_leg else None, "manual_steps": steps,
            "executed": False,
        })
        print(f"  intent {it.from_venue}→{it.to_venue} amount={it.amount:,.2f}"
              f" would_allow={decision.allow} auto_leg={auto_leg} violations={decision.violations}")

    print(f"note=DRY-RUN — 집행·상태갱신 없음. 실 ALLOWLIST 활성={bool(ALLOWLIST)}"
          f" (preview 판정 기준 {sorted(PREVIEW_ALLOWLIST)}). 실이체는 메타 검증·ALLOWLIST 후.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
