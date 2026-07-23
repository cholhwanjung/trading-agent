"""수동 이체 액추에이션 — API 없는 레그(증권↔은행)의 사람 집행 + 자동 검증 ([ADR-026] ⑥).

KIS 는 자금이동 실행 API 가 없어 증권→은행 레그는 사람이 앱에서 직접 옮긴다. 이 모듈은
그 레그를 '승인 게이트'가 아니라 '액추에이션'으로 다룬다 — 결정(금액·목적지)은 이미
결정론 가드(treasury.guard)가 내렸고, 사람은 물리적 이동만 수행한다.

핵심 안전 규약: 완료 판정은 **잔고 조회 검증**(`reconcile_verdict`)으로 하며, 사람이 입력한
confirmed_amount 를 맹신하지 않는다. 지시 문안(steps)은 고정 템플릿 — LLM 미개입.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from treasury.guard import TransferIntent

# 상태 전이: awaiting_user → user_confirmed → reconciled | mismatch (또는 expired)
AWAITING = "awaiting_user"
CONFIRMED = "user_confirmed"
RECONCILED = "reconciled"
MISMATCH = "mismatch"
PENDING = "pending"  # reconcile 검증 결과(무변화) — 상태는 유지

# 라우트별 사람 지시 템플릿 — grep 가능, 고정 문안(LLM 미개입).
MANUAL_ROUTES: dict[tuple[str, str], list[str]] = {
    ("KIS", "UPBIT"): [
        "KIS 앱/HTS: 증권계좌 → 본인 은행계좌로 출금",
        "은행 앱: Upbit 실명확인 입출금계좌로 이체",
    ],
    ("UPBIT", "KIS"): [
        "Upbit 앱: KRW 를 본인 은행계좌로 출금",
        "은행 앱: KIS 증권계좌로 입금",
    ],
}


@dataclass(frozen=True)
class ManualTransferInstruction:
    id: str
    created_at: str
    from_venue: str
    to_venue: str
    amount: float
    steps: list[str]
    reason: str
    status: str
    confirmed_amount: float | None = None
    confirmed_at: str | None = None


def _steps_for(from_venue: str, to_venue: str) -> list[str]:
    return MANUAL_ROUTES.get((from_venue, to_venue), [f"{from_venue} → {to_venue}: 본인 계좌 간 수동 이체"])


def reconcile_verdict(
    instruction: ManualTransferInstruction,
    dest_balance_before: float,
    dest_balance_now: float,
    tol: float = 0.02,
) -> str:
    """목적지 잔고 변화로 완료를 검증 — 사람 확인(confirmed_amount)을 맹신하지 않는다.

    - 무변화(≤ 기대액·tol) → PENDING (미도착/in-flight)
    - 기대액 ±tol(수수료 감안) 도달 → RECONCILED
    - 도착했으나 기대와 불일치 → MISMATCH (halt·재확인 대상)
    순수 함수 — I/O 없음.
    """
    received = dest_balance_now - dest_balance_before
    expected = instruction.amount
    if received <= expected * tol:
        return PENDING
    if expected * (1 - tol) <= received <= expected * (1 + tol):
        return RECONCILED
    return MISMATCH


class ManualLedger:
    """수동 이체 지시 원장 — JSON 영속. emit/confirm/reconcile 로 상태 전이."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        rows = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        self.instructions = [ManualTransferInstruction(**r) for r in rows]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(i) for i in self.instructions], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def _index(self, instruction_id: str) -> int:
        for i, inst in enumerate(self.instructions):
            if inst.id == instruction_id:
                return i
        raise ValueError(f"unknown instruction id={instruction_id}")

    def pending(self) -> list[ManualTransferInstruction]:
        """아직 정산 안 된 지시(awaiting_user·user_confirmed)."""
        return [i for i in self.instructions if i.status in (AWAITING, CONFIRMED)]

    def emit(self, intent: TransferIntent, now: datetime) -> ManualTransferInstruction:
        """가드를 통과한 이체 의도 → 구조화 수동 지시. status=awaiting_user 로 기록."""
        inst = ManualTransferInstruction(
            id=f"mt_{now:%Y%m%d%H%M%S}_{intent.from_venue}_{intent.to_venue}",
            created_at=now.isoformat(),
            from_venue=intent.from_venue,
            to_venue=intent.to_venue,
            amount=intent.amount,
            steps=_steps_for(intent.from_venue, intent.to_venue),
            reason=intent.reason,
            status=AWAITING,
        )
        self.instructions.append(inst)
        self._save()
        return inst

    def confirm(
        self, instruction_id: str, actual_amount: float, now: datetime
    ) -> ManualTransferInstruction:
        """사람이 '옮겼다' 표시 → status=user_confirmed. 완료 판정은 reconcile 이 잔고로 한다."""
        idx = self._index(instruction_id)
        inst = replace(
            self.instructions[idx],
            status=CONFIRMED,
            confirmed_amount=actual_amount,
            confirmed_at=now.isoformat(),
        )
        self.instructions[idx] = inst
        self._save()
        return inst

    def reconcile(
        self,
        instruction_id: str,
        dest_balance_before: float,
        dest_balance_now: float,
        tol: float = 0.02,
    ) -> str:
        """잔고 변화로 완료 검증 후 status 갱신. verdict 반환(로그용).
        RECONCILED·MISMATCH 는 종결 상태로 기록, PENDING 은 상태 유지(재검증 대기)."""
        idx = self._index(instruction_id)
        verdict = reconcile_verdict(
            self.instructions[idx], dest_balance_before, dest_balance_now, tol
        )
        if verdict in (RECONCILED, MISMATCH):
            self.instructions[idx] = replace(self.instructions[idx], status=verdict)
            self._save()
        return verdict
