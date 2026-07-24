"""라이브 주문 절대 가드 — 실자금 주문 경로 전용 결정론 레이어.

배분비율(∑=1) 가드레일은 절대 금액을 모른다. 계좌가 커지면 비율이 맞아도 1회 주문
금액이 위험해질 수 있어, 명목금액(통화 절대값) 상한과 kill switch 를 주문 POST 직전에
강제한다. 페이퍼/모의 경로에는 붙이지 않는다(시뮬레이션은 자본 손실이 없다).

- kill switch: 지정 파일이 존재하면 전 주문 차단(사용자 수동 정지). 코드 변경·재배포 없이
  `touch <path>` 로 즉시 정지, `rm` 으로 해제.
- 1회 주문 상한: 단일 주문 명목금액이 상한을 넘으면 그 주문만 스킵.
- 일일 누적 상한: 당일 제출 명목금액 합이 상한을 넘으면 이후 주문 스킵. 상태 파일에
  (날짜, 누적액)만 기록 — 날짜가 바뀌면 자동 리셋.

호출 규약: kill_switch_active() 로 전면 차단 확인 → 주문별 check() 로 허용 여부 →
허용·제출 성공분만 charge() 로 당일 누적 반영.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class LiveCaps:
    max_order_notional: float  # 1회 주문 명목 상한 (통화 절대값)
    max_daily_notional: float  # 일일 누적 명목 상한
    kill_switch_path: Path  # 존재 = 전 주문 차단
    state_path: Path  # 일일 누적 상태 (날짜별)


class LiveGuard:
    def __init__(self, caps: LiveCaps) -> None:
        self.caps = caps

    def kill_switch_active(self) -> bool:
        return self.caps.kill_switch_path.exists()

    def _spent_today(self, today: date) -> float:
        p = self.caps.state_path
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            if s.get("day") == today.isoformat():
                return float(s.get("spent") or 0.0)
        return 0.0  # 파일 없음/날짜 경과 = 당일 누적 0

    def check(self, notional: float, today: date) -> str | None:
        """주문 1건이 상한을 넘는지 — 넘으면 사유(str), 허용이면 None."""
        cap = self.caps
        if notional > cap.max_order_notional:
            return f"over_order_cap notional={notional:.2f} cap={cap.max_order_notional}"
        spent = self._spent_today(today)
        if spent + notional > cap.max_daily_notional:
            return f"over_daily_cap spent={spent:.2f} notional={notional:.2f} cap={cap.max_daily_notional}"
        return None

    def charge(self, notional: float, today: date) -> None:
        """제출 성공 주문의 명목금액을 당일 누적에 반영(영속)."""
        spent = self._spent_today(today) + notional
        self.caps.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.caps.state_path.write_text(
            json.dumps({"day": today.isoformat(), "spent": round(spent, 2)}),
            encoding="utf-8",
        )
