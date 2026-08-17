"""라이브 주문 절대 가드 — 실자금 주문 경로 전용 결정론 레이어.

배분비율(∑=1) 가드레일은 절대 금액을 모른다. 계좌가 커지면 비율이 맞아도 1회 주문
금액이 위험해질 수 있어, 명목금액(통화 절대값) 상한과 kill switch 를 주문 POST 직전에
강제한다. 페이퍼/모의 경로에는 붙이지 않는다(시뮬레이션은 자본 손실이 없다).

- kill switch: 지정 파일이 존재하면 전 주문 차단(사용자 수동 정지). 코드 변경·재배포 없이
  `touch <path>` 로 즉시 정지, `rm` 으로 해제. 매도도 함께 막는다 — 사람이 명시적으로
  내린 정지라, 아래 매수/매도 비대칭의 근거(추론 대신 사실)와 층이 다르다.
- 1회 주문 상한: 단일 **매수**의 명목금액이 상한을 넘으면 그 주문만 스킵.
- 일일 누적 상한: 당일 제출 **매수** 명목금액 합이 상한을 넘으면 이후 매수 스킵. 상태
  파일에 (날짜, 누적액)만 기록 — 날짜가 바뀌면 자동 리셋.

**상한은 매수에만 건다.** long-only 라 매도 수량의 상한은 보유량 자체이고, 잘못 나가도
결과는 '현금'이라 원금을 잃는 방향이 아니다. 반대로 상한이 매도를 막으면 위험을 줄이려는
바로 그 순간에 그것을 못 하게 된다. 게다가 아래 비중 상한은 평가액에 비례하므로 하락장에서
같이 좁아진다 — 방어가 가장 급할 때 방어 속도가 가장 느려지는 셈이다. 매수는 반대라 상한이
그대로 필요하다. **이 비대칭은 long-only 를 전제로 한다** — 공매도를 열면 '매도'가 무한
익스포저가 되므로 이 규칙을 먼저 되돌려야 한다.

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
    max_order_notional: float  # 1회 매수 명목 상한 (통화 절대값) — 천장
    max_daily_notional: float  # 일일 누적 매수 명목 상한
    kill_switch_path: Path  # 존재 = 전 주문 차단
    state_path: Path  # 일일 누적 상태 (날짜별)
    #: 평가액 대비 1회 매수 상한(비중). None = 절대 천장만 적용.
    #: 절대값만 쓰면 입금할 때마다 사람이 다시 정해야 하고, 비율만 쓰면 계좌가 커질수록
    #: 한 건의 피해 규모가 같이 커진다. 둘 중 작은 값을 쓰면 계좌가 작을 때도 보호되고
    #: 커져도 천장이 남는다.
    max_order_ratio: float | None = None


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

    def buy_cap(self, equity: float | None = None) -> float:
        """1회 매수 명목 상한 = min(절대 천장, 평가액 × 비율).

        평가액은 매 스텝 브로커에서 직접 읽는 값이라 **추론이 아니다** — 입출금을 감지해
        상한을 푸는 것(감지 버그 = 상한 해제)과는 층이 다르다. 여기서 상한은 사용자가
        선언한 두 숫자의 함수일 뿐이고, 입금·출금·손익 어느 쪽으로 평가액이 움직여도
        사람이 다시 정해줄 필요가 없다.

        평가액을 모르면(조회 실패) 천장만 적용한다 — 모른다는 이유로 상한을 넓히지 않는다.
        """
        cap = self.caps.max_order_notional
        if self.caps.max_order_ratio and equity and equity > 0:
            cap = min(cap, equity * self.caps.max_order_ratio)
        return cap

    def check(
        self,
        notional: float,
        today: date,
        side: str = "buy",
        equity: float | None = None,
    ) -> str | None:
        """주문 1건이 상한을 넘는지 — 넘으면 사유(str), 허용이면 None.

        매도는 항상 허용한다(모듈 docstring 의 매수/매도 비대칭).
        """
        if side == "sell":
            return None
        cap = self.buy_cap(equity)
        if notional > cap:
            return f"over_order_cap notional={notional:.2f} cap={cap:.2f}"
        spent = self._spent_today(today)
        limit = self.caps.max_daily_notional
        if spent + notional > limit:
            return f"over_daily_cap spent={spent:.2f} notional={notional:.2f} cap={limit}"
        return None

    def charge(self, notional: float, today: date, side: str = "buy") -> None:
        """제출 성공 **매수**의 명목금액을 당일 누적에 반영(영속).

        매도를 누적에 넣으면 위험을 줄인 만큼 그날 남은 매수 여력이 줄어드는데, 그것은
        상한이 재려던 것(신규 익스포저)이 아니다.
        """
        if side == "sell":
            return
        spent = self._spent_today(today) + notional
        self.caps.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.caps.state_path.write_text(
            json.dumps({"day": today.isoformat(), "spent": round(spent, 2)}),
            encoding="utf-8",
        )
