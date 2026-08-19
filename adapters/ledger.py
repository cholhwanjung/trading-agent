"""시장별 현금 장부 — 계좌 하나를 여러 시장이 나눠 쓸 때 각자의 자금을 분리 추적한다.

브로커 계좌는 하나지만 시장은 서로 다른 거래소를 쓰는 것처럼 취급한다. 보유 종목은
어느 시장 것인지 응답에 드러나므로 문제가 없고, 나뉘지 않는 것은 **현금**뿐이라 그것만
장부로 관리한다.

지분(share)은 **최초 배정과 외부 입출금 분배**에만 쓰인다. 그 뒤로는 각 시장이 자기
매매로만 현금을 움직이므로, 한 시장의 손익이 다음 스텝에 다른 시장의 예산으로 넘어가지
않는다 — 지분을 매 스텝 총자산에 곱하면 그렇게 된다(이긴 시장에서 돈을 빼 진 시장에
넣는 정기 리밸런스가 된다).

장부는 누적이 아니라 **화해**로 유지한다. 매매 뒤 계좌 현금에서 다른 시장의 장부를 빼
자기 몫으로 삼는다 — 계좌 락이 시장 간 동시 실행을 막으므로 그 사이의 현금 변화는 전부
매매한 시장의 것이다. 체결 금액을 따로 더하고 빼지 않으므로 부분체결·수수료·호가 차이가
장부와 실제를 벌려 놓지 못한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from adapters.allocation import split_account

# 계좌 현금 대비 이 비율을 넘는 불일치는 매매가 아니라 외부 입출금으로 본다.
# 수수료·이자·환율 잔돈은 이 아래로 떨어져 매매한 시장이 자기 비용으로 흡수한다.
FLOW_RATIO = 0.001


@dataclass
class AccountLedger:
    """시장별 현금 장부. 파일 하나에 계좌 하나."""

    path: Path
    shares: dict[str, float]  # 최초 배정·외부 입출금 분배 비율 (합 1)
    account_key: str = ""  # 계좌 지문 — 바뀌면 장부를 버린다
    last_reconcile: dict = field(default_factory=dict)  # 로깅용 관측치

    def _load(self) -> dict[str, float]:
        if not self.path.exists():
            return {}
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if self.account_key and state.get("account_key") != self.account_key:
            return {}  # 다른 계좌의 장부 — 물려받으면 남의 현금을 자기 것으로 본다
        return {m: float(v) for m, v in (state.get("cash") or {}).items()}

    def _save(self, cash: dict[str, float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"account_key": self.account_key, "cash": {m: round(v, 2) for m, v in cash.items()}},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    def peer_of(self, market: str) -> str:
        """이 계좌를 함께 쓰는 다른 시장. 장부는 두 시장 계좌를 전제한다."""
        return next(m for m in self.shares if m != market)

    def cash_for(
        self, market: str, account_cash: float, own_held: float, peer_held: float
    ) -> float:
        """이 시장이 쓸 수 있는 현금. 장부가 없으면 지분대로 최초 배정한다.

        보유 평가액(계좌 통화)은 최초 배정에만 쓰인다 — 이미 보유가 있는 상태에서
        시작하면 그 몫을 예산에서 빼야 지분이 맞기 때문이다.
        """
        cash = self._load()
        known = set(self.shares)
        if set(cash) != known:  # 최초 실행 또는 시장 구성 변경 → 재배정
            held = {market: own_held, self.peer_of(market): peer_held}
            total = account_cash + own_held + peer_held
            cash = {m: total * self.shares[m] - held.get(m, 0.0) for m in known}
            self._save(cash)
            self.last_reconcile = {"action": "seed", "account_cash": round(account_cash, 2)}
            return cash.get(market, 0.0)

        drift = account_cash - sum(cash.values())
        if abs(drift) > abs(account_cash) * FLOW_RATIO:
            # 외부 입출금 — 어느 시장의 매매로도 설명되지 않는다. 지분대로 나눈다.
            cash = {m: v + drift * self.shares[m] for m, v in cash.items()}
            self._save(cash)
            self.last_reconcile = {"action": "external_flow", "drift": round(drift, 2)}
        else:
            self.last_reconcile = {"action": "ok", "drift": round(drift, 2)}
        return cash.get(market, 0.0)

    def settle(self, market: str, account_cash: float) -> None:
        """매매 직후 호출 — 계좌 현금의 변화를 전부 이 시장 장부에 반영한다.

        계좌 락이 시장 간 동시 실행을 막으므로 직전 화해 이후의 현금 변화는 이 시장의
        매매(와 그 수수료)뿐이다. 주문이 실제로 나간 경우에만 부른다 — 주문 0건인 스텝에서
        부르면 그 사이의 외부 입금까지 이 시장이 통째로 삼킨다.
        """
        cash = self._load()
        if not cash:
            return  # 아직 배정 전 — cash_for 가 다음 조회에서 지분대로 배정한다
        others = sum(v for m, v in cash.items() if m != market)
        cash[market] = account_cash - others
        self._save(cash)


def market_funds(
    ledger: AccountLedger | None,
    market: str,
    account_cash: float,
    own_held: float,
    peer_held: float,
    share: float,
) -> tuple[float, float]:
    """이 시장의 (쓸 수 있는 현금, 예산). 계좌 통화 단위.

    장부가 있으면 장부가 예산을 정한다 — 각 시장이 자기 매매로만 현금을 움직이므로
    손익이 시장을 넘지 않는다. 장부가 없는 구성(계좌 단독 사용, 모의)에서는 총자산을
    지분으로 나눈다.

    브로커 여력은 **쓸 수 있는 금액**만 자른다. 예산까지 자르면 미결제 등으로 여력이
    일시적으로 좁아진 날 이 시장의 평가액이 줄어든 것처럼 보여 MDD 서킷이 오발동한다.
    """
    if ledger is None:
        return split_account(account_cash + own_held + peer_held, share, own_held, account_cash)
    cash = ledger.cash_for(market, account_cash, own_held, peer_held)
    return max(0.0, min(cash, account_cash)), max(cash, 0.0) + own_held


def split_record(
    currency: str,
    ledger: AccountLedger | None,
    market: str,
    account_cash: float,
    own_held: float,
    peer_held: float,
    cash: float,
    equity: float,
) -> dict:
    """예산 산출 내역 — 로깅 전용. 장부가 어긋나면 여기서만 보인다."""
    return {
        "currency": currency,
        "market": market,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "held": round(own_held, 2),
        "peer_held": round(peer_held, 2),
        "account_cash": round(account_cash, 2),
        "account_total": round(account_cash + own_held + peer_held, 2),
        "reconcile": (ledger.last_reconcile if ledger else {"action": "no_ledger"}),
    }
