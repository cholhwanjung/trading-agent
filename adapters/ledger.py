"""시장별 현금 장부 — 계좌 하나를 여러 시장이 나눠 쓸 때 각자의 자금을 분리 추적한다.

브로커 계좌는 하나지만 시장은 서로 다른 거래소를 쓰는 것처럼 취급한다. 보유 종목은
어느 시장 것인지 응답에 드러나므로 문제가 없고, 나뉘지 않는 것은 **현금**뿐이라 그것만
장부로 관리한다.

지분(share)은 **최초 배정과 외부 입출금 분배**에만 쓰인다. 그 뒤로는 각 시장이 자기
매매로만 현금을 움직이므로, 한 시장의 손익이 다음 스텝에 다른 시장의 예산으로 넘어가지
않는다 — 지분을 매 스텝 총자산에 곱하면 그렇게 된다(이긴 시장에서 돈을 빼 진 시장에
넣는 정기 리밸런스가 된다).

장부는 계좌 현금의 **수준**이 아니라 **변화량**으로 움직인다. 계좌 현금을 알려 주는
브로커 필드는 조회 경로마다 결제 반영 시점이 다르다 — 어떤 필드는 매도대금을 체결 즉시
싣고 어떤 필드는 결제일에야 싣는다. 두 시장이 서로 다른 필드를 같은 장부의 수준에 대면
그 시차가 "계좌에서 돈이 나갔다"로 읽히고, 있지도 않은 출금이 양쪽 시장에서 깎인다.
그래서 매매한 시장은 **같은 필드를 주문 직전·직후에 읽은 차이**만 자기 장부에 반영하고,
수준의 불일치는 며칠 지켜본 뒤에야 외부 입출금으로 확정한다. 체결 금액을 따로 더하고
빼지 않으므로 부분체결·수수료·호가 차이가 장부와 실제를 벌려 놓지 못하는 성질은 그대로다.

외화로 결제되는 시장은 한 가지가 더 있다. 그 시장의 매도대금은 계좌 통화로 돌아오지
않고 외화 예수금으로 남으며, 계좌 통화는 매수 결제 때 외화 쪽으로 넘어간다. 이 장부는
계좌 통화만 담으므로, 그 이동은 외화 현금성 잔액(예수금 + 미결제 정산분)이 **매매 없이**
변한 만큼으로 읽는다(`sync_foreign`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from adapters.allocation import split_account

# 계좌 현금 대비 이 비율을 넘는 불일치는 매매가 아니라 외부 입출금 후보로 본다.
# 수수료·이자·환율 잔돈은 이 아래로 떨어져 장부에 손대지 않는다.
FLOW_RATIO = 0.001
# 불일치가 같은 크기로 이 기간(달력일) 남아 있어야 외부 입출금으로 확정한다. 결제 시차가
# 만든 불일치는 결제와 함께 사라지고 진짜 입출금은 남는다 — 주말·연휴를 낀 결제도 이 안에
# 끝난다. 늦게 확정하는 비용은 입금이 며칠 노는 것뿐이지만, 성급히 나누면 있지도 않은
# 입금이 상대 시장의 평가액 고점을 밀어 올려 그 뒤 계속 가짜 낙폭으로 남는다.
FLOW_CONFIRM_DAYS = 5
# 외화 현금성 잔액의 변화 중 계좌 통화와의 이동으로 보는 하한(그 외화 단위). 실제 이동은
# 1주 값 이상이고, 이 아래는 미결제분의 수수료 조정·반올림·소액 배당이다 — 배당은 그
# 시장의 소득이므로 장부를 건드리지 않아야 평가액에 남는다.
TRANSFER_MIN = 1.0


@dataclass
class AccountLedger:
    """시장별 현금 장부. 파일 하나에 계좌 하나."""

    path: Path
    shares: dict[str, float]  # 최초 배정·외부 입출금 분배 비율 (합 1)
    account_key: str = ""  # 계좌 지문 — 바뀌면 장부를 버린다
    last_reconcile: dict = field(default_factory=dict)  # 가장 최근 수준 점검의 상태
    # 이 프로세스에서 장부가 움직인 내역. 수준 점검은 한 스텝에 여러 번 불리고 마지막 호출이
    # `last_reconcile` 을 덮으므로, 분배·이체처럼 **일어난 일**은 여기 따로 쌓는다.
    events: list[dict] = field(default_factory=list)

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if self.account_key and state.get("account_key") != self.account_key:
            return {}  # 다른 계좌의 장부 — 물려받으면 남의 현금을 자기 것으로 본다
        state["cash"] = {m: float(v) for m, v in (state.get("cash") or {}).items()}
        return state

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "account_key": self.account_key,
            "cash": {m: round(v, 2) for m, v in state["cash"].items()},
        }
        for key in ("pending_flow", "foreign"):
            if state.get(key):
                out[key] = state[key]
        self.path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    def _seeded(self, state: dict) -> bool:
        return set(state.get("cash") or {}) == set(self.shares)

    def peer_of(self, market: str) -> str:
        """이 계좌를 함께 쓰는 다른 시장. 장부는 두 시장 계좌를 전제한다."""
        return next(m for m in self.shares if m != market)

    def cash_of(self, market: str) -> float | None:
        """이 시장의 장부 값(수준 점검 없이). 아직 배정 전이면 None."""
        state = self._read()
        return state["cash"].get(market) if self._seeded(state) else None

    def total_cash(self) -> float:
        """두 시장 장부의 합 — 계좌 전체를 셀 때 쓴다. 배정 전이면 0."""
        state = self._read()
        return sum(state["cash"].values()) if self._seeded(state) else 0.0

    def cash_for(
        self,
        market: str,
        account_cash: float,
        own_held: float,
        peer_held: float,
        today: date | None = None,
    ) -> float:
        """이 시장이 쓸 수 있는 현금. 장부가 없으면 지분대로 최초 배정한다.

        보유 평가액(계좌 통화)은 최초 배정에만 쓰인다 — 이미 보유가 있는 상태에서
        시작하면 그 몫을 예산에서 빼야 지분이 맞기 때문이다.

        계좌 현금과 장부 합의 불일치는 **바로 나누지 않는다.** 처음 본 날을 적어 두고 같은
        크기로 `FLOW_CONFIRM_DAYS` 남아 있을 때만 외부 입출금으로 확정해 지분대로 나눈다.
        """
        today = today or date.today()
        state = self._read()
        if not self._seeded(state):  # 최초 실행 또는 시장 구성 변경 → 재배정
            held = {market: own_held, self.peer_of(market): peer_held}
            total = account_cash + own_held + peer_held
            cash = {m: total * self.shares[m] - held.get(m, 0.0) for m in self.shares}
            self._write({"cash": cash})
            self.last_reconcile = {"action": "seed", "account_cash": round(account_cash, 2)}
            self.events.append(dict(self.last_reconcile))
            return cash.get(market, 0.0)

        cash = state["cash"]
        drift = account_cash - sum(cash.values())
        tol = abs(account_cash) * FLOW_RATIO
        pending = state.get("pending_flow")
        if abs(drift) <= tol:
            if pending:  # 지켜보던 불일치가 사라졌다 — 결제 시차였다
                state.pop("pending_flow")
                self._write(state)
                self.events.append({"action": "flow_cleared", "drift": pending["drift"]})
            self.last_reconcile = {"action": "ok", "drift": round(drift, 2)}
            return cash.get(market, 0.0)

        same = pending is not None and abs(drift - pending["drift"]) <= tol
        if not same:  # 새로 나타났거나 크기가 달라졌다 — 그날부터 다시 센다
            state["pending_flow"] = {"first_seen": today.isoformat(), "drift": round(drift, 2)}
            self._write(state)
            self.events.append({"action": "flow_pending", **state["pending_flow"]})
        elif (today - date.fromisoformat(pending["first_seen"])).days >= FLOW_CONFIRM_DAYS:
            # 외부 입출금 — 어느 시장의 매매로도 결제로도 설명되지 않은 채 남았다.
            cash = {m: v + drift * self.shares[m] for m, v in cash.items()}
            state["cash"] = cash
            state.pop("pending_flow")
            self._write(state)
            self.last_reconcile = {"action": "flow_confirmed", "drift": round(drift, 2)}
            self.events.append(
                {**self.last_reconcile, "split": {m: round(drift * s, 2) for m, s in self.shares.items()}}
            )
            return cash.get(market, 0.0)
        self.last_reconcile = {
            "action": "flow_pending",
            "drift": round(drift, 2),
            "since": state["pending_flow"]["first_seen"],
        }
        return cash.get(market, 0.0)

    def settle(self, market: str, cash_before: float, cash_after: float) -> None:
        """매매 직후 호출 — 주문 직전·직후에 **같은 필드로** 읽은 계좌 현금의 차이를 반영한다.

        계좌 락이 같은 시장의 동시 실행을 막으므로 그 사이의 현금 변화는 이 시장의 매매(와
        그 수수료)뿐이다. 수준(`계좌현금 − 다른 시장 장부`)으로 맞추지 않는 이유는, 지켜보는
        중인 불일치나 상대 시장의 아직 반영 안 된 결제가 있으면 그것까지 이 시장이 삼키기
        때문이다. 주문이 실제로 나간 경우에만 부른다.
        """
        state = self._read()
        if not self._seeded(state):
            return  # 아직 배정 전 — cash_for 가 다음 조회에서 지분대로 배정한다
        delta = cash_after - cash_before
        state["cash"][market] += delta
        self._write(state)
        self.events.append({"action": "settle", "market": market, "delta": round(delta, 2)})

    def sync_foreign(
        self, market: str, pend: float, qty: dict[str, float], rate: float
    ) -> float:
        """외화 시장의 런 시작 시 호출 — 계좌 통화와 외화 사이에 넘어간 돈을 장부에 반영한다.

        pend 는 그 시장의 외화 현금성 잔액(외화 예수금 + 미결제 정산분, 외화 단위)이다. 매매가
        없던 구간에 이 값이 늘었다면 계좌 통화가 외화 쪽으로 넘어간 것이고(계좌 통화로 산
        매수의 결제), 줄었다면 돌아온 것이다(환전). 매도대금이 결제돼 예수금이 되는 것은
        미결제분이 예수금으로 자리만 옮기는 것이라 pend 가 변하지 않는다.

        직전 스냅샷 이후 **보유 수량이 바뀌었으면** 그 구간에 매매가 있었다는 뜻이다(수동
        주문, 체결 직후 스냅샷을 못 남기고 끝난 런, 뒤늦게 체결된 지정가). 매매는 pend 를
        움직이지만 계좌 통화를 옮기지 않으므로, 이체로 읽지 않고 스냅샷만 새로 뜬다.

        돌려주는 값은 장부에서 뺀 금액(계좌 통화). 같은 런에서 여러 번 불려도 첫 호출만
        움직인다 — 스냅샷이 그 자리에서 갱신되기 때문이다.
        """
        state = self._read()
        if not self._seeded(state):
            return 0.0
        snap = (state.get("foreign") or {}).get(market)
        clean = {s: float(q) for s, q in qty.items() if q}
        moved = 0.0
        if snap is None or snap.get("qty") != clean:
            reason = "first" if snap is None else "holdings_changed"
            self.events.append({"action": "resnap", "market": market, "reason": reason})
        else:
            delta = pend - float(snap["pend"])
            if abs(delta) >= TRANSFER_MIN:
                moved = delta * rate
                state["cash"][market] -= moved
                self.events.append(
                    {"action": "transfer", "market": market, "foreign_delta": round(delta, 2),
                     "rate": rate, "moved": round(moved, 2)}
                )
            elif abs(delta) < 0.005:
                return 0.0  # 변한 것이 없다(저장 시 반올림 차이뿐) — 파일을 다시 쓰지 않는다
        state.setdefault("foreign", {})[market] = {"pend": round(pend, 4), "qty": clean}
        self._write(state)
        return moved

    def snapshot_foreign(self, market: str, pend: float, qty: dict[str, float]) -> None:
        """체결 직후 호출 — 장부는 두고 스냅샷만 갱신한다.

        매매가 pend 를 움직인 것이 다음 런에서 이체로 읽히지 않게 한다.
        """
        state = self._read()
        if not self._seeded(state):
            return
        state.setdefault("foreign", {})[market] = {
            "pend": round(pend, 4),
            "qty": {s: float(q) for s, q in qty.items() if q},
        }
        self._write(state)


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


def foreign_funds(cash: float, own_side: float, pend: float) -> tuple[float, float]:
    """외화로 결제되는 시장의 (쓸 수 있는 현금, 예산). 계좌 통화 단위.

    own_side 는 그 시장의 외화 측 전체(보유 + 외화 예수금 + 미결제 정산분)이고 pend 는 그중
    현금성 부분이다. 외화 측은 그 시장의 매매로만 생기므로 보유 종목과 같이 전부 자기 몫이다.
    보유만 세면 매도할 때마다 그 대금이 평가액에서 사라져 손실로 읽힌다.
    """
    return max(0.0, cash + pend), max(cash, 0.0) + own_side


# 매매 전후 평가액 차이의 허용 한도. 매매는 자산의 형태만 바꾸므로 차이는 수수료·호가와
# 두 조회 사이의 시세 움직임뿐이어야 한다. 하한은 계좌 통화(원) 금액이다 — 소액 계좌에서는
# 비율만으로 한도가 1건 수수료보다 작아진다.
CONSERVATION_RATIO = 0.003
CONSERVATION_FLOOR = 1_000.0


def conservation_record(pre: float, post: float) -> dict:
    """매매 직전·직후 평가액의 연속성 — 로깅·통지용.

    평가액은 낙폭 서킷의 입력이다. 매매만으로 이 값이 뛰면 산식이 브로커 필드의 어떤
    움직임(결제 시차, 예수금 통화 전환 등)을 놓치고 있다는 뜻이고, 그 불연속은 손익이
    아닌데도 낙폭으로 쌓인다. 산식이 맞는지를 매 매매에서 그 소비 지점으로 확인한다.
    """
    diff = post - pre
    limit = max(abs(pre) * CONSERVATION_RATIO, CONSERVATION_FLOOR)
    return {
        "pre": round(pre, 2),
        "post": round(post, 2),
        "diff": round(diff, 2),
        "limit": round(limit, 2),
        "breach": abs(diff) > limit,
    }


def split_record(
    currency: str,
    ledger: AccountLedger | None,
    market: str,
    account_cash: float,
    own_held: float,
    peer_held: float,
    cash: float,
    equity: float,
    account_total: float | None = None,
) -> dict:
    """예산 산출 내역 — 로깅 전용. 장부가 어긋나면 여기서만 보인다."""
    total = account_cash + own_held + peer_held if account_total is None else account_total
    return {
        "currency": currency,
        "market": market,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "held": round(own_held, 2),
        "peer_held": round(peer_held, 2),
        "account_cash": round(account_cash, 2),
        "account_total": round(total, 2),
        "reconcile": (ledger.last_reconcile if ledger else {"action": "no_ledger"}),
    }
