"""Upbit 어댑터 2종 — 크립토 실계좌 체결(UpbitAdapter) + 자금 이체 자동 레그(UpbitTreasury).

둘 다 ccxt upbit 를 쓰지만 책임과 위험도가 다르다. 체결은 계좌 *안*에서 자산 구성만
바꾸므로(blast radius 가 계좌 안에서 닫힌다) 결정론 가드 하에 자율이고, 출금은 계좌
*밖*으로 자금을 내보내는 비가역 행위라 별도 이체 가드가 필요하다. 한 클래스에 합치면
그 경계가 흐려져 체결 경로의 버그가 출금 권한을 건드릴 수 있게 되므로 분리해 둔다.

- UpbitAdapter: 배분비율 → KRW 마켓 주문. 관측은 Binance 공개 USDT 채널을 그대로 승계.
- UpbitTreasury: 가용 KRW 조회 + 등록 계좌로 KRW 출금(TreasuryCapable).
"""

from __future__ import annotations

from datetime import datetime, timezone

from adapters.allocation import build_budget, project_to_executable, weights_from_quantities
from adapters.base import MarketAdapter, OrderResult, Position
from adapters.ccxt_adapter import BinanceDataFeed, _ensure_markets
from adapters.retry import with_retry

# Upbit KRW 마켓 최소 주문 금액(원). 이 미만의 배분 차이는 주문하지 않는다.
KRW_MIN_ORDER = 5_000.0


class UpbitAdapter(BinanceDataFeed, MarketAdapter):
    """크립토 실계좌 체결 — 주문·잔고는 Upbit KRW 마켓, 관측은 Binance 공개 USDT.

    **통화가 채널마다 다르다.** 관측(봉·현재가·뉴스)은 기존 크립토 경로와 한 글자도
    다르지 않은 USDT 기준이고, 체결·잔고·평가액만 KRW 다. 배분비율은 통화에 무관한
    비율이라 이 분리가 성립한다. 이렇게 두는 이유:

    1. 심볼 계보 — 봉·피처·메모리 pattern_key·가상 포트폴리오가 전부 `BTC/USDT` 로
       키잉돼 있다. 체결 거래소를 바꿨다고 관측 심볼을 `BTC/KRW` 로 갈면 축적된
       통계 표본이 다른 시계열로 갈라져 admission 반복 관측이 무효가 된다.
    2. 트리거 오탐 — 급변 판정(get_current_prices)까지 KRW 로 바꾸면 원달러 변동과
       김치 프리미엄이 코인 급변으로 잡히고, 저장된 참조가(USDT)와 단위가 어긋나
       전환 직후 첫 틱이 통째로 오발동한다.

    대가로 관측가와 실제 체결가가 환율·프리미엄만큼 벌어진다 — 배분 *비율* 결정에는
    영향이 없고, 그 괴리 자체를 관측에 넣는 것은 거시 예측이라 채택하지 않았다.
    """

    market = "CRYPTO"

    def __init__(
        self,
        api_key: str,
        secret: str,
        universe: list[str],  # 관측·배분 심볼 (USDT 표기) — 예: ["BTC/USDT", "ETH/USDT"]
        min_notional: float = KRW_MIN_ORDER,
        live_guard=None,  # 실자금 절대 금액 가드(LiveGuard). 없으면 미적용
    ) -> None:
        import ccxt.async_support as ccxt_async

        super().__init__()  # 관측 채널(Binance 메인넷 공개 — 키 불필요)
        self.ex = ccxt_async.upbit({"apiKey": api_key, "secret": secret, "timeout": 15000})
        self.universe = universe
        self.min_notional = min_notional
        self.live_guard = live_guard

    async def close(self) -> None:
        """aiohttp 세션 정리. 사용 후 반드시 호출."""
        await self.ex.close()
        await self.close_data()

    @staticmethod
    def to_krw_symbol(symbol: str) -> str:
        """관측 심볼(`BTC/USDT`) → 체결 심볼(`BTC/KRW`). 체결 경로에서만 쓴다."""
        return f"{symbol.split('/')[0]}/KRW"

    async def _snapshot(self) -> tuple[float, float, dict[str, float], dict[str, float]]:
        """(주문가능 KRW, 총 KRW, 보유수량, KRW 현재가) — 잔고 1회 + 티커 1회.

        주문에는 free 를 쓴다 — 미체결에 잠긴 KRW 로는 주문이 나가지 않는다(시장가
        체결이라 정상 상태에서는 free == total). 평가액에는 total 을 쓴다.
        """
        await _ensure_markets(self.ex)
        balance = await with_retry(self.ex.fetch_balance)
        free_krw = float(balance.get("free", {}).get("KRW") or 0)
        total_krw = float(balance.get("total", {}).get("KRW") or 0)
        krw_symbols = [self.to_krw_symbol(s) for s in self.universe]
        tickers = await with_retry(lambda: self.ex.fetch_tickers(krw_symbols))
        prices: dict[str, float] = {}
        qty: dict[str, float] = {}
        for symbol in self.universe:
            prices[symbol] = float(tickers[self.to_krw_symbol(symbol)]["last"])
            held = float(balance.get("total", {}).get(symbol.split("/")[0]) or 0)
            if held > 0:
                qty[symbol] = held
        return free_krw, total_krw, qty, prices

    async def get_budget(self, unit_prices: dict[str, float]):
        """분수 거래라 거래단위 제약은 없다 — 남는 건 최소 주문(5,000원)과 1회 상한.

        unit_prices(관측 USDT)는 쓰지 않는다. 계좌 통화가 KRW 라 단위가 어긋나는데,
        lot=0 이면 1단위 비중 자체가 성립하지 않아 필요가 없다.
        """
        free_krw, total_krw, qty, prices = await self._snapshot()
        equity = total_krw + sum(q * prices[s] for s, q in qty.items())
        return build_budget(
            "KRW",
            equity,
            free_krw,
            {},
            min_order=self.min_notional,
            max_order=self.live_guard.buy_cap(equity) if self.live_guard else None,
        )

    async def get_equity(self) -> float:
        """계좌 총 평가액(KRW). MDD 서킷 입력 — 통화가 바뀌므로 어댑터 전환 시 상태 리셋 필요."""
        _, total_krw, qty, prices = await self._snapshot()
        return total_krw + sum(q * prices[s] for s, q in qty.items())

    async def get_positions(self) -> list[Position]:
        """보유 포지션. symbol 은 관측 표기(USDT), **평가액 단위는 KRW**."""
        _, _, qty, prices = await self._snapshot()
        # 취득단가는 조회하지 않는다 — avg_price=0.0 은 "미상" 표기(크립토 공통)
        return [
            Position(symbol=s, quantity=q, avg_price=0.0, market_value=q * prices[s])
            for s, q in qty.items()
        ]

    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        now = datetime.now(timezone.utc)
        today = now.date()
        # kill switch — 사용자 수동 정지. 실자금 주문을 전면 차단(관측·결정은 이미 끝난 뒤).
        if self.live_guard and self.live_guard.kill_switch_active():
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error="kill_switch_active"
            )
        try:
            cash, total_krw, qty, prices = await self._snapshot()
            # 1회 매수 상한이 평가액에 비례하므로 여기서도 평가액이 필요하다(get_budget 과 동일 산식)
            equity = total_krw + sum(q * prices[s] for s, q in qty.items() if s in prices)
            # 집행가 타당성 검사는 여기에 두지 않는다 — 관측은 USDT, 체결가는 KRW 라
            # 두 값의 비는 환율이지 괴리가 아니다. 통화가 갈린 채로 비교하면 정상 주문이
            # 전부 오류로 걸린다. 검사하려면 먼저 환율로 같은 단위에 올려야 한다.
            # 분수 거래라 lot 제약이 없다 — 잘려나가는 건 최소 주문(5,000원)과 1회 상한뿐.
            plan = project_to_executable(
                weights, qty, cash, prices,
                min_notional=self.min_notional,
                max_order_notional=self.live_guard.buy_cap(equity)
                if self.live_guard else None,
            )
            final_qty = dict(qty)
            orders = []
            for it in plan.intents:
                notional = round(it.notional)  # KRW 는 원 단위 정수
                # 절대 금액 가드 — 1회/일일 명목 상한(KRW). 초과 주문은 그 건만 스킵.
                if self.live_guard:
                    # 투영은 1회 상한만 알고 일일 누적은 모른다 — 그 판정은 여기서.
                    reason = self.live_guard.check(notional, today, it.side, equity)
                    if reason:
                        orders.append({"symbol": it.symbol, "side": it.side, "skipped": reason})
                        plan.dropped[it.symbol] = reason
                        continue
                krw_symbol = self.to_krw_symbol(it.symbol)
                if it.side == "buy":
                    # Upbit 시장가 매수는 수량이 아니라 **금액(KRW)** 주문(ord_type=price).
                    # cost 를 명시해 수량×가격 환산 경로를 타지 않게 한다.
                    placed = await self.ex.create_order(
                        krw_symbol, "market", "buy", notional, params={"cost": notional}
                    )
                    filled_qty = None
                else:
                    # 보유량으로 상한 — 잔량 오차로 보유분보다 많이 팔면 주문 자체가 거부된다.
                    amount = min(float(it.qty or 0), qty.get(it.symbol, 0.0))
                    filled_qty = float(self.ex.amount_to_precision(krw_symbol, amount))
                    placed = await self.ex.create_order(krw_symbol, "market", "sell", filled_qty)
                if self.live_guard:
                    self.live_guard.charge(notional, today, it.side)  # 제출 성공분만
                delta = it.qty if it.side == "buy" else -(filled_qty or 0.0)
                final_qty[it.symbol] = final_qty.get(it.symbol, 0.0) + delta
                orders.append(
                    {
                        "symbol": it.symbol,
                        "side": it.side,
                        "qty": filled_qty,
                        "notional_krw": notional,
                        "order_id": placed.get("id"),
                    }
                )
            total = cash + sum(q * prices[s] for s, q in qty.items())
            return OrderResult(
                market=self.market,
                submitted_at=now,
                accepted=True,
                orders=orders,
                executed_weights=weights_from_quantities(final_qty, prices, total),
                dropped=plan.dropped,
            )
        except Exception as e:  # 주문 실패는 예외가 아니라 결과로 — 러너가 로그로 남긴다
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=str(e)[:300]
            )


class UpbitTreasury:
    """Upbit KRW 자금 이체(자동 레그). ccxt upbit 로 잔고 조회 + KRW 출금만 담당."""

    venue = "UPBIT"

    def __init__(self, api_key: str, secret: str) -> None:
        import ccxt.async_support as ccxt_async

        # timeout(ms) 명시 — 다른 브로커 어댑터 REST(15s)와 통일
        self.ex = ccxt_async.upbit({"apiKey": api_key, "secret": secret, "timeout": 15000})

    async def close(self) -> None:
        """aiohttp 세션 정리. 사용 후 반드시 호출."""
        await self.ex.close()

    async def withdrawable_krw(self) -> float:
        """출금 가능한 KRW 가용 잔고(free). 잠금·미체결분 제외."""
        balance = await with_retry(self.ex.fetch_balance)
        return float(balance.get("free", {}).get("KRW") or 0)

    async def withdraw_krw(self, amount: float) -> dict:
        """등록 계좌로 KRW 출금 — **실자금 이동, 비멱등**. 이체 가드 통과 후에만 호출.

        ccxt withdraw(code="KRW") 는 /withdraws/krw 로 라우팅되며 address 인자는 무시된다
        (KRW 목적지 = 거래소 KYC 등록 계좌 고정). 재시도 없음 — 이중 출금 방지.
        일부 계정은 Upbit 가 two_factor_type 를 요구할 수 있다(활성화 시 확인해 params 전달).
        """
        tx = await self.ex.withdraw("KRW", amount, "")
        info = tx.get("info") or {}
        return {
            "uuid": tx.get("id") or info.get("uuid"),
            "state": info.get("state"),
            "amount": float(tx.get("amount") or amount),
            "raw": info,
        }
