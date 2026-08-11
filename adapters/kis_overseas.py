"""미국 주식 어댑터 — KIS 해외주식 (모의/실전 겸용).

- 인증·스로틀은 국내 어댑터와 KISSession 공유. 같은 앱 키면 토큰 캐시 파일도 공유할 것.
- 시세: dailyprice(실전/모의 공통 TR) — 수정주가 기준. 시세계 거래소 코드(NAS)와
  주문계 코드(NASD)가 달라 내부에서 매핑한다.
- 주문: 미국 정규장에 순수 시장가가 없다(모의는 지정가만) → 현재가에 버퍼를 더한
  '체결형 지정가'(marketable limit)로 시장가를 대용한다. 정수 주 단위만 가능.
- 지정가 미체결 잔량이 남을 수 있어 주문 전 미체결 심볼은 제외(중복 주문 방지).
- 매수여력: **통합증거금 반영 '환전이후주문가능금액'(USD)** — 원화 담보를 자동환전한
  주문 여력이라, USD 예수금이 없어도 원화만으로 매수 가능(수동 환전 단계 불필요).
- 평가액(MDD 입력): **원화 총자산(tot_asst_amt)** — 원화 담보로 산 미국 자산은 환율
  손익에 노출되므로, USD 가 아닌 KRW 로 계상해 서킷이 FX 낙폭까지 포착하게 한다.
  (주의: 향후 KR 국내도 같은 실계좌로 오면 원화 담보를 KR/US 가 나눠 써야 함 — 지금은
  계좌 분리라 무관.)
- 뉴스: KIS 무료 원천 미정 — 빈 리스트(관측 배선 시 채널 별도 결정).
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from adapters.allocation import compute_order_deltas
from adapters.base import (
    Bar,
    MarketAdapter,
    NewsItem,
    OrderResult,
    Position,
    observation_window,
)
from adapters.kis import PAPER_BASE, REAL_BASE, KISSession, paginate_daily

if TYPE_CHECKING:
    from risk.live import LiveGuard

# 주문계(NASD) → 시세계(NAS) 거래소 코드. 현 유니버스는 나스닥 단일 — NYSE/AMEX 종목
# 편입 시 심볼별 거래소 매핑으로 확장해야 한다.
QUOTE_EXCD = {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}

# 미국 주문 tr_id. 모의는 지정가(00)만 지원 — ORD_DVSN 은 항상 "00".
ORDER_TR = {
    ("real", "buy"): "TTTT1002U",
    ("real", "sell"): "TTTT1006U",
    ("demo", "buy"): "VTTT1002U",
    ("demo", "sell"): "VTTT1001U",
}
BALANCE_TR = {"real": "TTTS3012R", "demo": "VTTS3012R"}
PRESENT_TR = {"real": "CTRP6504R", "demo": "VTRP6504R"}
NCCS_TR = {"real": "TTTS3018R", "demo": "VTTS3018R"}  # 미체결 조회
PSAMOUNT_TR = {"real": "TTTS3007R", "demo": "VTTS3007R"}  # 매수가능금액(통합증거금 여력)

LIMIT_BUFFER = 0.005  # marketable limit 버퍼 — 현재가 대비 0.5%


class KISOverseasAdapter(MarketAdapter):
    market = "US"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        account: str,  # "12345678-01" (계좌 8자리-상품코드 2자리)
        universe: list[str],  # 예: ["AAPL", "MSFT"] — 나스닥 종목
        token_cache: Path,
        mode: str = "demo",  # "demo"(모의) | "real"(실전 — 실자금)
        exchange: str = "NASD",
        min_notional: float = 10.0,  # USD
        limit_buffer: float = LIMIT_BUFFER,
        live_guard: LiveGuard | None = None,  # 실전 절대 금액 가드(모의는 None)
        alpaca_key: str | None = None,  # US 뉴스 원천(체결과 분리 — KIS 는 무료 뉴스 없음)
        alpaca_secret: str | None = None,
        sec_user_agent: str | None = None,  # SEC 공시·재무 조회 식별자 (키 아님)
    ) -> None:
        assert mode in ("demo", "real"), f"mode={mode!r} — 'demo' 또는 'real'"
        assert exchange in QUOTE_EXCD, f"exchange={exchange!r} — {sorted(QUOTE_EXCD)}"
        self.session = KISSession(
            app_key, app_secret, Path(token_cache),
            PAPER_BASE if mode == "demo" else REAL_BASE,
        )
        self.cano, _, self.prdt = account.partition("-")
        self.universe = universe
        self.mode = mode
        self.exchange = exchange
        self.excd = QUOTE_EXCD[exchange]
        self.min_notional = min_notional
        self.limit_buffer = limit_buffer
        self.live_guard = live_guard
        self._alpaca_key = alpaca_key
        self._alpaca_secret = alpaca_secret
        self._sec_user_agent = sec_user_agent

    async def close(self) -> None:
        await self.session.close()

    # ── 시세 ──

    @staticmethod
    def _parse_daily(rows: list[dict], start: date, end: date) -> list[Bar]:
        """dailyprice output2 행 → [start, end] 윈도우 Bar 오름차순 (당일 봉 차단)."""
        bars = []
        for r in rows:
            if not r.get("xymd"):
                continue  # 빈 placeholder 행 방어 (국내 API 와 동일 습성)
            day = datetime.strptime(r["xymd"], "%Y%m%d").date()
            if start <= day <= end:
                bars.append(
                    Bar(
                        day=day,
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["clos"]),
                        volume=float(r.get("tvol") or 0),
                    )
                )
        return sorted(bars, key=lambda b: b.day)

    async def _daily_page(
        self, symbol: str, start: date, cursor: date, bymd: str = ""
    ) -> list[Bar]:
        """[start, cursor] 일봉 1페이지 — 최대 ~100행, 기준일에서 과거로.

        bymd 공란은 '최근부터'. 응답 자체는 상한을 안 지키므로 [start, cursor] 필터로
        당일 봉을 막는다(파싱에서 재확인).
        """
        data = await self.session.get(
            "/uapi/overseas-price/v1/quotations/dailyprice",
            tr_id="HHDFS76240000",  # 실전/모의 공통
            params={
                "AUTH": "",
                "EXCD": self.excd,
                "SYMB": symbol,
                "GUBN": "0",  # 일봉
                "BYMD": bymd,
                "MODP": "1",  # 수정주가
            },
        )
        return self._parse_daily(data.get("output2") or [], start, cursor)

    async def _fetch_bars(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        # 관측·주문 경로 — 창이 짧아(최근 N거래일) 1회 조회(최대 ~100행)로 충분
        return {s: await self._daily_page(s, start, end) for s in symbols}

    async def _fetch_bars_history(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        """장기 창 전용 — ~100행 상한을 커서 페이지네이션으로 넘긴다(국면·feature 입력).

        300일 창은 ~205거래일이라 1회 조회로는 절반도 안 온다. 상한 t-1 은 호출부가 정한
        end 를 그대로 쓰고 파싱에서 재확인하므로 페이지를 이어도 유지된다.

        첫 페이지는 BYMD 공란 — 관측 경로와 **같은 요청 형태**다. 이어받는 페이지의 기준일은
        직전 페이지 최소일−1 이라 휴장일에 떨어질 수 있는데, 그때 응답이 어떻게 오는지 원천
        보증이 없다. 첫 페이지를 검증된 형태로 두면 최악의 경우에도 1회 조회분은 확보된다.
        """

        async def page(symbol: str, cursor: date) -> list[Bar]:
            bymd = "" if cursor == end else cursor.strftime("%Y%m%d")
            return await self._daily_page(symbol, start, cursor, bymd)

        return {
            s: await paginate_daily(lambda c, s=s: page(s, c), start, end) for s in symbols
        }

    async def get_news(self, symbols: list[str], asof_day: date) -> list[NewsItem]:
        # 체결은 KIS, 뉴스는 Alpaca(무료) — venue 분리. 키 없으면 빈 리스트.
        from adapters.edgar import fetch_edgar_filings
        from adapters.news_us import fetch_us_news

        start, end = observation_window(asof_day)
        news = await fetch_us_news(self._alpaca_key, self._alpaca_secret, symbols, start, end)
        # 공시는 뉴스와 별개 원천(키 불필요) — 실계좌 경로에서도 이벤트 채널을 얇게 두지 않는다.
        news += await fetch_edgar_filings(symbols, start, end, user_agent=self._sec_user_agent)
        return news

    async def get_financials(self, symbols: list[str], asof_day: date):
        from adapters.edgar import fetch_edgar_financials

        _, end = observation_window(asof_day)
        return await fetch_edgar_financials(symbols, end, user_agent=self._sec_user_agent)

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """현재 체결가(same-day, 지연 가능) — 행동(지정가 산정) 전용."""
        out: dict[str, float] = {}
        for symbol in symbols:
            data = await self.session.get(
                "/uapi/overseas-price/v1/quotations/price",
                tr_id="HHDFS00000300",  # 실전/모의 공통
                params={"AUTH": "", "EXCD": self.excd, "SYMB": symbol},
            )
            out[symbol] = float(data["output"]["last"])
        return out

    # ── 계좌 ──

    async def _balance_rows(self) -> list[dict]:
        data = await self.session.get(
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            tr_id=BALANCE_TR[self.mode],
            params={
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.prdt,
                "OVRS_EXCG_CD": self.exchange,  # NASD = 미국 전체
                "TR_CRCY_CD": "USD",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            },
        )
        return data.get("output1") or []

    @staticmethod
    def _parse_positions(rows: list[dict]) -> list[Position]:
        """잔고 응답(output1) → 보유 포지션 (0주 행 제외). 금액은 USD."""
        return [
            Position(
                symbol=row["ovrs_pdno"],
                quantity=float(row["ovrs_cblc_qty"]),
                avg_price=float(row.get("pchs_avg_pric") or 0),
                market_value=float(row.get("ovrs_stck_evlu_amt") or 0),
            )
            for row in rows
            if float(row.get("ovrs_cblc_qty") or 0) > 0
        ]

    async def get_positions(self) -> list[Position]:
        return self._parse_positions(await self._balance_rows())

    async def _buying_power(self, ref_symbol: str, ref_price: float) -> float:
        """통합증거금 반영 매수여력(USD) — '환전이후주문가능금액'. 원화 담보를 자동환전한
        USD 주문 여력이라 USD 예수금이 없어도 매수 가능. 금액은 계좌 단위(심볼 무관)이나
        API 가 종목·단가를 요구해 유니버스 대표 1종을 기준으로 조회한다."""
        data = await self.session.get(
            "/uapi/overseas-stock/v1/trading/inquire-psamount",
            tr_id=PSAMOUNT_TR[self.mode],
            params={
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.prdt,
                "OVRS_EXCG_CD": self.exchange,
                "OVRS_ORD_UNPR": f"{ref_price:.2f}",
                "ITEM_CD": ref_symbol,
            },
        )
        return float((data.get("output") or {}).get("echm_af_ord_psbl_amt") or 0)

    async def get_equity(self) -> float:
        """버킷 평가액 — **원화 총자산**(FX-aware). 원화 담보로 산 미국 자산의 환율 손익까지
        MDD 서킷이 포착하도록 USD 가 아닌 KRW 로 계상. Risk Engine MDD 입력."""
        data = await self.session.get(
            "/uapi/overseas-stock/v1/trading/inquire-present-balance",
            tr_id=PRESENT_TR[self.mode],
            params={
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.prdt,
                "WCRC_FRCR_DVSN_CD": "01",  # 원화 기준
                "NATN_CD": "840",  # 미국
                "TR_MKET_CD": "00",
                "INQR_DVSN_CD": "00",
            },
        )
        out3 = data.get("output3")
        if isinstance(out3, list):  # 계정 유형별 dict/list 차이 방어
            out3 = out3[0] if out3 else {}
        return float((out3 or {}).get("tot_asst_amt") or 0)

    # ── 주문 ──

    def _limit_price(self, side: str, last: float) -> float:
        """체결형 지정가 — 매수는 위로, 매도는 아래로 버퍼. 미국 호가 $0.01 단위."""
        price = last * (1 + self.limit_buffer) if side == "buy" else last * (1 - self.limit_buffer)
        return round(price, 2)

    async def _pending_symbols(self) -> set[str]:
        """미체결 주문이 걸린 심볼 — 지정가 잔량에 중복 주문이 나가지 않게 제외 대상."""
        data = await self.session.get(
            "/uapi/overseas-stock/v1/trading/inquire-nccs",
            tr_id=NCCS_TR[self.mode],
            params={
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.prdt,
                "OVRS_EXCG_CD": self.exchange,
                "SORT_SQN": "DS",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            },
        )
        return {row["pdno"] for row in data.get("output") or [] if row.get("pdno")}

    async def _post_order(self, side: str, symbol: str, qty: int, price: float) -> dict:
        """지정가 주문. EGW00201(초당 제한)은 게이트웨이 선차단 = 주문 미접수라
        재시도가 안전. 그 외 오류는 본문 포함해 즉시 실패 — 맹목 재시도는 중복 주문 위험."""
        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.prdt,
            "OVRS_EXCG_CD": self.exchange,
            "PDNO": symbol,
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": f"{price:.2f}",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": "00" if side == "sell" else "",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",  # 지정가 (모의 유일 지원 — marketable limit 로 시장가 대용)
        }
        for _ in range(3):
            resp = await self.session.post(
                "/uapi/overseas-stock/v1/trading/order", ORDER_TR[(self.mode, side)], body
            )
            if "EGW00201" in resp.text:
                await asyncio.sleep(1.0)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"kis_us order http={resp.status_code} body={resp.text[:200]}")
            data = resp.json()
            if data.get("rt_cd") != "0":
                raise RuntimeError(f"kis_us order rt_cd={data.get('rt_cd')} msg={data.get('msg1')}")
            return data
        raise RuntimeError("kis_us order rate-limit 재시도 소진 (EGW00201)")

    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        now = datetime.now(timezone.utc)
        today = now.date()
        # kill switch — 사용자 수동 정지. 실자금 주문을 전면 차단(관측·결정은 이미 끝난 뒤).
        if self.live_guard and self.live_guard.kill_switch_active():
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error="kill_switch_active"
            )
        try:
            positions = self._parse_positions(await self._balance_rows())
            holdings = {p.symbol: p.market_value for p in positions}
            qty_held = {p.symbol: p.quantity for p in positions}
            prices = await self.get_current_prices(self.universe)
            # 통합증거금 매수여력(USD) — 유니버스 대표 1종 기준(금액은 계좌 단위)
            cash = await self._buying_power(self.universe[0], prices[self.universe[0]])
            pending = await self._pending_symbols()

            intents = compute_order_deltas(
                weights, holdings, cash, prices, min_notional=self.min_notional
            )
            orders = []
            for it in intents:
                if it.symbol in pending:
                    orders.append({"symbol": it.symbol, "side": it.side, "skipped": "open_order"})
                    continue
                limit = self._limit_price(it.side, prices[it.symbol])
                if it.side == "buy":
                    qty = int(it.notional / limit)  # 정수 주 — 잔여는 CASH 로 남는다
                else:
                    qty = min(int(it.qty or 0), int(qty_held.get(it.symbol, 0)))
                if qty < 1:
                    orders.append({"symbol": it.symbol, "side": it.side, "skipped": "sub_share"})
                    continue
                # 절대 금액 가드 — 1회/일일 명목 상한(실전만). 초과 주문은 스킵.
                notional = round(qty * limit, 2)
                if self.live_guard:
                    reason = self.live_guard.check(notional, today)
                    if reason:
                        orders.append({"symbol": it.symbol, "side": it.side, "skipped": reason})
                        continue
                placed = await self._post_order(it.side, it.symbol, qty, limit)
                if self.live_guard:
                    self.live_guard.charge(notional, today)  # 제출 성공분만 당일 누적
                orders.append(
                    {
                        "symbol": it.symbol,
                        "side": it.side,
                        "qty": qty,
                        "limit_price": limit,
                        "notional": notional,
                        "order_id": (placed.get("output") or {}).get("ODNO"),
                    }
                )
            return OrderResult(market=self.market, submitted_at=now, accepted=True, orders=orders)
        except Exception as e:  # 주문 실패는 예외가 아니라 결과로 — 러너가 로그로 남긴다
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=str(e)[:300]
            )
