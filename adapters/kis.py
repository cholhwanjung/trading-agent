"""한국 주식 어댑터 — KIS(한국투자증권) 모의투자 (마지막 시장).

- 토큰: 발급 분당 1회 제한 + 24h 유효 → 파일 캐시(data/state/kis_token.json)로
  일일 루프·검증 스크립트가 재발급 제한에 걸리지 않게 한다.
- 시세: 기간별 일봉(FHKST03010100) — 모의/실전 동일 데이터. 수정주가 기준.
- 잔고 VTTC8434R · 시장가 현금주문 VTTC0802U(매수)/VTTC0801U(매도) — 모의 전용 tr_id.
- 뉴스: 무료 원천 미정 — 빈 리스트 (DART 공시 연동은 향후 작업).
- KR 은 정수 주식 수만 주문 가능 — qty < 1주 는 dust 로 스킵.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from adapters.allocation import compute_order_deltas
from adapters.base import Bar, MarketAdapter, NewsItem, OrderResult, Position, observation_window
from adapters.retry import with_retry

PAPER_BASE = "https://openapivts.koreainvestment.com:29443"
TOKEN_SAFETY_S = 600  # 만료 10분 전부터 재발급
RATE_GAP_S = 0.6  # 모의투자 초당 2건 제한(EGW00201) — 요청 간 최소 간격


class KISPaperAdapter(MarketAdapter):
    market = "KR"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        account: str,  # "12345678-01" (계좌 8자리-상품코드 2자리)
        universe: list[str],  # 예: ["069500"] (KODEX 200)
        token_cache: Path,
        min_notional: float = 10_000.0,  # KRW — 1주 미만 잔주문 방지
    ) -> None:
        self.app_key = app_key
        self.app_secret = app_secret
        self.cano, _, self.prdt = account.partition("-")
        self.universe = universe
        self.token_cache = Path(token_cache)
        self.min_notional = min_notional
        self._client = httpx.AsyncClient(base_url=PAPER_BASE, timeout=15.0)
        self._last_request = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        """연속 호출 간격 강제 — 초당 건수 제한(EGW00201) 예방."""
        wait = self._last_request + RATE_GAP_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()

    # ── 인증 ──

    async def _token(self) -> str:
        if self.token_cache.exists():
            cached = json.loads(self.token_cache.read_text(encoding="utf-8"))
            # 앱 키 교체 시 이전 키의 토큰 재사용 방지 — 지문 불일치면 재발급
            if (
                cached.get("app_key") == self.app_key[:8]
                and cached.get("expires_at", 0) - TOKEN_SAFETY_S > time.time()
            ):
                return cached["access_token"]
        resp = await self._client.post(
            "/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self.token_cache.parent.mkdir(parents=True, exist_ok=True)
        self.token_cache.write_text(
            json.dumps(
                {
                    "access_token": data["access_token"],
                    "expires_at": time.time() + int(data.get("expires_in", 86400)),
                    "app_key": self.app_key[:8],
                }
            ),
            encoding="utf-8",
        )
        return data["access_token"]

    async def _headers(self, tr_id: str) -> dict[str, str]:
        return {
            "authorization": f"Bearer {await self._token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",  # 개인
        }

    async def _get(self, path: str, tr_id: str, params: dict) -> dict:
        headers = await self._headers(tr_id)

        async def call():
            await self._throttle()
            resp = await self._client.get(path, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":  # KIS 는 HTTP 200 + rt_cd 로 오류 표현
                raise RuntimeError(f"kis rt_cd={data.get('rt_cd')} msg={data.get('msg1')}")
            return data

        return await with_retry(call, exceptions=(httpx.HTTPError,))

    # ── 관측 ──

    @staticmethod
    def _parse_daily(rows: list[dict], start: date, end: date) -> list[Bar]:
        """output2 일봉 행 → [start, end] 윈도우 Bar 오름차순 (당일 봉 차단)."""
        bars = []
        for r in rows:
            if not r.get("stck_bsop_date"):
                continue  # KIS 는 빈 placeholder 행을 섞어 보낸다
            day = datetime.strptime(r["stck_bsop_date"], "%Y%m%d").date()
            if start <= day <= end:
                bars.append(
                    Bar(
                        day=day,
                        open=float(r["stck_oprc"]),
                        high=float(r["stck_hgpr"]),
                        low=float(r["stck_lwpr"]),
                        close=float(r["stck_clpr"]),
                        volume=float(r["acml_vol"]),
                    )
                )
        return sorted(bars, key=lambda b: b.day)

    async def _fetch_bars(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        out: dict[str, list[Bar]] = {}
        for symbol in symbols:
            data = await self._get(
                "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                tr_id="FHKST03010100",
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": symbol,
                    "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D",
                    "FID_ORG_ADJ_PRC": "0",  # 수정주가
                },
            )
            out[symbol] = self._parse_daily(data.get("output2") or [], start, end)
        return out

    async def get_ohlcv(self, symbols: list[str], asof_day: date) -> dict[str, list[Bar]]:
        start, end = observation_window(asof_day)
        return await self._fetch_bars(symbols, start, end)

    async def get_ohlcv_history(
        self, symbols: list[str], asof_day: date, lookback_days: int = 90
    ) -> dict[str, list[Bar]]:
        # 상한 t-1. API 1회 응답 최대 100행 — lookback 90 은 1회로 충분
        return await self._fetch_bars(
            symbols, asof_day - timedelta(days=lookback_days), asof_day - timedelta(days=1)
        )

    async def get_news(self, symbols: list[str], asof_day: date) -> list[NewsItem]:
        from adapters.news_kr import fetch_kr_news

        start, end = observation_window(asof_day)
        return await fetch_kr_news(symbols, start, end)

    # ── 계좌 ──

    async def _balance(self) -> dict:
        return await self._get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id="VTTC8434R",  # 모의투자 잔고조회
            params={
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.prdt,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )

    async def get_positions(self) -> list[Position]:
        data = await self._balance()
        return [
            Position(
                symbol=row["pdno"],
                quantity=float(row["hldg_qty"]),
                avg_price=float(row["pchs_avg_pric"]),
                market_value=float(row["evlu_amt"]),
            )
            for row in data.get("output1") or []
            if float(row.get("hldg_qty") or 0) > 0
        ]

    async def get_equity(self) -> float:
        data = await self._balance()
        total = (data.get("output2") or [{}])[0]
        return float(total.get("tot_evlu_amt") or 0)  # 예수금 + 평가액

    # ── 주문 ──

    async def _current_price(self, symbol: str) -> float:
        data = await self._get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
        )
        return float(data["output"]["stck_prpr"])

    async def _post_order(self, side: str, symbol: str, qty: int) -> dict:
        """시장가 현금주문. EGW00201(초당 제한)은 게이트웨이 선차단 = 주문 미접수라
        재시도가 안전. 그 외 오류는 본문 포함해 즉시 실패 — 맹목 재시도는 중복 주문 위험."""
        tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"  # 모의 매수/매도
        headers = await self._headers(tr_id)
        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.prdt,
            "PDNO": symbol,
            "ORD_DVSN": "01",  # 시장가
            "ORD_QTY": str(qty),
            "ORD_UNPR": "0",
        }
        for _ in range(3):
            await self._throttle()
            resp = await self._client.post(
                "/uapi/domestic-stock/v1/trading/order-cash", headers=headers, json=body
            )
            if "EGW00201" in resp.text:
                await asyncio.sleep(1.0)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"kis order http={resp.status_code} body={resp.text[:200]}")
            data = resp.json()
            if data.get("rt_cd") != "0":
                raise RuntimeError(f"kis order rt_cd={data.get('rt_cd')} msg={data.get('msg1')}")
            return data
        raise RuntimeError("kis order rate-limit 재시도 소진 (EGW00201)")

    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        now = datetime.now(timezone.utc)
        try:
            data = await self._balance()
            total_eval = float((data.get("output2") or [{}])[0].get("tot_evlu_amt") or 0)
            holdings = {p.symbol: p.market_value for p in await self.get_positions()}
            # 예수금(dnca_tot_amt)은 T+2 정산 미반영으로 과대계상 — 총평가에서 역산
            cash = total_eval - sum(holdings.values())
            prices = {s: await self._current_price(s) for s in self.universe}

            intents = compute_order_deltas(
                weights, holdings, cash, prices, min_notional=self.min_notional
            )
            orders = []
            for it in intents:
                qty = int(it.qty or 0)  # KR 은 정수 주만
                if qty < 1:
                    orders.append({"symbol": it.symbol, "side": it.side, "skipped": "sub_share"})
                    continue
                placed = await self._post_order(it.side, it.symbol, qty)
                orders.append(
                    {
                        "symbol": it.symbol,
                        "side": it.side,
                        "qty": qty,
                        "notional": round(it.notional, 0),
                        "order_id": (placed.get("output") or {}).get("ODNO"),
                    }
                )
            return OrderResult(market=self.market, submitted_at=now, accepted=True, orders=orders)
        except Exception as e:  # 주문 실패는 예외가 아니라 결과로 — 러너가 로그로 남긴다
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=str(e)[:300]
            )
