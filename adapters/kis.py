"""한국 주식 어댑터 — KIS(한국투자증권) 국내주식. 모의/실전 겸용.

- 토큰: 발급 분당 1회 제한 + 24h 유효 → 파일 캐시(data/state/kis_token.json)로
  일일 루프·검증 스크립트가 재발급 제한에 걸리지 않게 한다.
- 시세: 기간별 일봉(FHKST03010100) — 모의/실전 동일 데이터. 수정주가 기준.
- 잔고·주문 tr_id 는 모의/실전이 다르다(V… / T…) — mode 로 전환.
- 뉴스: 무료 원천 미정 — 빈 리스트 (DART 공시 연동은 향후 작업).
- KR 은 정수 주식 수만 주문 가능 — qty < 1주 는 dust 로 스킵.
- 통합증거금 계좌는 국내와 해외가 같은 원화 현금을 쓴다. 시장마다 배분 벡터가 따로
  있으므로 현금은 bucket_share 지분만 자기 몫으로 본다.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from adapters.allocation import (
    bucket_cash,
    build_budget,
    project_to_executable,
    weights_from_quantities,
)
from adapters.base import (
    Bar,
    MarketAdapter,
    NewsItem,
    OrderResult,
    Position,
    execution_gaps,
    observation_window,
    price_outliers,
)
from adapters.retry import with_retry
from adapters.universe import ETF, asset_class

if TYPE_CHECKING:  # 런타임 임포트는 순환(risk → adapters.allocation)을 만든다
    from risk.live import LiveGuard

PAPER_BASE = "https://openapivts.koreainvestment.com:29443"
REAL_BASE = "https://openapi.koreainvestment.com:9443"
TOKEN_SAFETY_S = 600  # 만료 10분 전부터 재발급
RATE_GAP_S = 0.6  # 모의투자 초당 2건 제한(EGW00201) — 요청 간 최소 간격
HISTORY_MAX_PAGES = 12  # 장기 조회 페이지 상한(1회 ~100행 → ~1200거래일) — 폭주 방지

# 현재가 응답에서 '지금 이 종목에 주문이 닿는가'를 말해주는 필드들. 시세 조회 한 번에
# 같이 실려오므로 추가 호출이 없다 — 종전엔 체결가만 꺼내고 전부 버렸다.
#
# 판정하는 것과 기록만 하는 것을 나눈다. `_yn` 접미사 필드는 Y/N 이 자명하고 제한폭은
# 산수라 판정할 수 있지만, 구분코드들은 값 표를 확보하지 못했다 — 정상 거래 중인 유니버스
# 4종을 실제로 조회해 보니 iscd_stat_cls_code 가 "55", vi_cls_code 가 "N" 으로 왔다.
# "00이 정상"이라는 첫 가정대로였다면 멀쩡한 종목을 전부 비정상으로 몰 뻔했다.
# 그래서 코드값은 판정에 쓰지 않고 원값 그대로 쌓아 분포를 먼저 본다.


@dataclass(frozen=True)
class Quote:
    """체결가 + 그 시점의 거래 가능 상태.

    상한가·하한가는 값으로 온다(비율이 아니다). 현재가가 거기 닿아 있으면 그 방향 주문은
    호가가 없어 체결되지 않는다 — 하한가에서 매도가 안 나가는 상황이 대표적이며, 종전에는
    그것이 '미체결'로만 남아 거래단위·예산 때문에 못 담은 경우와 구분되지 않았다.
    """

    price: float
    halted: bool  # 임시 정지(temp_stop_yn) — Y/N
    liquidation: bool  # 정리매매(sltr_yn) — Y/N
    upper_limit: float | None  # 상한가
    lower_limit: float | None  # 하한가
    codes: dict[str, str] = field(default_factory=dict)  # 구분코드 원값(판정 안 함)

    @property
    def at_upper(self) -> bool:
        return self.upper_limit is not None and self.price >= self.upper_limit

    @property
    def at_lower(self) -> bool:
        return self.lower_limit is not None and self.price <= self.lower_limit

    @property
    def tradable(self) -> bool:
        """주문이 체결될 수 있는 상태인가. 의미가 자명한 Y/N 플래그만 본다.

        상/하한가는 방향 의존이라 여기 넣지 않는다 — 하한가에서도 매수는 가능하다.
        방향을 아는 호출부가 at_upper·at_lower 로 따로 판단해야 한다.
        """
        return not (self.halted or self.liquidation)

    def flags(self) -> dict[str, object]:
        """주의가 필요한 상태만 뽑는다. 정상이면 빈 dict — 코드값은 여기 넣지 않는다."""
        out: dict[str, object] = {}
        if self.halted:
            out["halted"] = True
        if self.liquidation:
            out["liquidation"] = True
        if self.at_upper:
            out["at_upper_limit"] = self.upper_limit
        if self.at_lower:
            out["at_lower_limit"] = self.lower_limit
        return out


BALANCE_TR = {"real": "TTTC8434R", "demo": "VTTC8434R"}
ORDER_TR = {
    ("real", "buy"): "TTTC0802U",
    ("real", "sell"): "TTTC0801U",
    ("demo", "buy"): "VTTC0802U",
    ("demo", "sell"): "VTTC0801U",
}


class KISSession:
    """KIS REST 세션 — 토큰 캐시·스로틀·공통 요청. 국내/해외 어댑터가 공유한다.

    토큰은 파일 캐시(발급 분당 1회 제한 + 24h 유효). 같은 앱 키를 쓰는 어댑터끼리는
    같은 캐시 파일을 공유해야 재발급 제한에 안 걸린다(키가 다르면 파일도 분리할 것).
    """

    def __init__(
        self, app_key: str, app_secret: str, token_cache: Path, base_url: str = PAPER_BASE
    ) -> None:
        self.app_key = app_key
        self.app_secret = app_secret
        self.token_cache = Path(token_cache)
        self._client = httpx.AsyncClient(base_url=base_url, timeout=15.0)
        self._last_request = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        """연속 호출 간격 강제 — 초당 건수 제한(EGW00201) 예방."""
        wait = self._last_request + RATE_GAP_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()

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

    async def headers(self, tr_id: str) -> dict[str, str]:
        return {
            "authorization": f"Bearer {await self._token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",  # 개인
        }

    async def get(self, path: str, tr_id: str, params: dict) -> dict:
        headers = await self.headers(tr_id)

        async def call():
            await self._throttle()
            resp = await self._client.get(path, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":  # KIS 는 HTTP 200 + rt_cd 로 오류 표현
                raise RuntimeError(f"kis rt_cd={data.get('rt_cd')} msg={data.get('msg1')}")
            return data

        return await with_retry(call, exceptions=(httpx.HTTPError,))

    async def post(self, path: str, tr_id: str, body: dict) -> httpx.Response:
        """스로틀 + 인증 POST. 상태 해석·재시도는 호출부 책임(주문 규약이 어댑터별)."""
        headers = await self.headers(tr_id)
        await self._throttle()
        return await self._client.post(path, headers=headers, json=body)


async def paginate_daily(
    fetch_page: Callable[[date], Awaitable[list[Bar]]], start: date, end: date
) -> list[Bar]:
    """기준일 역순 일봉 API 를 [start, end] 전 구간으로 이어붙인다 (오름차순·중복 제거).

    KIS 시세는 국내·해외 모두 1회 응답이 ~100행에서 잘리고 기준일에서 과거로 내려간다.
    한 번만 부르면 300일 창을 요청해도 최근 ~100봉만 와서, 크래시 없이 **요청보다 짧은
    창**으로 통과한다. fetch_page(cursor) 가 [start, cursor] 봉을 오름차순으로 돌려주면
    그 페이지의 최소 날짜 직전으로 커서를 당겨 start 까지 거슬러 올라간다.

    커서는 매 회 최소 하루씩 줄고 페이지 수도 상한을 둬 응답이 이상해도 멈춘다. 요청
    간격은 세션 스로틀이 지키므로(초당 건수 제한) 여기서 따로 대기하지 않는다.

    '기준일에서 과거로 자른다'는 전제가 원천에서 깨지면 다시 짧은 창으로 조용히 통과한다 —
    그래서 국면 로그에 실제로 받은 봉 수를 남긴다(요청 창 대비 대조가 유일한 확인 수단).
    """
    bars: dict[date, Bar] = {}
    cursor = end
    for _ in range(HISTORY_MAX_PAGES):
        if cursor < start:
            break
        page = await fetch_page(cursor)
        if not page:  # 더 과거 데이터 없음(상장 이전 등)
            break
        bars.update({b.day: b for b in page})
        if page[0].day <= start:  # 요청 하한 도달
            break
        cursor = page[0].day - timedelta(days=1)
    return sorted(bars.values(), key=lambda b: b.day)


def summarize_flows(rows: list[dict], short: int = 5, long: int = 20) -> dict:
    """투자자별 순매수 요약 — 최근 short 일 합(백만원)과 z-점수.

    z = short합 / (최근 long 일 일별 표준편차 × √short) — 종목 규모와 무관하게 비교 가능.
    표본이 long 미만이면 있는 만큼으로 계산, 표준편차 0(무변동)이면 z=0.
    """
    out: dict = {"days": len(rows), "last_day": rows[-1]["day"].isoformat() if rows else None}
    for who in ("foreign", "inst", "retail"):
        daily = [r[who] for r in rows][-long:]
        recent = daily[-short:]
        std = statistics.pstdev(daily) if len(daily) > 1 else 0.0
        out[f"{who}_{short}d"] = round(sum(recent))
        out[f"{who}_{short}d_z"] = (
            round(sum(recent) / (std * short**0.5), 3) if std > 0 else 0.0
        )
    return out


class KISDomesticAdapter(MarketAdapter):
    market = "KR"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        account: str,  # "12345678-01" (계좌 8자리-상품코드 2자리)
        universe: list[str],  # 예: ["069500"] (KODEX 200)
        token_cache: Path,
        mode: str = "demo",  # "demo" = 모의투자 / "real" = 실자금
        min_notional: float = 10_000.0,  # KRW — 1주 미만 잔주문 방지
        dart_api_key: str | None = None,  # 있으면 공시를 관측 뉴스에 편입
        live_guard: LiveGuard | None = None,  # 실전 절대 금액 가드(모의는 None)
        bucket_share: float = 1.0,  # 이 계좌의 현금 중 KR 몫 (계좌 단독 사용이면 1.0)
    ) -> None:
        assert mode in ("demo", "real"), f"mode={mode!r} — 'demo' 또는 'real'"
        self.session = KISSession(
            app_key, app_secret, Path(token_cache), PAPER_BASE if mode == "demo" else REAL_BASE
        )
        self.cano, _, self.prdt = account.partition("-")
        self.universe = universe
        self.mode = mode
        self.min_notional = min_notional
        self._dart_key = dart_api_key
        self.live_guard = live_guard
        self.bucket_share = bucket_share

    async def close(self) -> None:
        await self.session.close()

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

    async def _daily_page(self, symbol: str, start: date, cursor: date) -> list[Bar]:
        """[start, cursor] 일봉 1페이지 — 응답은 cursor 에서 과거로 최대 100행."""
        data = await self.session.get(
            "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            tr_id="FHKST03010100",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": cursor.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "0",  # 수정주가
            },
        )
        return self._parse_daily(data.get("output2") or [], start, cursor)

    async def _fetch_bars(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        # 관측·주문 경로 — 창이 짧아(최근 N거래일) 1회 조회(최대 100행)로 충분
        return {s: await self._daily_page(s, start, end) for s in symbols}

    async def _fetch_bars_history(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        """장기 창 전용 — 100행 상한을 커서 페이지네이션으로 넘긴다(국면·feature 입력).

        300일 창은 ~205거래일이라 1회 조회로는 절반도 안 온다. 상한 t-1 은 호출부가
        정한 end 를 그대로 쓰고 파싱에서 재확인하므로 페이지를 이어도 유지된다.
        """
        return {
            s: await paginate_daily(lambda c, s=s: self._daily_page(s, start, c), start, end)
            for s in symbols
        }

    async def get_news(self, symbols: list[str], asof_day: date) -> list[NewsItem]:
        from adapters.news_kr import fetch_kr_news

        start, end = observation_window(asof_day)
        news = await fetch_kr_news(symbols, start, end)
        if self._dart_key:  # DART 공시를 구조화 이벤트로 관측에 편입 (같은 뉴스 채널)
            from adapters.dart import fetch_dart_disclosures

            news += await fetch_dart_disclosures(self._dart_key, symbols, start, end)
        return news

    async def get_financials(self, symbols: list[str], asof_day: date):
        if not self._dart_key:
            return {}
        from adapters.dart import fetch_dart_financials

        _, end = observation_window(asof_day)
        return await fetch_dart_financials(self._dart_key, symbols, end)

    async def get_etf_nav(self, symbols: list[str], asof_day: date) -> dict[str, float]:
        """ETF 의 **전일 최종** 순자산가치(주당). 지수 ETF 만 조회, 실패는 무시.

        같은 응답의 `nav`·`dprt` 는 **당일** 값이라 쓰지 않는다 — 장중에 그 값을 받으면
        관측이 당일 가격을 아는 셈이 된다. 전일값만 관측 상한 안에 있다.

        주의: KIS 는 이 값이 **어느 날짜의 것인지 함께 주지 않는다**. 직전 거래일이라는
        전제로 쓰며, 어긋나면 t-1 종가와의 괴리가 비정상적으로 벌어진다 — 그 검사는
        비율을 계산하는 쪽에 둔다(여기서는 원값만 싣는다).
        """
        out: dict[str, float] = {}
        for symbol in symbols:
            if asset_class(symbol) != ETF:
                continue  # 개별주에는 순자산가치 개념이 없다
            try:
                data = await self.session.get(
                    "/uapi/etfetn/v1/quotations/inquire-price",
                    tr_id="FHPST02400000",
                    params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
                )
                nav = float((data.get("output") or {}).get("prdy_last_nav") or 0)
            except (httpx.HTTPError, RuntimeError, ValueError):
                continue  # 관측 보조 — 한 종목 실패로 결정 경로를 막지 않는다
            if nav > 0:
                out[symbol] = nav
        return out

    # ── 계좌 ──

    async def _balance(self) -> dict:
        return await self.session.get(
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id=BALANCE_TR[self.mode],
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

    @staticmethod
    def _parse_positions(data: dict) -> list[Position]:
        """잔고 응답(output1) → 보유 포지션 (0주 행 제외)."""
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

    async def get_positions(self) -> list[Position]:
        return self._parse_positions(await self._balance())

    def _bucket(self, data: dict) -> tuple[float, float]:
        """잔고 응답 → (이 시장이 쓸 수 있는 현금, 이 시장의 순자산).

        예수금(dnca_tot_amt)은 T+2 정산 미반영으로 과대계상 — 총평가에서 역산한다.
        """
        total_eval = float((data.get("output2") or [{}])[0].get("tot_evlu_amt") or 0)
        held = sum(p.market_value for p in self._parse_positions(data))
        cash = bucket_cash(total_eval - held, self.bucket_share)
        return cash, cash + held

    async def get_equity(self) -> float:
        return self._bucket(await self._balance())[1]

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """실시간 체결가 — 당일, 행동 전용(실시간 이벤트 트리거 감시·재결정 입력)."""
        return {s: await self._current_price(s) for s in symbols}

    # ── 수급 (투자자별 순매수) ──

    @staticmethod
    def _parse_investor(rows: list[dict], end: date) -> list[dict]:
        """일별 투자자 순매수 행 → [{day, foreign, inst, retail}] 오름차순 (대금, 백만원).

        당일 행은 장중 집계 전이라 필드가 빈 문자열로 온다 — 빈 행과 end(t-1) 초과분을
        함께 버려 미확정·당일 데이터가 관측에 들어오지 않게 한다.
        """
        out = []
        for r in rows:
            raw = r.get("stck_bsop_date") or ""
            if len(raw) != 8 or not r.get("frgn_ntby_tr_pbmn"):
                continue
            day = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
            if day <= end:
                out.append({
                    "day": day,
                    "foreign": float(r["frgn_ntby_tr_pbmn"]),
                    "inst": float(r["orgn_ntby_tr_pbmn"]),
                    "retail": float(r["prsn_ntby_tr_pbmn"]),
                })
        return sorted(out, key=lambda x: x["day"])

    async def get_investor_flows(
        self, symbols: list[str], asof_day: date
    ) -> dict[str, list[dict]]:
        """투자자별 순매수 동향(외인/기관/개인, 최근 ~30거래일) — 상한 t-1.

        수급 축적·반전(외인 연속 매도, 개인 투매 전환 등)은 봉·뉴스가 못 담는
        포지셔닝 신호라 별도 채널로 관측한다. 대금(백만원) 기준.
        """
        end = asof_day - timedelta(days=1)
        out: dict[str, list[dict]] = {}
        for symbol in symbols:
            data = await self.session.get(
                "/uapi/domestic-stock/v1/quotations/inquire-investor",
                tr_id="FHKST01010900",
                params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
            )
            out[symbol] = self._parse_investor(data.get("output") or [], end)
        return out

    # ── 주문 ──

    @staticmethod
    def _parse_quote(output: dict) -> Quote:
        """현재가 응답 → 체결가 + 거래 가능 상태. 상태 필드 결측은 정상으로 읽는다.

        모의투자 응답이 실전과 필드 구성이 다를 수 있어, 없는 필드로 종목을 비정상 처리하면
        멀쩡한 주문이 막힌다. 판정 재료가 없으면 판정하지 않는 쪽이 안전하다.
        """

        def _price(key: str) -> float | None:
            raw = (output.get(key) or "").strip()
            try:
                value = float(raw)
            except ValueError:
                return None
            return value or None  # 0 = 미제공(상/하한가 없는 종목)

        return Quote(
            price=float(output["stck_prpr"]),
            halted=(output.get("temp_stop_yn") or "N").strip() == "Y",
            liquidation=(output.get("sltr_yn") or "N").strip() == "Y",
            upper_limit=_price("stck_mxpr"),
            lower_limit=_price("stck_llam"),
            codes={
                key: value.strip()
                for key in ("vi_cls_code", "iscd_stat_cls_code", "mrkt_warn_cls_code")
                if (value := output.get(key))
            },
        )

    async def _quote(self, symbol: str) -> Quote:
        data = await self.session.get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            tr_id="FHKST01010100",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
        )
        return self._parse_quote(data["output"])

    async def _current_price(self, symbol: str) -> float:
        return (await self._quote(symbol)).price

    async def _post_order(self, side: str, symbol: str, qty: int) -> dict:
        """시장가 현금주문. EGW00201(초당 제한)은 게이트웨이 선차단 = 주문 미접수라
        재시도가 안전. 그 외 오류는 본문 포함해 즉시 실패 — 맹목 재시도는 중복 주문 위험."""
        tr_id = ORDER_TR[(self.mode, side)]
        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.prdt,
            "PDNO": symbol,
            "ORD_DVSN": "01",  # 시장가
            "ORD_QTY": str(qty),
            "ORD_UNPR": "0",
        }
        for _ in range(3):
            resp = await self.session.post(
                "/uapi/domestic-stock/v1/trading/order-cash", tr_id, body
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

    async def get_budget(self, unit_prices: dict[str, float]):
        """국내주식은 정수 주만 거래된다 — 1주 값이 곧 배분의 최소 눈금이다."""
        cash, equity = self._bucket(await self._balance())
        return build_budget(
            "KRW",
            equity,
            cash,
            unit_prices,
            lot=1,
            min_order=self.min_notional,
            max_order=self.live_guard.caps.max_order_notional if self.live_guard else None,
        )

    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        now = datetime.now(timezone.utc)
        today = now.date()
        # kill switch — 사용자 수동 정지. 실자금 주문을 전면 차단(관측·결정은 이미 끝난 뒤).
        if self.live_guard and self.live_guard.kill_switch_active():
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error="kill_switch_active"
            )
        try:
            data = await self._balance()  # 잔고 1회로 총평가·보유 동시 파싱
            cash, equity = self._bucket(data)
            held_qty = {p.symbol: p.quantity for p in self._parse_positions(data)}
            quotes = {s: await self._quote(s) for s in self.universe}
            prices = {s: q.price for s, q in quotes.items()}
            # 집행가 타당성 — 오류값 하나가 평가액 합계를 통해 다른 종목의 목표 수량까지
            # 흔들기 때문에, 그 종목만 빼는 것이 아니라 주문 전체를 접는다.
            gaps = execution_gaps(self._ref_closes, prices)
            outliers = price_outliers(gaps)
            if outliers:
                return OrderResult(
                    market=self.market, submitted_at=now, accepted=False,
                    error=f"price_outlier {outliers}", price_gap=gaps,
                )
            # 주문 시점의 종목 상태를 감사 로그로 넘긴다. **아직 주문을 막지는 않는다** —
            # 판정이 실거래를 자르기 전에 라이브에서 무엇이 얼마나 잡히는지 먼저 본다.
            # 코드값은 정상일 때도 남긴다(값 표가 없어 분포부터 모아야 한다).
            quote_status = {
                s: (q.flags() | ({"codes": q.codes} if q.codes else {}))
                for s, q in quotes.items()
            }

            # 정수 주만 거래된다 — 1주 값이 목표 금액보다 크면 그 종목은 통째로 빠진다.
            plan = project_to_executable(
                weights,
                held_qty,
                cash,
                prices,
                lot=1,
                min_notional=self.min_notional,
                max_order_notional=self.live_guard.caps.max_order_notional
                if self.live_guard else None,
            )
            final_qty = dict(held_qty)
            orders = []
            for it in plan.intents:
                qty = int(it.qty)
                if self.live_guard:
                    reason = self.live_guard.check(it.notional, today)
                    if reason:
                        orders.append({"symbol": it.symbol, "side": it.side, "skipped": reason})
                        plan.dropped[it.symbol] = reason
                        continue
                placed = await self._post_order(it.side, it.symbol, qty)
                if self.live_guard:
                    self.live_guard.charge(it.notional, today)  # 제출 성공분만 당일 누적
                final_qty[it.symbol] = final_qty.get(it.symbol, 0.0) + (
                    qty if it.side == "buy" else -qty
                )
                orders.append(
                    {
                        "symbol": it.symbol,
                        "side": it.side,
                        "qty": qty,
                        "notional": round(it.notional, 0),
                        "order_id": (placed.get("output") or {}).get("ODNO"),
                    }
                )
            return OrderResult(
                market=self.market,
                submitted_at=now,
                accepted=True,
                orders=orders,
                executed_weights=weights_from_quantities(final_qty, prices, equity),
                dropped=plan.dropped,
                quote_status=quote_status,
                price_gap=gaps,
            )
        except Exception as e:  # 주문 실패는 예외가 아니라 결과로 — 러너가 로그로 남긴다
            return OrderResult(
                market=self.market, submitted_at=now, accepted=False, error=str(e)[:300]
            )
