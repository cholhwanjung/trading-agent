"""Market Adapter 통일 인터페이스.

모든 시장(KIS/Alpaca/ccxt)은 이 인터페이스를 구현한다. Trader는 어댑터 구현을
알지 못한 채 배분비율 벡터만 넘기고, 배분비율 → 주문(Δq) 변환은 어댑터 책임이다
(LiveTradeBench 방식).

불변 제약:
- 관측 상한은 t-1 고정(당일 t 데이터 절대 차단) — 누출 통제의 본질. 봉은 최근 N거래일,
  뉴스는 최근 N캘린더일(길이는 configure_observation 으로 조정 가능).
- 모든 관측에 수집 타임스탬프 기록 → 사후 감사 가능.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from adapters.financials import Financials
from adapters.index_valuation import IndexValuation


# 관측 윈도우 기본값. 상한은 항상 t-1(당일 t 데이터 절대 포함 금지) — 이게 누출 통제의 본질이며
# 아래 길이 값과 무관하게 불변이다. 운영 스크립트가 configure_observation()으로 .env 값을 주입해
# 실험적으로 조정할 수 있다(미설정 시 기본 유지).
OBSERVATION_TRADING_DAYS = 3  # 원시 최근 봉: 최근 N '거래일'(요일·휴장과 무관하게 일정)
OBSERVATION_NEWS_DAYS = 7     # 뉴스: 최근 N 캘린더일(봉 창과 디커플 — 사건은 거래세션에 안 매임)


@dataclass(frozen=True)
class Bar:
    """OHLCV 한 개 봉. 하루 1봉(일봉) 기준."""

    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class NewsItem:
    """뉴스 한 건. published_at 은 관측 윈도우 검증에 사용된다."""

    published_at: datetime
    headline: str
    source: str
    url: str | None = None


# 공시 어댑터가 NewsItem.source 에 붙이는 값. 공시는 수집·누출검증을 뉴스와 같은 리스트로
# 통과하지만(둘 다 published_at 기준 윈도우 판정) 소비 시점에는 갈라내야 한다 —
# 어댑터가 뉴스 뒤에 이어붙이고 소비자가 앞에서 잘라 쓰면 공시가 통째로 잘려나간다.
DISCLOSURE_SOURCES: frozenset[str] = frozenset({"DART", "SEC"})


def round_robin_news(groups: list[list[NewsItem]], limit: int) -> list[NewsItem]:
    """질의별 리스트에서 번갈아 한 건씩 뽑는다 — 기사 많은 채널의 창 독식 차단.

    같은 이유로 존재한다: 소비자는 뉴스 리스트를 앞에서 잘라 쓰므로, 채널을 이어붙이면
    뒤에 붙은 채널이 한 건도 남지 않는다. 각 그룹은 최신순이라 채널마다 '가장 최근
    것부터' 들어가고, 소진된 그룹은 건너뛰어 슬롯을 남기지 않는다. 그룹 간 엄밀한
    시각순은 포기한다 — 며칠 창 안에서 헤드라인 사이의 분 단위 선후보다 채널 커버리지가
    결정에 쓸모 있다.
    """

    out: list[NewsItem] = []
    for row in itertools.zip_longest(*groups):
        for item in row:
            if item is not None:
                out.append(item)
                if len(out) >= limit:
                    return out
    return out


@dataclass(frozen=True)
class Position:
    """자산 1종의 현재 보유 상태."""

    symbol: str
    quantity: float
    avg_price: float
    market_value: float


@dataclass(frozen=True)
class Observation:
    """어댑터가 반환하는 관측 묶음. 모든 관측은 이 컨테이너로 감사된다.

    collected_at: 수집 시각(UTC). asof_day: 관측 기준일 t. 봉=최근 N거래일·뉴스=N캘린더일(상한 t-1).
    financials: 분기 재무 스냅샷(symbol -> 값). 저속 채널이라 창 길이가 아니라 제출일 상한만 건다.
    index_valuation: 시장 전체의 밸류에이션 수준(시장당 1건). 종목별이 아닌 이유는
    지수 ETF 에 종목 재무가 없기 때문 — 대응물이 없는 시장은 None.
    etf_nav: ETF 의 전일 최종 순자산가치(symbol -> 주당 금액). 원천이 없는 시장은 빈 dict.
    """

    market: str
    asof_day: date
    collected_at: datetime
    bars: dict[str, list[Bar]] = field(default_factory=dict)  # symbol -> 봉 리스트
    news: list[NewsItem] = field(default_factory=list)
    financials: dict[str, Financials] = field(default_factory=dict)
    index_valuation: IndexValuation | None = None
    etf_nav: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderResult:
    """submit_allocation 결과. 어댑터가 배분비율을 주문으로 변환한 뒤 돌려준다.

    executed_weights 는 이 주문들이 반영된 뒤 **실제로 보유하게 되는** 배분이다. 목표와
    다를 수 있다 — 정수 주 단위, 최소 주문 금액, 1회 명목 상한, 미체결 종목이 의도를
    잘라내기 때문이다. 학습은 의도가 아니라 이 값을 써야 한다: 목표 배분으로 성과를
    귀속시키면 보유한 적 없는 포트폴리오의 손익을 교훈으로 승격시키게 된다.
    dropped 는 그렇게 잘려나간 종목과 사유(진단용).
    """

    market: str
    submitted_at: datetime
    accepted: bool
    orders: list[dict] = field(default_factory=list)  # 어댑터별 주문 표현(Δq 포함)
    error: str | None = None
    executed_weights: dict[str, float] | None = None
    dropped: dict[str, str] = field(default_factory=dict)
    #: 주문 시점의 종목 상태(symbol -> 상태). 거래소가 종목을 멈추거나 가격이 제한폭에
    #: 닿으면 주문이 나가도 체결되지 않는데, 그 사실이 결과에 남지 않으면 나중에 미체결의
    #: 원인을 거래단위·예산 문제와 구분할 수 없다. 상태를 제공하지 않는 시장은 빈 dict.
    quote_status: dict[str, dict] = field(default_factory=dict)


# 브로커가 "장이 닫혀 있다"는 뜻으로 돌려주는 거부 문구. 브로커마다 표현이 달라
# 코드가 아니라 메시지로 식별할 수밖에 없다(공통 에러코드 규약이 없다).
_MARKET_CLOSED_HINTS = (
    "영업일이 아닙니다",  # KIS 모의투자 — 주말·공휴일
    "장운영시간이 아닙니다",  # KIS — 장 시작 전/마감 후
    "market is closed",
)


def is_market_closed_error(error: str | None) -> bool:
    """주문 거부 사유가 '장 마감'인지. 운영 장애와 구분하기 위한 판정.

    장이 닫혀 있으면 체결될 주문 자체가 없으므로 이것은 실패가 아니다. 실패로 세면
    주말마다 종료코드·통지·대시보드 경고가 켜져서 진짜 장애 신호가 그 잡음에 묻힌다.
    """
    return bool(error) and any(hint in error for hint in _MARKET_CLOSED_HINTS)


def observation_window(asof_day: date, lookback: int | None = None) -> tuple[date, date]:
    """뉴스·일반 관측의 캘린더 윈도우 [t-lookback, t-1]. end=t-1(당일 차단, 불변).

    lookback=None 이면 모듈 기본값(OBSERVATION_NEWS_DAYS)을 호출 시점에 읽는다
    (configure_observation 오버라이드가 반영되도록 기본 인자로 캡처하지 않음).
    """

    n = OBSERVATION_NEWS_DAYS if lookback is None else lookback
    return asof_day - timedelta(days=n), asof_day - timedelta(days=1)


def bar_observation_window(asof_day: date, trading_days: int | None = None) -> tuple[date, date]:
    """원시 봉 조회 구간. end=t-1(당일 차단, 불변).

    최근 N '거래일'을 주말·휴장을 넘어 확보하기 위해 캘린더 하한을 여유있게(2N+10일) 잡는다.
    실제 봉은 get_ohlcv 가 최근 N개로 절삭하므로, 이 구간은 '넓게 떠서 뒤에서 N개'용 fetch 범위다.
    """

    n = OBSERVATION_TRADING_DAYS if trading_days is None else trading_days
    return asof_day - timedelta(days=n * 2 + 10), asof_day - timedelta(days=1)


def configure_observation(env: dict[str, str]) -> None:
    """운영 스크립트가 .env로 관측 윈도우 길이를 오버라이드(실험 변수). 미설정 키는 기본 유지."""

    global OBSERVATION_TRADING_DAYS, OBSERVATION_NEWS_DAYS
    if env.get("OBSERVATION_TRADING_DAYS"):
        OBSERVATION_TRADING_DAYS = int(env["OBSERVATION_TRADING_DAYS"])
    if env.get("OBSERVATION_NEWS_DAYS"):
        OBSERVATION_NEWS_DAYS = int(env["OBSERVATION_NEWS_DAYS"])


class LeakageError(AssertionError):
    """관측 윈도우 밖(특히 same-day t 이후) 데이터가 섞였을 때."""


def assert_no_leakage(obs: Observation) -> None:
    """Observation 에 same-day leakage 가 없는지 검증 (verify).

    누출의 본질은 상한 end=t-1 — 봉·뉴스 모두 당일(t) 이후를 절대 포함하지 않아야 한다.
    하한은 각 채널 윈도우(봉=최근 N거래일 fetch 구간, 뉴스=N캘린더일)를 쓴다.
    위반 시 LeakageError. 하니스·테스트가 모든 관측에 대해 호출한다.
    """

    bar_start, end = bar_observation_window(obs.asof_day)
    for symbol, bars in obs.bars.items():
        for bar in bars:
            if not (bar_start <= bar.day <= end):
                raise LeakageError(
                    f"leakage market={obs.market} symbol={symbol} "
                    f"bar_day={bar.day} window=[{bar_start},{end}] asof={obs.asof_day}"
                )
    news_start, _ = observation_window(obs.asof_day)
    for item in obs.news:
        news_day = item.published_at.date()
        if not (news_start <= news_day <= end):
            raise LeakageError(
                f"leakage market={obs.market} news_day={news_day} "
                f"window=[{news_start},{end}] asof={obs.asof_day} headline={item.headline!r}"
            )
    # 재무는 회계기간이 아니라 **공시 제출일**로 자른다. 분기 종료 후 수 주 뒤에야 공시되므로
    # 종료일 기준으로 자르면 그 사이 날짜의 관측이 아직 발표되지 않은 실적을 아는 셈이 된다.
    # 하한은 두지 않는다 — 분기 공시라 며칠~수개월 묵은 것이 정상이다.
    for symbol, fin in obs.financials.items():
        if fin.filed > end:
            raise LeakageError(
                f"leakage market={obs.market} symbol={symbol} "
                f"filed={fin.filed} max={end} asof={obs.asof_day} period_end={fin.period_end}"
            )
    # 밸류에이션은 최근 종가로 계산되므로 기준일이 곧 가격의 날짜다 — 봉과 같은 상한을 건다.
    if obs.index_valuation is not None and obs.index_valuation.asof > end:
        raise LeakageError(
            f"leakage market={obs.market} index_valuation_asof={obs.index_valuation.asof} "
            f"max={end} asof={obs.asof_day} proxy={obs.index_valuation.proxy}"
        )


class MarketAdapter(ABC):
    """시장 어댑터 계약. 구현체는 market 이름 + _fetch_bars(시세) + 뉴스·포지션·주문을 제공한다."""

    #: "KR" | "US" | "CRYPTO" — 메모리 네임스페이스 키로도 쓰인다.
    market: str

    async def _fetch_bars(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        """[start, end] 구간 일봉 조회 (1회) — 관측·주문 경로(get_ohlcv)의 시세 소스.

        실브로커 어댑터는 이것만 구현하면 두 조회 메서드가 기본 제공된다.
        Mock/baseline 은 get_ohlcv 를 직접 재정의해도 된다.
        """
        raise NotImplementedError(f"{type(self).__name__}는 _fetch_bars 미구현")

    async def _fetch_bars_history(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[Bar]]:
        """[start, end] **전 구간** 일봉 — get_ohlcv_history 전용 읽기 경로.

        기본은 _fetch_bars 위임(1회 조회로 구간이 다 오는 원천). 1회 응답 행 수가
        제한된 원천은 **이것만** 재정의해 페이지네이션을 붙인다 — 집행에 쓰이는
        _fetch_bars 는 그대로 둬서 주문 경로에 조회 횟수·실패면을 늘리지 않는다.
        """
        return await self._fetch_bars(symbols, start, end)

    async def get_ohlcv(self, symbols: list[str], asof_day: date) -> dict[str, list[Bar]]:
        """최근 N '거래일'의 일봉을 symbol별로 반환. same-day(t) 봉 포함 금지.

        주말·휴장으로 캘린더 [t-N, t-1] 이 요일마다 1~N개로 들쭉날쭉해지는 것을 막기 위해,
        넓은 구간으로 조회한 뒤 오름차순 정렬해 최근 N개만 남긴다(거래일 기준 일정).
        """
        start, end = bar_observation_window(asof_day)
        fetched = await self._fetch_bars(symbols, start, end)
        n = OBSERVATION_TRADING_DAYS
        return {s: sorted(bars, key=lambda b: b.day)[-n:] for s, bars in fetched.items()}

    async def get_ohlcv_history(
        self, symbols: list[str], asof_day: date, lookback_days: int = 90
    ) -> dict[str, list[Bar]]:
        """feature 계산용 장기 일봉 [t-lookback, t-1] (상한 t-1 은 동일 강제).

        요청한 창이 실제로 다 와야 한다 — 시세 API 가 1회 응답을 잘라도 크래시가 아니라
        **짧은 창**으로 조용히 통과하고, 그 위에서 계산되는 롤링 피크·긴 반감기 피처가
        설계보다 약해진다. 이어붙이기는 _fetch_bars_history 가 담당한다.
        """
        return await self._fetch_bars_history(
            symbols, asof_day - timedelta(days=lookback_days), asof_day - timedelta(days=1)
        )

    @abstractmethod
    async def get_news(self, symbols: list[str], asof_day: date) -> list[NewsItem]:
        """최근 N 캘린더일에 발행된 뉴스만 반환. published_at >= t(당일) 인 건 제외."""

    async def get_financials(
        self, symbols: list[str], asof_day: date
    ) -> dict[str, Financials]:
        """제출일이 t-1 이하인 최신 분기 재무. 원천이 없는 시장은 빈 dict.

        기본 미구현이 아니라 빈 반환 — 재무 대응물이 없는 시장(크립토)이 정상 상태다.
        """
        return {}

    async def get_etf_nav(self, symbols: list[str], asof_day: date) -> dict[str, float]:
        """ETF 의 전일 최종 순자산가치(주당). 원천이 없는 시장은 빈 dict.

        미구현이 아니라 빈 반환 — 순자산가치 대응물이 없는 시장이 정상 상태다.
        """
        return {}

    async def get_index_valuation(self, asof_day: date):
        """이 시장의 밸류에이션 수준 1건. 대응물이 없는 시장은 None.

        어댑터별로 재정의하지 않는다 — 원천을 고르는 기준이 브로커가 아니라 **시장**이라
        (같은 US 라도 Alpaca 든 KIS 든 보는 시장 밸류에이션은 하나다) 구현체마다 두면
        같은 한 줄이 복제된다.
        """
        from adapters.index_valuation import fetch_index_valuation

        return await fetch_index_valuation(self.market, asof_day)

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """현재 페이퍼 계좌의 보유 포지션. 현금은 별도 조회(구현체 책임)."""

    async def get_equity(self) -> float:
        """페이퍼 계좌 총 평가액(현금 포함, quote 통화). Risk Engine MDD 서킷 입력.

        기본 미구현 — 실브로커 어댑터만 구현하면 된다.
        """
        raise NotImplementedError(f"{type(self).__name__}는 get_equity 미구현")

    async def get_budget(self, unit_prices: dict[str, float]):
        """결정 시점의 예산 제약(BudgetSnapshot) — 트레이더 관측용. 미제공이면 None.

        unit_prices 는 t-1 종가다(당일 시세를 결정에 흘리지 않기 위해). 계좌 통화와
        관측 통화가 다른 거래소는 이 값을 쓸 수 없지만, 그런 경우는 분수 거래라
        거래단위 제약 자체가 없어 문제가 되지 않는다.

        기본 미구현이 아니라 None — 예산 제약을 모르는 것이 곧 제약이 없다는 뜻은
        아니므로, 값을 지어내지 않고 '모른다'로 둔다(프롬프트에서 블록 자체가 빠진다).
        """
        return None

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """현재 체결가(same-day, 실시간). **행동 전용** — 관측·feature·학습에 쓰지 말 것
        실시간 이벤트 트리거와 주문 집행 용도.

        기본 미구현 — 트리거 대상 어댑터만 구현.
        """
        raise NotImplementedError(f"{type(self).__name__}는 get_current_prices 미구현")

    @abstractmethod
    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        """배분비율 벡터(∑=1, 현금 포함)를 받아 주문(Δq)으로 변환·제출.

        weights 예: {"BTC/USDT": 0.4, "ETH/USDT": 0.2, "CASH": 0.4}
        """

    async def observe_and_audit(self, symbols: list[str], asof_day: date | None = None) -> Observation:
        """observe 후 누출 검사까지 수행. 위반 시 LeakageError. 하니스 기본 진입점."""

        obs = await self.observe(symbols, asof_day)
        assert_no_leakage(obs)
        return obs

    async def observe(self, symbols: list[str], asof_day: date | None = None) -> Observation:
        """get_ohlcv + get_news 를 묶어 감사 가능한 Observation 으로 반환.

        하위 클래스가 재정의할 필요 없는 공통 조립 + 타임스탬프 부여 지점.
        """

        asof_day = asof_day or datetime.now(timezone.utc).date()
        bars = await self.get_ohlcv(symbols, asof_day)
        news = await self.get_news(symbols, asof_day)
        financials = await self.get_financials(symbols, asof_day)
        index_valuation = await self.get_index_valuation(asof_day)
        etf_nav = await self.get_etf_nav(symbols, asof_day)
        return Observation(
            market=self.market,
            asof_day=asof_day,
            collected_at=datetime.now(timezone.utc),
            bars=bars,
            news=news,
            financials=financials,
            index_valuation=index_valuation,
            etf_nav=etf_nav,
        )


@runtime_checkable
class TreasuryCapable(Protocol):
    """버킷 간 자본 이체의 '자동 레그' 계약 — 가용 잔고 조회 + 등록 계좌로 출금.

    API 로 집행 가능한 레그(Upbit KRW 출금)만 구현한다. 출금 목적지는 거래소가 KYC 로
    고정한 본인 명의 계좌 — 자유 주소가 아니므로 이체 목적지 조작이 구조적으로 불가능하다.
    실집행은 이체 가드(allowlist·상한·쿨다운) 통과분만; API 부재 레그(증권·은행 입금)는
    수동 액추에이션(별도)으로 처리한다.

    코인 출금은 의도적으로 제외 — 자본 이동은 온-거래소 KRW 환전 후 KRW 출금으로만 하고
    코인을 외부 지갑으로 반출하지 않는다(반출 공격면 회피).
    """

    venue: str

    async def withdrawable_krw(self) -> float:
        """지금 출금 가능한 KRW 가용 잔고(잠금·미체결 제외). 이체 가드 잔고 대조 입력."""
        ...

    async def withdraw_krw(self, amount: float) -> dict:
        """등록 계좌로 KRW 출금 집행 — **실자금 이동**. 이체 가드 통과 후에만 호출.

        반환은 거래 레코드({uuid, state, amount})로 이후 잔고 대조(reconcile)에 쓴다.
        """
        ...
