"""Market Adapter 통일 인터페이스 (R1).

모든 시장(KIS/Alpaca/ccxt)은 이 인터페이스를 구현한다. Trader는 어댑터 구현을
알지 못한 채 배분비율 벡터만 넘기고, 배분비율 → 주문(Δq) 변환은 어댑터 책임이다
(LiveTradeBench 방식, [docs/ARCHITECTURE.md] · [ADR-006]).

하드룰 (CLAUDE.md (C)):
- 관측 윈도우는 [t-3, t-1]로 고정, same-day leakage 차단 (R2).
- 모든 관측에 수집 타임스탬프 기록 → 사후 감사 가능.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone


# 관측 윈도우: 오늘(t) 기준 [t-3, t-1]. same-day(t) 데이터는 절대 포함 금지.
OBSERVATION_LOOKBACK_DAYS = 3


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

    collected_at: 수집 시각(UTC). asof_day: 관측 기준일 t. window: 실제 [t-3, t-1].
    """

    market: str
    asof_day: date
    collected_at: datetime
    bars: dict[str, list[Bar]] = field(default_factory=dict)  # symbol -> 봉 리스트
    news: list[NewsItem] = field(default_factory=list)


@dataclass(frozen=True)
class OrderResult:
    """submit_allocation 결과. 어댑터가 배분비율을 주문으로 변환한 뒤 돌려준다."""

    market: str
    submitted_at: datetime
    accepted: bool
    orders: list[dict] = field(default_factory=list)  # 어댑터별 주문 표현(Δq 포함)
    error: str | None = None


def observation_window(asof_day: date, lookback: int = OBSERVATION_LOOKBACK_DAYS) -> tuple[date, date]:
    """관측 허용 구간 [t-lookback, t-1] 을 (start, end) 로 반환. end 는 t-1 (포함)."""

    from datetime import timedelta

    start = asof_day - timedelta(days=lookback)
    end = asof_day - timedelta(days=1)
    return start, end


class LeakageError(AssertionError):
    """관측 윈도우 [t-3, t-1] 밖(특히 same-day t 이후) 데이터가 섞였을 때."""


def assert_no_leakage(obs: Observation) -> None:
    """Observation 이 same-day leakage 없이 [t-3, t-1] 안에 있는지 검증 (R2 verify).

    위반 시 LeakageError. 하니스·테스트가 모든 관측에 대해 호출한다.
    """

    start, end = observation_window(obs.asof_day)
    for symbol, bars in obs.bars.items():
        for bar in bars:
            if not (start <= bar.day <= end):
                raise LeakageError(
                    f"leakage market={obs.market} symbol={symbol} "
                    f"bar_day={bar.day} window=[{start},{end}] asof={obs.asof_day}"
                )
    for item in obs.news:
        news_day = item.published_at.date()
        if not (start <= news_day <= end):
            raise LeakageError(
                f"leakage market={obs.market} news_day={news_day} "
                f"window=[{start},{end}] asof={obs.asof_day} headline={item.headline!r}"
            )


class MarketAdapter(ABC):
    """시장 어댑터 계약. 구현체는 market 이름과 4개 메서드를 제공한다."""

    #: "KR" | "US" | "CRYPTO" — 메모리 네임스페이스 키로도 쓰인다 (ADR-007).
    market: str

    @abstractmethod
    async def get_ohlcv(self, symbols: list[str], asof_day: date) -> dict[str, list[Bar]]:
        """[t-3, t-1] 구간의 일봉을 symbol별로 반환. same-day(t) 봉 포함 금지 (R2)."""

    @abstractmethod
    async def get_news(self, symbols: list[str], asof_day: date) -> list[NewsItem]:
        """[t-3, t-1] 구간에 발행된 뉴스만 반환. published_at >= t 인 건 제외 (R2)."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """현재 페이퍼 계좌의 보유 포지션. 현금은 별도 조회(구현체 책임)."""

    async def get_equity(self) -> float:
        """페이퍼 계좌 총 평가액(현금 포함, quote 통화). Risk Engine MDD 서킷 입력.

        기본 미구현 — 실브로커 어댑터만 구현하면 된다.
        """
        raise NotImplementedError(f"{type(self).__name__}는 get_equity 미구현")

    async def get_ohlcv_history(
        self, symbols: list[str], asof_day: date, lookback_days: int = 90
    ) -> dict[str, list[Bar]]:
        """feature 계산용 장기 일봉 [t-lookback, t-1] ([ADR-013] — 상한 t-1 은 동일 강제).

        기본 미구현 — 실브로커 어댑터만 구현하면 된다(Mock/baseline 은 불필요).
        """
        raise NotImplementedError(f"{type(self).__name__}는 get_ohlcv_history 미구현")

    async def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """현재 체결가(same-day, 실시간). **행동 전용** — 관측·feature·학습에 쓰지 말 것
        (하드룰 7 · [ADR-013]). 실시간 이벤트 트리거([ADR-021])와 주문 집행 용도.

        기본 미구현 — 트리거 대상 어댑터만 구현.
        """
        raise NotImplementedError(f"{type(self).__name__}는 get_current_prices 미구현")

    @abstractmethod
    async def submit_allocation(self, weights: dict[str, float]) -> OrderResult:
        """배분비율 벡터(∑=1, 현금 포함)를 받아 주문(Δq)으로 변환·제출 (R6).

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
        return Observation(
            market=self.market,
            asof_day=asof_day,
            collected_at=datetime.now(timezone.utc),
            bars=bars,
            news=news,
        )
