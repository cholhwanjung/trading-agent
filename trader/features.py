"""Trader 관측 feature — 정예 7종 (R17: 5~8개 한정, 60종 주입 금지 · [ADR-012]).

Alpha Arena 근거(단순 지표 우승 패턴)에 따라 검증된 고전 지표만.
알파 원천이 아니라 관측 보조 — LLM 에게 구조화 숫자로 주입된다.

누출 규약: 입력 봉은 asof_day 전일(t-1)까지의 과거 시계열. 지표 lookback 은
[t-3, t-1] 결정 컨텍스트 윈도우와 별개다(하드룰 7의 본질은 same-day 차단).
t 이후 봉이 섞이면 LeakageError.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from adapters.base import Bar, LeakageError

# 정예 feature 목록 — 이 튜플이 유일한 진실. 8개 초과 금지 (R17).
FEATURE_NAMES: tuple[str, ...] = (
    "rsi_14",          # 모멘텀 과열/침체 (0~100)
    "macd_hist",       # 추세 모멘텀 (종가 대비 정규화)
    "sma20_gap",       # 20일 평균 대비 종가 괴리율
    "ret_20d",         # 20일 수익률
    "atr14_ratio",     # 변동성 (ATR/종가) — regime·리스크 게이트 입력 겸용
    "vol_ratio_20d",   # 거래량/20일 평균 — 참여 강도
    "drawdown_60d",    # 60일 고점 대비 낙폭 — 리스크 상태
)
assert len(FEATURE_NAMES) <= 8, "R17 위반: 관측 feature 는 8개 이하"

MIN_BARS = 40  # MACD(26)+signal(9) 안정 계산 하한


class InsufficientHistoryError(ValueError):
    """feature 계산에 필요한 봉 수 미달."""


@dataclass(frozen=True)
class FeatureSet:
    symbol: str
    asof_day: date
    features: dict[str, float]


def _ema(values: list[float], period: int) -> list[float | None]:
    """SMA seed 표준 EMA. 인덱스 period-1 부터 값이 정의된다."""
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    out[period - 1] = sum(values[:period]) / period
    k = 2 / (period + 1)
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: list[float], period: int = 14) -> float:
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:]):  # Wilder smoothing
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _macd_hist(closes: list[float]) -> float:
    e12, e26 = _ema(closes, 12), _ema(closes, 26)
    macd = [a - b for a, b in zip(e12, e26) if a is not None and b is not None]
    signal = _ema(macd, 9)[-1]
    assert signal is not None  # MIN_BARS 가 보장
    return macd[-1] - signal


def _atr(bars: list[Bar], period: int = 14) -> float:
    trs = [bars[0].high - bars[0].low]
    for prev, cur in zip(bars, bars[1:]):
        trs.append(
            max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        )
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:  # Wilder smoothing
        atr = (atr * (period - 1) + tr) / period
    return atr


def compute_features(symbol: str, bars: list[Bar], asof_day: date) -> FeatureSet:
    """t-1 까지의 일봉 시계열에서 정예 feature 7종 계산.

    - bars 는 day 오름차순이어야 하며 asof_day 이후(당일 포함) 봉은 LeakageError (R2).
    - MIN_BARS 미만이면 InsufficientHistoryError — NaN 침묵 전파 금지.
    """

    if len(bars) < MIN_BARS:
        raise InsufficientHistoryError(
            f"symbol={symbol} n_bars={len(bars)} required={MIN_BARS}"
        )
    if any(b1.day >= b2.day for b1, b2 in zip(bars, bars[1:])):
        raise ValueError(f"symbol={symbol} bars 가 day 오름차순이 아님")
    if bars[-1].day > asof_day - timedelta(days=1):
        raise LeakageError(
            f"leakage symbol={symbol} last_bar={bars[-1].day} asof={asof_day} (same-day 차단)"
        )

    closes = [b.close for b in bars]
    close = closes[-1]
    sma20 = sum(closes[-20:]) / 20
    vol_sma20 = sum(b.volume for b in bars[-20:]) / 20
    high60 = max(closes[-60:])

    features = {
        "rsi_14": _rsi(closes),
        "macd_hist": _macd_hist(closes) / close,  # 종가 대비 정규화 (자산 간 비교 가능)
        "sma20_gap": close / sma20 - 1.0,
        "ret_20d": close / closes[-21] - 1.0,
        "atr14_ratio": _atr(bars) / close,
        "vol_ratio_20d": (bars[-1].volume / vol_sma20) if vol_sma20 > 0 else 0.0,
        "drawdown_60d": close / high60 - 1.0,
    }
    assert set(features) == set(FEATURE_NAMES)
    return FeatureSet(symbol=symbol, asof_day=asof_day, features=features)
