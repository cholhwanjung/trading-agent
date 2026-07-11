"""outer loop verifier — 경량 IC 백테스터 ([ADR-016] · R11).

백테스트는 **신호 스크리닝 전용**([ADR-002]) — 승격 근거가 아니다.
지표: 일별 횡단면 rank IC(스피어만) vs 익일 수익률 → mean IC · ICIR.
train/OOS 시계열 분할(70/30)로 admission 4단계 중 OOS 검증을 수행한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_lab.dsl import evaluate

MIN_CROSS_SECTION = 4  # 하루 IC 계산에 필요한 최소 유효 자산 수


@dataclass(frozen=True)
class ICStats:
    mean_ic: float
    icir: float
    n_days: int
    positive_ratio: float  # IC > 0 비율


@dataclass(frozen=True)
class BacktestResult:
    train: ICStats
    oos: ICStats
    scores: np.ndarray  # (T, N) 팩터 행렬 — 라이브러리 상관 체크용


def _rank_row(row: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(row)
    out = np.full_like(row, np.nan)
    if valid.sum() >= 2:
        out[valid] = row[valid].argsort().argsort().astype(float)
    return out


def daily_rank_ic(scores: np.ndarray, fwd_returns: np.ndarray) -> np.ndarray:
    """일별 스피어만 IC 벡터 (계산 불가일은 NaN)."""
    T = scores.shape[0]
    ics = np.full(T, np.nan)
    for t in range(T):
        f, r = scores[t], fwd_returns[t]
        valid = ~np.isnan(f) & ~np.isnan(r)
        if valid.sum() < MIN_CROSS_SECTION:
            continue
        rf, rr = _rank_row(np.where(valid, f, np.nan)), _rank_row(np.where(valid, r, np.nan))
        rf, rr = rf[valid], rr[valid]
        sf, sr = rf.std(), rr.std()
        if sf > 0 and sr > 0:
            ics[t] = np.corrcoef(rf, rr)[0, 1]
    return ics


def _stats(ics: np.ndarray) -> ICStats:
    valid = ics[~np.isnan(ics)]
    if len(valid) == 0:
        return ICStats(0.0, 0.0, 0, 0.0)
    mean = float(valid.mean())
    std = float(valid.std())
    return ICStats(
        mean_ic=mean,
        icir=mean / std if std > 0 else 0.0,
        n_days=len(valid),
        positive_ratio=float((valid > 0).mean()),
    )


def forward_returns(close: np.ndarray) -> np.ndarray:
    """t 시점 팩터 vs t→t+1 수익률. 마지막 날은 NaN (미래 없음)."""
    fwd = np.full_like(close, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[:-1] = close[1:] / close[:-1] - 1.0
    return fwd


def run_backtest(
    expression: str, panel: dict[str, np.ndarray], train_ratio: float = 0.7
) -> BacktestResult:
    scores = evaluate(expression, panel)
    fwd = forward_returns(panel["close"])
    ics = daily_rank_ic(scores, fwd)
    split = int(len(ics) * train_ratio)
    return BacktestResult(train=_stats(ics[:split]), oos=_stats(ics[split:]), scores=scores)


def score_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """두 팩터 행렬의 겹치는 유효 구간 상관 — 라이브러리 중복 통제 입력."""
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 30:
        return 0.0
    va, vb = a[valid], b[valid]
    sa, sb = va.std(), vb.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.corrcoef(va, vb)[0, 1])
