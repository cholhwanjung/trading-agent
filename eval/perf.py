"""위험조정 성과 지표 — 가상 arm equity 곡선에서 파생하는 순수 함수.

tear sheet 표준 지표(Sharpe·Sortino·Calmar·변동성·승률)를 일간 수익률 기준으로 계산.
무위험수익률 rf=0 가정(짧은 라이브 표본·크립토 관행). 연율화 계수는 시장별로 다르므로
(크립토 365, 주식 ~252) 호출부가 넘긴다. 표본이 짧으면 지표는 노이즈가 크다 — n 을
함께 반환해 소비자가 신뢰도를 표시하게 한다.
"""

from __future__ import annotations

from math import sqrt

from eval.meta import max_drawdown


def daily_returns(equity: list[float]) -> list[float]:
    """equity 곡선 → 일간 단순수익률. 0 이하 값은 건너뛴다(로그 아님, 분모 보호)."""
    out: list[float] = []
    for prev, cur in zip(equity, equity[1:]):
        if prev > 0:
            out.append(cur / prev - 1.0)
    return out


def drawdown_series(equity: list[float]) -> list[float]:
    """언더워터 곡선 — 각 시점의 직전 고점 대비 낙폭(≤ 0). 빈 곡선은 []."""
    out: list[float] = []
    peak = equity[0] if equity else 0.0
    for v in equity:
        peak = max(peak, v)
        out.append(v / peak - 1.0 if peak > 0 else 0.0)
    return out


def perf_stats(equity: list[float], periods_per_year: float = 252.0) -> dict | None:
    """equity 곡선 → 위험조정 지표 dict. 수익률 관측 2개 미만이면 None.

    연변동성·Sharpe·Sortino 는 √periods_per_year 로 연율화, 연수익률은 단순 연율화
    (mean × periods_per_year — 짧은 표본에서 기하 CAGR 의 폭주를 피함). Calmar 는
    연수익률/MDD. 분모가 0(무변동·무손실)인 지표는 None.
    """
    rets = daily_returns(equity)
    n = len(rets)
    if n < 2:
        return None
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)  # 표본분산(ddof=1)
    std = sqrt(var)
    downside = sqrt(sum(min(r, 0.0) ** 2 for r in rets) / n)  # 하방편차(목표 0)
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    mdd = max_drawdown(equity)
    ann_return = mean * periods_per_year
    return {
        "n": n,
        "total_return": equity[-1] / equity[0] - 1.0 if equity[0] > 0 else 0.0,
        "ann_return": ann_return,
        "ann_vol": std * sqrt(periods_per_year),
        "sharpe": (mean / std * sqrt(periods_per_year)) if std > 0 else None,
        "sortino": (mean / downside * sqrt(periods_per_year)) if downside > 0 else None,
        "calmar": (ann_return / mdd) if mdd > 0 else None,
        "mdd": mdd,
        "win_rate": len(wins) / n,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "best": max(rets),
        "worst": min(rets),
    }
