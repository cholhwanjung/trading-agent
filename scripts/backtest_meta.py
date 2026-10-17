"""시장 간 배분 규칙 스크리닝 백테스트 — 균등 1/3 을 이겨야 할 baseline 으로 둔다.

백테스트는 **스크리닝 전용**이다. 여기서 고른 규칙은 라이브 shadow 로 다시 재고, 승격은
라이브 델타로만 판정한다. 입력은 `scripts/fetch_meta_history.py` 가 저장한 지수 프록시
일봉(라이브 국면 계산과 같은 프록시)이고, 측정 방법은 라이브 채점기와 같다 —
리밸런싱 결합 지수 + 20일 비중첩 청크 부호검정(`eval.rolling.rolling_delta`).

규칙(전부 결정론·t−1 상한·일간 무료 데이터):
    equal        균등 1/3 (baseline)
    invvol{N}    역변동성 — N 거래일 실현변동성의 역수에 비례(리스크 패리티 근사)
    sma{N}       종가 > N 거래일 단순이동평균이면 1/3 보유, 아니면 그 몫은 현금
    tsmom{N}     N 거래일 수익률 > 0 이면 1/3 보유, 아니면 현금
    oneil        라이브 shadow 와 같은 규칙 — 국면 FSM 점수 × max_tilt 0.10 틸트
    invvol60+sma210   역변동성 anchor 위에 SMA 필터
    invvol60+oneil    역변동성 anchor 위에 국면 틸트

리밸런싱 cadence:
    daily    매일 목표로 재조정 — 라이브 채점기(`combined_index_dynamic`)와 동일
    monthly  매월 첫 거래일에만 재조정, 그 사이 비중은 수익률대로 표류
    band5    목표와 5%p 이상 벌어진 날만 재조정 — 수동 이체 현실에 가장 가깝다

스크리닝 판정(규칙마다, 미리 고정):
    PASS_ALPHA  (a) daily cadence 에서 균등 대비 비중첩 20일 청크 부호검정 p < 0.05
                (b) 연도별 균등 대비 델타가 양수인 해가 70% 이상
                (c) 파라미터 이웃 전부가 균등 대비 평균 청크 델타 > 0
    RISK_ONLY   (a) 실패이지만 MDD 가 균등보다 20% 이상 낮고 Sharpe 도 높다 — 수익
                알파가 아니라 낙폭 축소. 베이스라인 교체 근거는 되지만 "알파" 는 아니다
    FAIL        그 외

verify(스크립트가 직접 찍는다):
    engine_matches_live_scorer  daily·비용 0 의 결합 지수가 `eval.meta.combined_index_dynamic`
                                과 전 구간 1e-6 이내로 일치하는가(채점기가 곡선을 소수 6자리로 반올림한다)
    oneil_matches_live_shadow   oneil 규칙의 가중치가 라이브 `meta_shadow.json` 과 같은 날
                                얼마나 일치하는가(국면 계산 시각 차이로 전환일은 어긋날 수 있다)

사용: .venv/bin/python scripts/backtest_meta.py [--cost-bp 10] [--start 2018-09-01]
출력: 표(stdout) + data/research/meta_backtest.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.base import Bar  # noqa: E402
from eval.meta import combined_index_dynamic, load_meta_shadow, max_drawdown  # noqa: E402
from eval.rolling import ROLLING_K, rolling_delta  # noqa: E402
from regime.live import LOOKBACK_DAYS  # noqa: E402
from regime.meta import MarketSignal, propose_meta_weights  # noqa: E402
from regime.pulse import classify_regime  # noqa: E402

HIST_DIR = ROOT / "data" / "research" / "meta_hist"
OUT_PATH = ROOT / "data" / "research" / "meta_backtest.json"
SHADOW_PATH = ROOT / "data" / "state" / "meta_shadow.json"
MARKETS = ("CRYPTO", "KR", "US")
WARMUP_BARS = 260  # 가장 긴 신호(tsmom252)보다 길게 — 전 규칙이 같은 날 시작
BAND = 0.05
P_ALPHA = 0.05
YEAR_SHARE = 0.70
MDD_RELIEF = 0.20

# 규칙 → (anchor, filter, tilt, 파라미터). 이웃은 같은 family 의 다른 파라미터.
RULES: dict[str, dict] = {
    "equal": {},
    "invvol40": {"anchor": ("invvol", 40)},
    "invvol60": {"anchor": ("invvol", 60)},
    "invvol120": {"anchor": ("invvol", 120)},
    "sma150": {"filter": ("sma", 150)},
    "sma210": {"filter": ("sma", 210)},
    "sma250": {"filter": ("sma", 250)},
    "tsmom126": {"filter": ("tsmom", 126)},
    "tsmom252": {"filter": ("tsmom", 252)},
    "oneil": {"tilt": "oneil"},
    "invvol60+sma210": {"anchor": ("invvol", 60), "filter": ("sma", 210)},
    "invvol60+oneil": {"anchor": ("invvol", 60), "tilt": "oneil"},
}
FAMILY_NEIGHBORS = {
    "invvol": ["invvol40", "invvol60", "invvol120"],
    "sma": ["sma150", "sma210", "sma250"],
    "tsmom": ["tsmom126", "tsmom252"],
}
CADENCES = ("daily", "monthly", "band5")


# ── 데이터 ────────────────────────────────────────────────────────────────────
class Series:
    """시장 1곳의 일봉 + 봉 인덱스 기준 신호 캐시."""

    def __init__(self, market: str) -> None:
        raw = json.loads((HIST_DIR / f"{market}.json").read_text(encoding="utf-8"))
        self.market = market
        self.bars = [
            Bar(
                date.fromisoformat(r["day"]),
                r["open"],
                r["high"],
                r["low"],
                r["close"],
                r["volume"],
            )
            for r in raw["bars"]
        ]
        self.days = [b.day for b in self.bars]
        self.close = [b.close for b in self.bars]
        self.rets = [None] + [
            self.close[i] / self.close[i - 1] - 1 for i in range(1, len(self.close))
        ]
        self.index_of = {d: i for i, d in enumerate(self.days)}
        self._regime: dict[int, str | None] = {}

    def last_index_leq(self, d: date, hint: int) -> int:
        """day ≤ d 인 마지막 봉 인덱스(hint 에서 전진). 없으면 -1."""
        i = hint
        while i + 1 < len(self.days) and self.days[i + 1] <= d:
            i += 1
        return i

    def sma_above(self, i: int, n: int) -> bool | None:
        if i < n - 1:
            return None
        return self.close[i] > sum(self.close[i - n + 1 : i + 1]) / n

    def tsmom_pos(self, i: int, n: int) -> bool | None:
        if i < n:
            return None
        return self.close[i] > self.close[i - n]

    def vol(self, i: int, n: int) -> float | None:
        if i < n:
            return None
        return statistics.pstdev(self.rets[i - n + 1 : i + 1])

    def regime(self, i: int, asof: date) -> str | None:
        """라이브와 같은 창 — [asof−LOOKBACK_DAYS, asof−1] 봉으로 FSM replay."""
        if i < 0:
            return None
        if i not in self._regime:
            lo = asof - timedelta(days=LOOKBACK_DAYS)
            j = i
            while j > 0 and self.days[j - 1] >= lo:
                j -= 1
            r = classify_regime(self.bars[j : i + 1])
            self._regime[i] = r.state if r else None
        return self._regime[i]


# ── 규칙 ─────────────────────────────────────────────────────────────────────
def rule_weights(
    spec: dict, series: dict[str, Series], idx: dict[str, int], asof: date
) -> dict[str, float]:
    """asof 하루의 목표 비중(현금 = 1 − 합). 신호 미성숙(None)은 보유·균등으로 처리."""
    anchor = {m: 1.0 / len(MARKETS) for m in MARKETS}
    if "anchor" in spec:
        _, n = spec["anchor"]
        inv = {m: 1.0 / v if (v := series[m].vol(idx[m], n)) else None for m in MARKETS}
        if all(inv.values()):
            total = sum(inv.values())
            anchor = {m: inv[m] / total for m in MARKETS}
    weights = dict(anchor)
    if "tilt" in spec:
        signals = [MarketSignal(m, series[m].regime(idx[m], asof)) for m in MARKETS]
        weights = propose_meta_weights(signals, anchor=anchor).weights
    if "filter" in spec:
        kind, n = spec["filter"]
        for m in MARKETS:
            on = series[m].sma_above(idx[m], n) if kind == "sma" else series[m].tsmom_pos(idx[m], n)
            if on is False:
                weights[m] = 0.0
    return weights


# ── 엔진 ─────────────────────────────────────────────────────────────────────
def run(
    spec: dict, series: dict[str, Series], calendar: list[date], cadence: str, cost: float
) -> tuple[list[dict], dict[str, dict[str, float]], float]:
    """결합 지수 곡선·일별 목표 비중·연간 회전율. 비중은 t−1 정보, 수익률은 t 봉."""
    holdings = {m: 0.0 for m in MARKETS}
    cash = 1.0
    idx = {m: series[m].last_index_leq(calendar[0] - timedelta(days=1), 0) for m in MARKETS}
    curve: list[dict] = []
    targets: dict[str, dict[str, float]] = {}
    turnover = 0.0
    prev_month = None
    for d in calendar:
        prev = d - timedelta(days=1)
        for m in MARKETS:
            idx[m] = series[m].last_index_leq(prev, idx[m])
        w = rule_weights(spec, series, idx, d)
        targets[d.isoformat()] = w
        wealth = cash + sum(holdings.values())
        current = {m: holdings[m] / wealth for m in MARKETS}
        if cadence == "daily":
            rebalance = True
        elif cadence == "monthly":
            rebalance = prev_month != (d.year, d.month)
        else:
            rebalance = not curve or any(abs(current[m] - w[m]) > BAND for m in MARKETS)
        if rebalance:
            delta = sum(abs(w[m] - current[m]) for m in MARKETS) / 2
            turnover += delta
            wealth *= 1 - delta * cost
            holdings = {m: wealth * w[m] for m in MARKETS}
            cash = wealth * (1 - sum(w.values()))
        prev_month = (d.year, d.month)
        if curve:  # 첫날은 1.0 에서 시작(수익률 미적용) — 라이브 채점기의 index_0 정의와 동일
            for m in MARKETS:
                i = series[m].index_of.get(d)
                if i is not None and i > 0:
                    holdings[m] *= 1 + series[m].rets[i]
        curve.append({"day": d.isoformat(), "equity": cash + sum(holdings.values())})
    years = (calendar[-1] - calendar[0]).days / 365.25
    return curve, targets, turnover / years


def metrics(curve: list[dict]) -> dict:
    eq = [p["equity"] for p in curve]
    years = (
        date.fromisoformat(curve[-1]["day"]) - date.fromisoformat(curve[0]["day"])
    ).days / 365.25
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
    ppy = len(rets) / years
    mu, sd = statistics.mean(rets), statistics.pstdev(rets)
    return {
        "ret_pct": (eq[-1] - 1) * 100,
        "cagr_pct": (eq[-1] ** (1 / years) - 1) * 100,
        "vol_pct": sd * math.sqrt(ppy) * 100,
        "sharpe": (mu / sd * math.sqrt(ppy)) if sd > 0 else 0.0,
        "mdd_pct": max_drawdown(eq) * 100,
    }


def yearly_returns(curve: list[dict]) -> dict[int, float]:
    by_year: dict[int, list[float]] = {}
    for p in curve:
        by_year.setdefault(date.fromisoformat(p["day"]).year, []).append(p["equity"])
    out, prev_end = {}, None
    for y in sorted(by_year):
        start = prev_end if prev_end is not None else by_year[y][0]
        out[y] = by_year[y][-1] / start - 1
        prev_end = by_year[y][-1]
    return out


# ── verify ───────────────────────────────────────────────────────────────────
def verify_engine(
    series: dict[str, Series], calendar: list[date], targets: dict, curve: list[dict]
) -> float:
    """daily·비용 0 곡선 vs 라이브 채점기 `combined_index_dynamic` — 최대 절대 오차."""
    with tempfile.TemporaryDirectory() as tmp:
        for m in MARKETS:
            hist = [
                {"day": d.isoformat(), "equity": series[m].close[series[m].index_of[d]]}
                for d in calendar
                if d in series[m].index_of
            ]
            (Path(tmp) / f"{m}_proxy.json").write_text(
                json.dumps({"history": hist}), encoding="utf-8"
            )
        ref = combined_index_dynamic(tmp, "proxy", targets)
    ref_by_day = {p["day"]: p["index"] for p in ref["curve"]}
    return max(abs(ref_by_day[p["day"]] - p["equity"]) for p in curve if p["day"] in ref_by_day)


def verify_shadow(targets: dict[str, dict[str, float]]) -> dict:
    """oneil 규칙 가중치 vs 라이브 shadow 원장 — 같은 날 일치 비율과 불일치 날짜."""
    live = load_meta_shadow(SHADOW_PATH)
    common = [d for d in live if d in targets and set(live[d]) == set(MARKETS)]
    mismatch = [d for d in common if max(abs(live[d][m] - targets[d][m]) for m in MARKETS) > 1e-6]
    return {
        "n": len(common),
        "match_rate": 1 - len(mismatch) / len(common) if common else None,
        "mismatch_days": mismatch,
    }


# ── 판정 ─────────────────────────────────────────────────────────────────────
def verdict(name: str, rows: dict[str, dict]) -> str:
    r, base = rows[name], rows["equal"]
    vs = r["vs_equal"]
    p_ok = vs["p_value"] is not None and vs["p_value"] < P_ALPHA
    years = r["yearly_delta_pct"]
    year_ok = sum(1 for v in years.values() if v > 0) / len(years) >= YEAR_SHARE
    fam = next((f for f, names in FAMILY_NEIGHBORS.items() if name in names), None)
    neighbors = FAMILY_NEIGHBORS.get(fam, [name])
    neigh_ok = all(rows[n]["vs_equal"]["mean_delta_pct"] > 0 for n in neighbors)
    if p_ok and year_ok and neigh_ok:
        return "PASS_ALPHA"
    mdd_ok = r["mdd_pct"] <= base["mdd_pct"] * (1 - MDD_RELIEF) and r["sharpe"] > base["sharpe"]
    return "RISK_ONLY" if mdd_ok else "FAIL"


# ── main ─────────────────────────────────────────────────────────────────────
def main(cost_bp: float, start: date | None) -> int:
    series = {m: Series(m) for m in MARKETS}
    for m in MARKETS:  # 분할·오염 봉 점검 — 일간 ±30% 는 지수 프록시에서 데이터 오류다
        bad = [
            (series[m].days[i].isoformat(), round(r, 3))
            for i, r in enumerate(series[m].rets)
            if r is not None and abs(r) > 0.30
        ]
        print(
            f"data market={m} n={len(series[m].days)} first={series[m].days[0]} "
            f"last={series[m].days[-1]} extreme_moves={bad}"
        )
    first_ready = max(series[m].days[WARMUP_BARS] for m in MARKETS)
    start = max(start, first_ready) if start else first_ready
    calendar = sorted({d for s in series.values() for d in s.days if d >= start})
    cost = cost_bp / 10_000
    print(f"window start={calendar[0]} end={calendar[-1]} days={len(calendar)} cost_bp={cost_bp}")

    results: dict[str, dict[str, dict]] = {c: {} for c in CADENCES}
    oneil_targets: dict = {}
    for cadence in CADENCES:
        curves: dict[str, list[dict]] = {}
        for name, spec in RULES.items():
            curve, targets, tover = run(spec, series, calendar, cadence, cost)
            curves[name] = curve
            results[cadence][name] = {**metrics(curve), "turnover_per_year": tover}
            if cadence == "daily" and name == "oneil":
                oneil_targets = targets
                if cost == 0:
                    err = verify_engine(series, calendar, targets, curve)
                    print(
                        f"verify engine_matches_live_scorer max_abs_err={err:.2e} ok={err < 1e-6}"
                    )
        base_years = yearly_returns(curves["equal"])
        for name in RULES:
            vs = (
                rolling_delta(curves[name], curves["equal"], ROLLING_K) if name != "equal" else None
            )
            ys = yearly_returns(curves[name])
            results[cadence][name]["vs_equal"] = vs or {
                "p_value": None,
                "mean_delta_pct": 0.0,
                "win_rate": None,
                "chunks_positive": 0,
                "n_chunks": 0,
            }
            results[cadence][name]["yearly_delta_pct"] = {
                y: (ys[y] - base_years[y]) * 100 for y in ys
            }
        for name in RULES:
            results[cadence][name]["verdict"] = (
                verdict(name, results[cadence]) if name != "equal" else "baseline"
            )

    if cost != 0:  # 엔진 동일성은 비용 0 에서만 정의된다
        curve0, targets0, _ = run(RULES["oneil"], series, calendar, "daily", 0.0)
        err = verify_engine(series, calendar, targets0, curve0)
        print(f"verify engine_matches_live_scorer max_abs_err={err:.2e} ok={err < 1e-6}")
    shadow = verify_shadow(oneil_targets)
    print(
        f"verify oneil_matches_live_shadow n={shadow['n']} match_rate={shadow['match_rate']} "
        f"mismatch_days={shadow['mismatch_days']}"
    )

    for cadence in CADENCES:
        rows = results[cadence]
        print(f"\n=== cadence={cadence} (vs equal: k={ROLLING_K} 비중첩 청크 부호검정) ===")
        print(
            f"{'rule':18}{'cagr%':>8}{'vol%':>8}{'sharpe':>8}{'mdd%':>8}{'turn/yr':>9}"
            f"{'win':>6}{'chunks+':>9}{'p':>8}{'yrs+':>6}  verdict"
        )
        for name, r in rows.items():
            vs = r["vs_equal"]
            yrs = r["yearly_delta_pct"]
            yrs_pos = (
                f"{sum(1 for v in yrs.values() if v > 0)}/{len(yrs)}" if name != "equal" else "-"
            )
            print(
                f"{name:18}{r['cagr_pct']:8.2f}{r['vol_pct']:8.2f}{r['sharpe']:8.2f}{r['mdd_pct']:8.2f}"
                f"{r['turnover_per_year']:9.2f}"
                f"{(vs['win_rate'] if vs['win_rate'] is not None else float('nan')):6.2f}"
                f"{vs['chunks_positive']:>4}/{vs['n_chunks']:<4}"
                f"{(vs['p_value'] if vs['p_value'] is not None else float('nan')):8.3f}"
                f"{yrs_pos:>6}  {r['verdict']}"
            )
        years = sorted(next(iter(rows.values()))["yearly_delta_pct"])
        print(f"{'yearly delta% vs equal':18}" + "".join(f"{y:>8}" for y in years))
        for name, r in rows.items():
            if name != "equal":
                print(f"{name:18}" + "".join(f"{r['yearly_delta_pct'][y]:8.2f}" for y in years))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "window": {
                    "start": calendar[0].isoformat(),
                    "end": calendar[-1].isoformat(),
                    "days": len(calendar),
                },
                "cost_bp": cost_bp,
                "verify_shadow": shadow,
                "results": results,
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten={OUT_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-bp", type=float, default=10.0, help="회전율 단위당 비용(bp)")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    raise SystemExit(main(args.cost_bp, args.start))
