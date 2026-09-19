"""노출 계측 — B&H 대비 갭을 **현금 드래그**와 **종목선택**으로 가른다.

`α vs B&H` 한 숫자는 "졌다"만 알려주고 왜 졌는지는 말하지 않는다. 두 원인은 처방이
완전히 다르다 — 노출이 낮아 잃은 것이라면 배분 정책의 문제고, 같은 노출에서 잘못
골라 잃은 것이라면 선별의 문제다.

분해는 **합성 포트폴리오**로 한다: 매일 llm 이 실제로 가졌던 노출(1−현금)만큼만 B&H
바스켓을 든 가상 포트폴리오를 굴린다. 가상 arm 은 결정을 다음 봉 시가에 체결하므로 한
결정의 노출은 **체결 시가부터 다음 체결 시가까지** 유효하다 — 구간을 종가로 끊으면 밤사이
갭이 엉뚱한 결정의 노출에 곱해져 '종목선택' 으로 둔갑한다.

    synth ← synth · (1 + exposure_t · r_bh(시가_{t+1} → 시가_{t+2}))

    현금 드래그 = synth 수익 − bh 수익    (노출이 낮아서 잃은 몫)
    종목선택   = llm 수익  − synth 수익   (같은 노출에서 무엇을 골랐나)

두 항의 합이 정확히 `llm − bh` = 화면의 α 다. 순수 계측이며 결정·리스크에 개입하지
않는다 — 노출을 바꾸는 처방은 플레이북·프롬프트 소관이라 별도 승인 사항이다.
"""

from __future__ import annotations

from pathlib import Path

from eval.meta import MARKET_CAPITAL_WEIGHTS, load_arm_history
from harness.jsonlog import iter_events


def cash_weights(log_dir: Path | str, market: str, arm: str = "llm") -> dict[str, float]:
    """virtual_step 로그 → {day: CASH 비중}. 같은 날 재실행은 최신 레코드로 덮어쓴다."""
    out: dict[str, float] = {}
    for rec in iter_events(Path(log_dir), market, "virtual_step"):
        if rec.get("portfolio") != arm:
            continue
        day = str(rec.get("day", ""))[:10]
        weights = rec.get("weights") or {}
        if day and "CASH" in weights:
            out[day] = float(weights["CASH"])
    return out


def exposure_summary(log_dir: Path | str, market: str, arm: str = "llm") -> dict | None:
    """노출(1−현금)의 평균·최근·범위. 기록이 없으면 None."""
    cash = cash_weights(log_dir, market, arm)
    if not cash:
        return None
    days = sorted(cash)
    exposure = [1.0 - cash[d] for d in days]
    return {
        "market": market,
        "n": len(exposure),
        "mean": sum(exposure) / len(exposure),
        "last": exposure[-1],
        "last_day": days[-1],
        "min": min(exposure),
        "max": max(exposure),
    }


def alpha_decomposition(
    state_dir: Path | str, log_dir: Path | str, market: str, arm: str = "llm"
) -> dict | None:
    """시장 1곳의 α 갭 분해. arm·bh 이력이나 현금 기록이 없으면 None.

    합성 곡선은 bh 의 **일간 수익률**에 그날의 노출을 곱해 굴린다 — bh 가 그 시장의
    "전부 투자했을 때"이므로, 노출만 낮춘 반사실이 정확히 이 곡선이다.
    """
    state_dir = Path(state_dir)
    llm_hist = load_arm_history(state_dir, market, arm)
    bh_hist = load_arm_history(state_dir, market, "bh")
    cash = cash_weights(log_dir, market, arm)
    if not llm_hist or not bh_hist or not cash:
        return None

    bh_by_day = {p["day"]: p["equity"] for p in bh_hist}
    bh_open = {p["day"]: p.get("equity_open") for p in bh_hist}
    days = sorted(bh_by_day)

    def fill_mark(i: int) -> float:
        """days[i] 봉의 체결 직전(시가) 평가액. 시가 기록이 없는 이력은 직전 종가로 친다."""
        return bh_open[days[i]] or bh_by_day[days[i - 1]]

    synth = 1.0
    used: list[float] = []
    for i in range(1, len(days)):
        c = cash.get(days[i - 1])  # days[i] 시가에 체결되는 결정
        start = fill_mark(i)
        # 다음 체결 시가까지. 마지막 구간은 아직 다음 봉이 없으므로 종가에서 끊는다
        end = fill_mark(i + 1) if i + 1 < len(days) else bh_by_day[days[i]]
        if c is None or start <= 0:
            continue  # 그날 llm 이 돌지 않았다 — 노출을 지어내지 않고 건너뛴다
        exposure = 1.0 - c
        used.append(exposure)
        synth *= 1.0 + exposure * (end / start - 1.0)
    if not used:
        return None

    llm_ret = (llm_hist[-1]["equity"] / llm_hist[0]["equity"] - 1.0) * 100
    bh_ret = (bh_by_day[days[-1]] / bh_by_day[days[0]] - 1.0) * 100
    synth_ret = (synth - 1.0) * 100
    return {
        "market": market,
        "n": len(used),
        "mean_exposure": sum(used) / len(used),
        "llm_ret_pct": llm_ret,
        "bh_ret_pct": bh_ret,
        "synth_ret_pct": synth_ret,
        "cash_drag_pct": synth_ret - bh_ret,  # ≤ 0 (노출 ≤ 1 이면)
        "selection_pct": llm_ret - synth_ret,
        "alpha_pct": llm_ret - bh_ret,  # = cash_drag + selection
    }


def meta_exposure(
    log_dir: Path | str,
    markets: tuple[str, ...] = ("CRYPTO", "US", "KR"),
    weights: dict[str, float] | None = None,
    arm: str = "llm",
) -> dict | None:
    """시장별 평균 노출을 자본 비중으로 가중 결합. 데이터 있는 시장으로 재정규화."""
    weights = weights or MARKET_CAPITAL_WEIGHTS
    parts = {m: s for m in markets if (s := exposure_summary(log_dir, m, arm))}
    if not parts:
        return None
    total_w = sum(weights.get(m, 0.0) for m in parts)
    if total_w <= 0:
        return None
    return {
        "n_markets": len(parts),
        "mean": sum(weights[m] / total_w * s["mean"] for m, s in parts.items()),
        "last": sum(weights[m] / total_w * s["last"] for m, s in parts.items()),
        "by_market": {m: s["mean"] for m, s in parts.items()},
    }


def meta_alpha_decomposition(
    state_dir: Path | str,
    log_dir: Path | str,
    markets: tuple[str, ...] = ("CRYPTO", "US", "KR"),
    weights: dict[str, float] | None = None,
    arm: str = "llm",
) -> dict | None:
    """시장별 분해를 자본 비중으로 결합 — KPI 행의 α 와 같은 축.

    `combined_index` 가 시장별 ratio 의 **선형** 결합(리밸런싱 없음)이라 각 항도 같은
    비중으로 더해진다. 그래서 결합된 `cash_drag + selection` 이 화면의 α 와 일치한다.
    """
    weights = weights or MARKET_CAPITAL_WEIGHTS
    parts = {
        m: d for m in markets
        if (d := alpha_decomposition(state_dir, log_dir, m, arm))
    }
    if not parts:
        return None
    total_w = sum(weights.get(m, 0.0) for m in parts)
    if total_w <= 0:
        return None

    def blend(key: str) -> float:
        return sum(weights[m] / total_w * d[key] for m, d in parts.items())

    return {
        "markets": sorted(parts),
        "cash_drag_pct": blend("cash_drag_pct"),
        "selection_pct": blend("selection_pct"),
        "alpha_pct": blend("alpha_pct"),
        "mean_exposure": blend("mean_exposure"),
        "by_market": parts,
    }
