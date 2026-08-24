"""rolling-k delta — 승격 판정 입력: 상대 성과의 일관성 측정.

누적 delta 하나는 한 번의 행운/불운에 좌우된다. k일 창을 하루씩 굴려
비교 arm 대비 승률(창 delta > 0 비율)을 본다. **유의성은 겹치는 창으로
판단하지 않는다** — 중첩 창은 자기상관으로 표본이 부풀려지므로, 부호검정은
겹치지 않는 k일 청크에만 적용한다 (memory.admission.sign_test_p 재사용).
"""

from __future__ import annotations

from pathlib import Path

from eval.index_bench import index_hist
from eval.meta import combined_index, combined_index_dynamic, load_arm_history
from memory.admission import sign_test_p

ROLLING_K = 20  # 거래일 기준 ~1개월
MIN_CHUNKS_FOR_TEST = 5  # 부호검정 최소 청크 수 (admission 게이트와 동일 기준)


def _align(hist_a: list[dict], hist_b: list[dict]) -> tuple[list[float], list[float]]:
    """공통 날짜 교집합으로 equity 시계열 정렬."""
    a_by_day = {p["day"]: p["equity"] for p in hist_a}
    b_by_day = {p["day"]: p["equity"] for p in hist_b}
    days = sorted(set(a_by_day) & set(b_by_day))
    return [a_by_day[d] for d in days], [b_by_day[d] for d in days]


def rolling_delta(hist_a: list[dict], hist_b: list[dict], k: int = ROLLING_K) -> dict | None:
    """arm A vs B 의 k일 창 상대 성과. 데이터가 k+1 미만이면 None (판단 불가)."""
    ea, eb = _align(hist_a, hist_b)
    n = len(ea)
    if n < k + 1:
        return None

    # 중첩 창 (기술 통계 전용): 하루씩 굴린 k일 수익률 차이
    deltas = [
        (ea[t] / ea[t - k] - 1) - (eb[t] / eb[t - k] - 1) for t in range(k, n)
    ]
    # 비중첩 청크 (유의성 전용): 독립에 가까운 표본
    chunks = [
        (ea[t + k] / ea[t] - 1) - (eb[t + k] / eb[t] - 1) for t in range(0, n - k, k)
    ]
    k_pos = sum(1 for c in chunks if c > 0)
    p_value = sign_test_p(k_pos, len(chunks)) if len(chunks) >= MIN_CHUNKS_FOR_TEST else None

    return {
        "k": k,
        "n_windows": len(deltas),
        "win_rate": sum(1 for d in deltas if d > 0) / len(deltas),
        "mean_delta_pct": sum(deltas) / len(deltas) * 100,
        "latest_delta_pct": deltas[-1] * 100,
        "n_chunks": len(chunks),
        "chunks_positive": k_pos,
        "p_value": p_value,  # None = 청크 부족으로 검정 불가 (겹침 표본으로 대체하지 않는다)
    }


def rolling_report(
    state_dir: Path | str, market: str, k: int = ROLLING_K, index_dir: Path | str | None = None
) -> dict:
    """시장 1곳의 rolling 지표 — memory(llm−llm_base) · alpha(llm−bh) · index(llm−지수).

    승격 판정에 쓰는 것은 memory·alpha 뿐이다. index 는 벤치마크가 유니버스가 아니라
    시장 전체라 "그 기간이 어떤 장이었나"를 답하는 맥락 열이고, 지수 원천이 없는
    시장에서는 None 이다.

    `state_dir` 는 가상 arm 디렉토리(`data/state/virtual`)인데 지수 시계열은 arm 이
    아니라서 그 부모(`data/state`)에 산다 — `index_dir` 기본값이 그 관계를 담는다.
    """
    state_dir = Path(state_dir)
    hists = {arm: load_arm_history(state_dir, market, arm) for arm in ("llm", "llm_base", "bh")}
    idx = index_hist(index_dir if index_dir is not None else state_dir.parent, market)
    return {
        "market": market,
        "memory": rolling_delta(hists["llm"], hists["llm_base"], k) if hists["llm"] else None,
        "alpha": rolling_delta(hists["llm"], hists["bh"], k) if hists["llm"] else None,
        "index": rolling_delta(hists["llm"], idx, k) if hists["llm"] and idx else None,
    }


def _curve_hist(result: dict | None) -> list[dict] | None:
    """combined_index 계열 결과 → rolling_delta 입력 형태([{day, equity}]). None 은 전파."""
    if result is None:
        return None
    return [{"day": p["day"], "equity": p["index"]} for p in result["curve"]]


def meta_rolling_report(state_dir: Path | str, k: int = ROLLING_K) -> dict:
    """결합 지수 층위의 rolling 지표 — memory(llm−llm_base) · alpha(llm−bh).

    `rolling_report` 의 META 판. 시장 하나의 승률은 그 시장의 운·불운에 좌우되고, 셋을
    따로 읽으면 "전체가 이기고 있는가"에 답하지 못한다. 결합은 KPI 행과 같은
    `combined_index`(고정 1/3·리밸런싱 없음)라 화면의 α 와 같은 곡선을 본다.

    지수 열은 없다 — CRYPTO 에 대응하는 지수 원천이 없어 2/3 만으로 결합하면 arm 과
    구성이 달라져 비교가 성립하지 않는다. 지수 대비는 시장별로만 읽는다.
    """
    state_dir = Path(state_dir)
    hists = {arm: _curve_hist(combined_index(state_dir, arm))
             for arm in ("llm", "llm_base", "bh")}
    llm = hists["llm"]
    return {
        "market": "META",
        "memory": rolling_delta(llm, hists["llm_base"], k) if llm and hists["llm_base"] else None,
        "alpha": rolling_delta(llm, hists["bh"], k) if llm and hists["bh"] else None,
    }


def meta_shadow_delta(
    state_dir: Path | str,
    arm: str,
    weights_by_day: dict[str, dict[str, float]],
    k: int = ROLLING_K,
) -> dict | None:
    """동적 메타 배분 vs 고정 균등의 rolling delta.

    두 지수를 **같은 리밸런싱 방법**(combined_index_dynamic)으로 산출 — 가중치만 달라
    배분 스킬을 분리 측정한다. dynamic − equal 의 창 승률·비중첩 청크 부호검정.
    집행 승격은 이 델타>0·유의 + 실계좌 전환 후. 데이터 부족 시 None.
    """
    dyn = combined_index_dynamic(state_dir, arm, weights_by_day)
    fixed = combined_index_dynamic(state_dir, arm, {})  # 빈 dict = 고정 균등 baseline
    if dyn is None or fixed is None:
        return None
    return rolling_delta(_curve_hist(dyn), _curve_hist(fixed), k)
