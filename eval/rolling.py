"""rolling-k delta — 승격 판정 입력: 상대 성과의 일관성 측정.

누적 delta 하나는 한 번의 행운/불운에 좌우된다. k일 창을 하루씩 굴려
비교 arm 대비 승률(창 delta > 0 비율)을 본다. **유의성은 겹치는 창으로
판단하지 않는다** — 중첩 창은 자기상관으로 표본이 부풀려지므로, 부호검정은
겹치지 않는 k일 청크에만 적용한다 (memory.admission.sign_test_p 재사용).
"""

from __future__ import annotations

from pathlib import Path

from eval.meta import _load_history
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


def rolling_report(state_dir: Path | str, market: str, k: int = ROLLING_K) -> dict:
    """시장 1곳의 승격 판정용 rolling 지표 — memory(llm−llm_base) · alpha(llm−bh)."""
    state_dir = Path(state_dir)
    hists = {arm: _load_history(state_dir, market, arm) for arm in ("llm", "llm_base", "bh")}
    return {
        "market": market,
        "memory": rolling_delta(hists["llm"], hists["llm_base"], k) if hists["llm"] else None,
        "alpha": rolling_delta(hists["llm"], hists["bh"], k) if hists["llm"] else None,
    }
