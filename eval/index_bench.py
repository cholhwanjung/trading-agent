"""지수 벤치마크 — 종가 시계열 영속 + arm 대비 비교.

대시보드는 읽기 전용이라 브로커·FRED 를 직접 부르지 않는다. 그래서 일일 루프가 받은
지수 종가를 시장별 상태 파일로 남기고, 화면은 그 파일만 읽는다. 첫 실행이 lookback
창 전체를 받아오므로 별도 백필 절차가 없다.

`eval/meta.py` 가 `meta_shadow.json` 을 소유하는 것과 같은 자리 — 어댑터는 조회만 하고
상태 파일의 규약은 소비자 쪽에 둔다.
"""

from __future__ import annotations

import json
from pathlib import Path

from adapters.market_index import IndexSeries


def index_path(state_dir: Path | str, market: str) -> Path:
    return Path(state_dir) / f"index_series_{market}.json"


def record_index_series(path: Path | str, series: IndexSeries) -> int:
    """지수 종가를 날짜별로 병합 저장(같은 날 재실행은 갱신, 멱등). 새로 추가된 날 수 반환."""
    path = Path(path)
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged = {h["day"]: h["close"] for h in state.get("history") or []}
    before = len(merged)
    merged.update({day.isoformat(): close for day, close in series.closes})
    history = [{"day": d, "close": merged[d]} for d in sorted(merged)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"index": series.name, "source": series.source, "history": history},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return len(merged) - before


def load_index_series(state_dir: Path | str, market: str) -> dict | None:
    """저장된 지수 시계열 → {"index", "source", "history"}. 없으면 None."""
    path = index_path(state_dir, market)
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if not state.get("history"):
        return None
    return state


def index_hist(state_dir: Path | str, market: str) -> list[dict] | None:
    """지수 시계열 → arm equity 와 같은 형태([{day, equity}]).

    `rolling_delta` 는 창 양끝의 비율만 쓰므로 수준(index level)을 그대로 넣어도 된다 —
    정규화는 곡선을 겹쳐 그릴 때만 필요하다(`normalized`).
    """
    state = load_index_series(state_dir, market)
    if state is None:
        return None
    return [{"day": h["day"], "equity": h["close"]} for h in state["history"]]


def normalized(history: list[dict], first_day: str) -> list[dict]:
    """first_day 이후 구간을 그 시작값=1.0 으로 정규화. 시작 전 데이터는 버린다.

    arm 곡선(시작=자기 첫날)과 같은 출발선에 세우기 위한 것 — 지수는 arm 보다 훨씬 긴
    이력을 갖고 있어 그대로 겹치면 비교가 되지 않는다.
    """
    tail = [h for h in history if h["day"] >= first_day]
    if not tail or not tail[0]["equity"]:
        return []
    base = tail[0]["equity"]
    return [{"day": h["day"], "equity": h["equity"] / base} for h in tail]
