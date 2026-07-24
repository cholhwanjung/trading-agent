"""시장별 최신 regime 의 cross-job 공유 상태.

시장은 장 시간이 달라 별도 잡으로 실행된다(예: KR 10:00 / CRYPTO·US 23:00). 각 잡이
자기 시장의 regime 만으로 메타 제안을 만들면 다른 시장이 빠진 부분 제안이 되고, 같은 날
기록을 덮어써 먼저 실행된 시장이 탈락한다. 이를 막기 위해 각 잡은 자기 시장의 최신 regime
을 이 공유 상태에 **병합**(다른 시장 항목 보존)하고, 메타 제안은 여기서 전 시장을 읽는다.

jsonl 감사 로그(append-only)와는 목적이 다르다 — 이 파일은 '시장별 현재 regime' 캐시
(mutable, 결정 입력). 순수 함수는 아니지만 regime/meta.py 의 순수성은 유지한다.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from regime.meta import MarketSignal


def update_regime_signal(
    path: Path | str, market: str, state: str | None, drawdown: float, asof_day: date
) -> None:
    """시장 1곳의 최신 regime 을 공유 상태에 병합 기록(다른 시장 항목은 보존)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[market] = {"state": state, "drawdown": drawdown, "asof": asof_day.isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_market_signals(path: Path | str) -> list[MarketSignal]:
    """공유 regime 상태 → MarketSignal 목록(파일에 있는 전 시장). 없으면 []."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        MarketSignal(market, entry.get("state"), drawdown=entry.get("drawdown", 0.0))
        for market, entry in sorted(data.items())
    ]
