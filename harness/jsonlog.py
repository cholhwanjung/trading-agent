"""구조화 JSON Lines 로거 — 자유 텍스트 로그 금지, grep 가능해야 한다.

한 줄 = JSON 객체 1개. 파일은 data/logs/{market}/{YYYY-MM-DD}.jsonl 로 일 단위 분리.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path


def iter_events(root: Path | str, market: str, event: str) -> Iterator[dict]:
    """{root}/{market}/*.jsonl 을 날짜 파일 순으로 읽어 해당 event 레코드만 낸다.

    로그를 읽는 모든 소비자(GUI·context·갭 신호)의 단일 리더. 깨진 줄은
    건너뛴다(append 전용 로그 — 부분 기록 허용). 디렉토리 없으면 빈 이터레이터.
    """
    market_dir = Path(root) / market
    if not market_dir.exists():
        return
    for path in sorted(market_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == event:
                yield record


class JsonlLogger:
    """append 전용 JSON Lines 로거. 이벤트마다 ts·event 키를 강제한다."""

    def __init__(self, root: Path | str = "data/logs") -> None:
        self.root = Path(root)

    def log(self, market: str, event: str, payload: dict) -> Path:
        """이벤트 1건 기록 후 로그 파일 경로 반환."""

        now = datetime.now(timezone.utc)
        record = {"ts": now.isoformat(), "event": event, "market": market, **payload}
        path = self.root / market / f"{now.date().isoformat()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return path
