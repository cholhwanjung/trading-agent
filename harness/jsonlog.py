"""구조화 JSON Lines 로거 — 자유 텍스트 로그 금지, grep 가능해야 한다.

한 줄 = JSON 객체 1개. 파일은 data/logs/{market}/{YYYY-MM-DD}.jsonl 로 일 단위 분리.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


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
