"""메모리 3-store — SQLite 단일 파일, 시장별 네임스페이스 격리.

한 테이블 + (market, store) 네임스페이스. 임베딩은 float32 blob — 규모가 수백 건
수준이라 코사인은 Python 으로 충분(sqlite-vec/FAISS 는 규모 커지면).

store 종류:
- episodic   — 매매 1건 전체 맥락 (원천 기록, outcome 은 다음 날 소급 기입)
- semantic   — admission 통과 교훈 (probation → active)
- procedural — 플레이북·Forbidden 실패 패턴 (하드 veto 소스)
"""

from __future__ import annotations

import json
import sqlite3
import struct
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

STORES = ("episodic", "semantic", "procedural")
STATUSES = ("active", "probation", "retired")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id          TEXT PRIMARY KEY,
  market      TEXT NOT NULL,
  store       TEXT NOT NULL,
  day         TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  content     TEXT NOT NULL,
  data        TEXT NOT NULL,
  pattern_key TEXT,
  outcome     REAL,
  status      TEXT NOT NULL DEFAULT 'active',
  importance  REAL NOT NULL DEFAULT 0.5,
  embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_mem_ns ON memories(market, store, status);
CREATE INDEX IF NOT EXISTS idx_mem_pattern ON memories(market, pattern_key);
"""


def _pack(vec: list[float] | None) -> bytes | None:
    return struct.pack(f"{len(vec)}f", *vec) if vec else None


def _unpack(blob: bytes | None) -> list[float] | None:
    return list(struct.unpack(f"{len(blob) // 4}f", blob)) if blob else None


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    market: str
    store: str
    day: date
    content: str
    data: dict
    pattern_key: str | None
    outcome: float | None
    status: str
    importance: float
    embedding: list[float] | None


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def add(
        self,
        market: str,
        store: str,
        day: date,
        content: str,
        data: dict,
        pattern_key: str | None = None,
        outcome: float | None = None,
        status: str = "active",
        importance: float = 0.5,
        embedding: list[float] | None = None,
    ) -> str:
        assert store in STORES and status in STATUSES
        entry_id = f"mem_{store[:4]}_{uuid.uuid4().hex[:10]}"
        self._db.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                entry_id,
                market,
                store,
                day.isoformat(),
                datetime.now(timezone.utc).isoformat(),
                content,
                json.dumps(data, ensure_ascii=False, default=str),
                pattern_key,
                outcome,
                status,
                importance,
                _pack(embedding),
            ),
        )
        self._db.commit()
        return entry_id

    def update(self, entry_id: str, **fields) -> None:
        allowed = {"outcome", "status", "importance", "content"}
        assert set(fields) <= allowed, f"수정 불가 필드: {set(fields) - allowed}"
        sets = ", ".join(f"{k}=?" for k in fields)
        self._db.execute(
            f"UPDATE memories SET {sets} WHERE id=?", [*fields.values(), entry_id]
        )
        self._db.commit()

    def get(self, entry_id: str) -> MemoryEntry | None:
        row = self._db.execute("SELECT * FROM memories WHERE id=?", (entry_id,)).fetchone()
        return self._to_entry(row) if row else None

    def query(
        self,
        market: str,
        store: str | None = None,
        status: str | None = None,
        pattern_key: str | None = None,
        day: date | None = None,
        outcome_missing: bool = False,
    ) -> list[MemoryEntry]:
        """네임스페이스(market) 필수 — 시장 간 교차 읽기는 API 레벨에서 불가."""
        sql, params = "SELECT * FROM memories WHERE market=?", [market]
        if store:
            sql += " AND store=?"
            params.append(store)
        if status:
            sql += " AND status=?"
            params.append(status)
        if pattern_key:
            sql += " AND pattern_key=?"
            params.append(pattern_key)
        if day:
            sql += " AND day=?"
            params.append(day.isoformat())
        if outcome_missing:
            sql += " AND outcome IS NULL"
        sql += " ORDER BY day ASC, created_at ASC"
        return [self._to_entry(r) for r in self._db.execute(sql, params)]

    @staticmethod
    def _to_entry(row: tuple) -> MemoryEntry:
        return MemoryEntry(
            id=row[0],
            market=row[1],
            store=row[2],
            day=date.fromisoformat(row[3]),
            content=row[5],
            data=json.loads(row[6]),
            pattern_key=row[7],
            outcome=row[8],
            status=row[9],
            importance=row[10],
            embedding=_unpack(row[11]),
        )
