"""토론 결론 → 플레이북 제안초안. **적용 경로 없음** — 파일로 쓰기만 한다.

대상은 플레이북 한 파일이다. 리스크 한도는 LLM 이 개입하지 않는 결정론 가드 층이라,
사람 승인이 사이에 있더라도 "완화안을 기계가 먼저 제시한다"는 구조 자체를 만들지
않는다. 유니버스·집행 설정은 운영 스크립트 안에 있어 초안 대상이 파이썬 소스가 되고,
유니버스 변경은 성과 곡선의 해석을 앞뒤로 갈라놓는다.

초안은 통합 diff 로 낸다 — 산문 제안서는 사람이 손으로 옮겨 적으며 문안이 변형되고,
나중에 "무엇이 승인된 것인지"를 복원할 수 없다. 적용은 여전히 사람이 파일을 고칠 때만
일어나므로 승인 게이트는 그대로다.

헤더에 인용 ID 와 `grounding` 을 싣는다. 사용자가 대화에서 관철시킨 견해가 그대로
초안이 되면 검증 안 된 주장이 승인 도장을 달고 들어오는데, 근거 유무를 숨기지 않으면
사람이 그것을 보고 판단할 수 있다. 근거가 없으면 `grounding: none` 으로 명시한다.

`applies` 는 diff 의 문맥·삭제 줄이 현재 원문과 맞는지 검사한 결과다 — 붙지 않는
초안을 조용히 넘기지 않기 위한 표시이며, 검사는 적용이 아니다(파일을 건드리지 않는다).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

#: 초안이 건드릴 수 있는 유일한 파일 (레포 루트 기준 상대 경로)
TARGET = "trader/playbook.md"
PROPOSAL_DIR = "data/proposals"

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


@dataclass(frozen=True)
class Proposal:
    path: Path
    diff: str
    applies: bool
    reason: str | None  # applies=False 일 때 어긋난 지점


def _hunks(diff: str) -> list[tuple[int, list[str]]]:
    """diff → [(선언된 시작줄, 원문쪽 블록)]. 원문쪽 = 문맥 줄 + 삭제 줄."""
    out: list[tuple[int, list[str]]] = []
    for raw in diff.splitlines():
        m = _HUNK.match(raw)
        if m:
            out.append((int(m.group(1)), []))
            continue
        if not out or raw.startswith(("+++", "---", "+", "\\")):
            continue  # 헤더·추가 줄·"\ No newline" 은 원문과 대조할 것이 없다
        if raw.startswith(("-", " ")) or raw == "":
            out[-1][1].append(raw[1:] if raw else "")
    return out


def diff_applies(diff: str, original: str) -> tuple[bool, str | None]:
    """통합 diff 가 원문에 붙는지 검사. **적용하지 않는다.**

    판정 기준은 `git apply` 와 같게 둔다 — 문맥 블록이 원문 어딘가에 그대로 있으면
    붙는다. @@ 헤더의 줄번호가 어긋나는 것은 실패가 아니다(offset 으로 흡수된다).
    반대로 문맥 한 글자가 틀리면 붙지 않으므로, 사람이 손으로 적용하다 발견하는
    것보다 초안에 미리 표시한다.
    """
    lines = original.splitlines()
    # 빈 diff 는 실패가 아니라 설계된 결과다 — 토론이 변경으로 이어지지 않으면 억지로
    # 만들지 않는다. 형식이 깨진 초안과 같은 사유로 묶으면 둘을 구분할 수 없다.
    if not diff.strip():
        return False, "변경 없음 — 토론이 플레이북 수정으로 이어지지 않았다"
    hunks = _hunks(diff)
    if not hunks:
        return False, "hunk 없음 (@@ 헤더가 없다)"
    cursor = 0  # 앞 hunk 가 끝난 지점 — git 처럼 순서를 지킨다
    for start, block in hunks:
        if not block:
            continue  # 문맥 없는 순수 추가 — 대조할 것이 없다
        found = -1
        for i in range(cursor, len(lines) - len(block) + 1):
            if lines[i : i + len(block)] == block:
                found = i
                break
        if found < 0:
            at = min(max(start - 1, 0), max(len(lines) - 1, 0))
            got = lines[at] if lines else ""
            return False, (
                f"@@ -{start} 의 문맥을 원문에서 찾지 못함 — "
                f"기대 {block[0]!r}, line={at + 1} 원문 {got!r}"
            )
        cursor = found + len(block)
    return True, None


def proposal_path(root: Path, session_id: str, day: date | None = None) -> Path:
    """세션발 초안 경로. 월간 자동 제안서(`{YYYY-MM}.md`)와 파일명으로 갈린다 —
    같은 평면에 두면 몇 달 뒤 어느 쪽이 사람 대화에서 나온 것인지 가릴 수 없다."""
    day = day or datetime.now(timezone.utc).date()
    return root / PROPOSAL_DIR / f"session-{day.isoformat()}-{session_id}.md"


def render(
    diff: str,
    *,
    session_id: str,
    market: str | None,
    cited_ids: list[str],
    applies: bool,
    reason: str | None,
    now: datetime | None = None,
) -> str:
    """초안 파일 본문 — frontmatter(감사용 안정 키) + 통합 diff."""
    now = now or datetime.now(timezone.utc)
    head = [
        "---",
        "source: session",  # 월간 자동 제안서와 구분되는 출처 표시
        f"session_id: {session_id}",
        f"market: {market or ''}",
        f"created: {now.isoformat()}",
        f"target: {TARGET}",
        f"cited_ids: [{', '.join(cited_ids)}]",
        f"grounding: {'cited' if cited_ids else 'none'}",
        f"applies: {'true' if applies else 'false'}",
    ]
    if reason:
        head.append(f"applies_reason: {reason}")
    head += [
        "applied: false",  # 적용 경로 없음 — 사람이 파일을 고친 뒤 직접 바꾼다
        "---",
        "",
    ]
    return "\n".join(head) + f"```diff\n{diff.strip()}\n```\n"
