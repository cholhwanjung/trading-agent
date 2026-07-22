"""Chat 엔진 — grounded 답변(인용 강제) + 전략 토론 모드.

- 답변의 모든 주장은 context 항목 인용으로 뒷받침되어야 한다. cited_ids 가 비었거나
  context 밖 id 면 GroundingError — 게이트웨이는 이를 5xx 로 노출한다(조용한 환각 금지).
- 토론 결론은 episodic 에 "user_session" 태그로 기록 — **semantic 승격은 동일
  admission 게이트를 타야 한다**(사용자 설득이 검증을 대체하지 않음). outcome 없는
  user_session 엔트리는 통계 게이트에 들어갈 수 없어 구조적으로 승격 불가.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from interaction.context import allowed_ids, build_context
from memory import MemoryStore

SYSTEM_PROMPT = """\
너는 실행 중인 트레이딩 에이전트 본인이다. 아래 context 항목들이 너의 기억·포지션·성과의
전부다 — 이 밖의 사실은 모른다.

규칙 (위반 시 답변이 거부된다):
- 모든 주장은 context 항목에 근거해야 하며, 실제 참고한 항목의 id 를 cited_ids 에 넣는다 (최소 1개).
- context 에 근거가 없으면 지어내지 말고 "기록에 근거가 없다"고 답하되, 가장 관련된 항목을 인용한다.
- 토론에서 사용자가 반론하면 메모리·신호·성과 근거로 방어하고, 근거가 밀리면 수정을 인정한다.
  단, 토론 결론은 검증 게이트를 통과해야 정책에 반영됨을 명시한다.
- 한국어로 답한다.

JSON 만 출력:
{"answer": "<답변>", "cited_ids": ["<context id>", ...]}

context:
"""


class GroundingError(RuntimeError):
    """grounding 위반 — 인용 없음 또는 context 밖 인용."""


@dataclass
class ChatAnswer:
    answer: str
    cited_ids: list[str]


@dataclass
class DiscussionSession:
    session_id: str
    market: str | None
    messages: list[dict] = field(default_factory=list)  # {"role", "content"}


def _parse(text: str) -> ChatAnswer:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise GroundingError(f"JSON 아님: {text[:100]!r}")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise GroundingError(f"JSON 파싱 실패: {e}") from e
    return ChatAnswer(
        answer=str(data.get("answer", "")).strip(),
        cited_ids=[str(x) for x in data.get("cited_ids") or []],
    )


def enforce_grounding(answer: ChatAnswer, allowed: set[str]) -> None:
    if not answer.answer:
        raise GroundingError("빈 답변")
    if not answer.cited_ids:
        raise GroundingError("인용 없음")
    unknown = set(answer.cited_ids) - allowed
    if unknown:
        raise GroundingError(f"context 밖 인용: {sorted(unknown)}")


class ChatEngine:
    def __init__(self, router, root: Path | str) -> None:
        self.router = router
        self.root = Path(root)
        self.sessions: dict[str, DiscussionSession] = {}

    async def ask(
        self, question: str, session_id: str | None = None, market: str | None = None
    ) -> tuple[ChatAnswer, str]:
        """질문/토론 1턴. (답변, session_id) 반환. grounding 위반 시 GroundingError."""
        context = build_context(self.root)
        allowed = allowed_ids(context)
        if not allowed:
            raise GroundingError("context 비어 있음 — 일일 루프가 먼저 돌아야 한다")

        session = self.sessions.get(session_id) if session_id else None
        if session is None:
            session = DiscussionSession(session_id=uuid.uuid4().hex[:12], market=market)
            self.sessions[session.session_id] = session

        session.messages.append({"role": "user", "content": question})
        resp = await self.router.complete(
            "smart",
            system=SYSTEM_PROMPT + json.dumps(context, ensure_ascii=False, indent=1),
            messages=session.messages,
            max_tokens=4096,
            json_mode=True,
        )
        answer = _parse(resp.text)
        enforce_grounding(answer, allowed)
        session.messages.append({"role": "assistant", "content": resp.text})
        return answer, session.session_id

    async def conclude(self, session_id: str) -> str:
        """토론 결론을 episodic(user_session 태그)으로 기록. 엔트리 id 반환."""
        session = self.sessions.get(session_id)
        if session is None or not session.messages:
            raise KeyError(f"세션 없음: {session_id}")

        resp = await self.router.complete(
            "fast",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "다음 토론의 결론을 3문장 이내로 요약하라. 에이전트 입장이 수정됐다면 "
                        "무엇이 어떻게 바뀌었는지 명시하라.\n"
                        + json.dumps(session.messages, ensure_ascii=False)
                    ),
                }
            ],
            max_tokens=2048,
        )
        summary = resp.text.strip()[:600]
        market = session.market or "CRYPTO"
        store = MemoryStore(self.root / "data" / "memory.sqlite")
        try:
            entry_id = store.add(
                market,
                "episodic",
                datetime.now(timezone.utc).date(),
                content=f"[{market} 사용자 토론] {summary}",
                data={"kind": "user_session", "n_turns": len(session.messages),
                      "session_id": session_id},
                # pattern_key·outcome 없음 → admission 통계에 진입 불가 (승격은 게이트만)
            )
        finally:
            store.close()
        del self.sessions[session_id]
        return entry_id
