"""MCP 서버 (stdio) — Claude Code/Desktop 에서 실행 중 에이전트와 대화.

등록 (Claude Code):
    claude mcp add trading-agent -- uv run --directory /Users/cholhwan/Documents/ai/trading-agent \
        python scripts/run_mcp_server.py

도구: ask_trader(질문/토론) · conclude_discussion(결론 기록) · get_briefing(일일 브리핑)
모든 답변은 grounding — 인용 ID 가 함께 반환된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from interaction.api import load_env  # noqa: E402
from interaction.briefing import build_briefing  # noqa: E402
from interaction.chat import ChatEngine, GroundingError  # noqa: E402
from llm import LLMRouter  # noqa: E402

mcp = FastMCP("trading-agent")
_engine = ChatEngine(LLMRouter(load_env(ROOT / ".env")), ROOT)


@mcp.tool()
async def ask_trader(question: str, session_id: str = "", market: str = "") -> str:
    """실행 중인 트레이딩 에이전트에게 질문/반론. 답변은 포지션·메모리·성과에 grounded.
    토론을 이어가려면 반환된 session_id 를 다시 넘겨라."""
    try:
        answer, sid = await _engine.ask(question, session_id or None, market or None)
    except GroundingError as e:
        return f"[grounding 실패] {e}"
    return f"{answer.answer}\n\n[인용: {', '.join(answer.cited_ids)}]\n[session_id: {sid}]"


@mcp.tool()
async def conclude_discussion(session_id: str) -> str:
    """토론 결론을 episodic 메모리에 user_session 태그로 기록 (승격은 admission 게이트 필요)."""
    try:
        entry_id = await _engine.conclude(session_id)
    except KeyError as e:
        return f"[오류] {e}"
    return f"기록 완료: {entry_id} (semantic 승격은 검증 게이트를 통과해야 한다)"


@mcp.tool()
def get_briefing() -> str:
    """오늘의 일일 브리핑 (결정 + 근거 + 성과 + 승격 교훈)."""
    return build_briefing(ROOT)


if __name__ == "__main__":
    mcp.run()
