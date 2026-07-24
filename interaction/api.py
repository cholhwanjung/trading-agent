"""Chat Gateway — FastAPI.

실행:
    uv run uvicorn interaction.api:app --host 0.0.0.0 --port 8721

인증: .env 의 INTERACTION_API_TOKEN 설정 시 Authorization: Bearer <token> 필수.
비우면 로컬 단독 사용(인증 없음). grounding 실패는 502 — 조용한 환각 대신 명시 실패.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from harness.env import load_env
from interaction.briefing import build_briefing
from interaction.chat import ChatEngine, GroundingError

ROOT = Path(__file__).resolve().parent.parent


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    market: str | None = None


class ConcludeRequest(BaseModel):
    session_id: str


def create_app(engine: ChatEngine | None = None, token: str | None = None, root: Path = ROOT) -> FastAPI:
    if engine is None:
        from harness.usage import make_usage_sink
        from llm import LLMRouter

        env = load_env(root / ".env")
        engine = ChatEngine(LLMRouter(env, usage_sink=make_usage_sink(root)), root)
        token = env.get("INTERACTION_API_TOKEN") or None

    app = FastAPI(title="trading-agent gateway")
    app.state.engine = engine
    app.state.root = root

    async def auth(request: Request) -> None:
        if token and request.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(401, "invalid token")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/chat", dependencies=[Depends(auth)])
    async def chat(req: ChatRequest):
        try:
            answer, session_id = await engine.ask(req.question, req.session_id, req.market)
        except GroundingError as e:
            raise HTTPException(502, f"grounding 실패: {e}") from e
        return {"answer": answer.answer, "cited_ids": answer.cited_ids, "session_id": session_id}

    @app.post("/discuss/conclude", dependencies=[Depends(auth)])
    async def conclude(req: ConcludeRequest):
        try:
            entry_id = await engine.conclude(req.session_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        return {"memory_id": entry_id, "note": "user_session 태그 기록 — 승격은 admission 게이트 필요"}

    @app.get("/briefing", dependencies=[Depends(auth)])
    async def briefing():
        return {"markdown": build_briefing(app.state.root)}

    return app


app = create_app()
