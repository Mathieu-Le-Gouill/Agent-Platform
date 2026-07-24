from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.config import Settings, build_agent, get_settings, setup_logging
from agent_platform.core.tracing import configure_tracing


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(getattr(logging, settings.log_level.upper(), logging.INFO))
    configure_tracing()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.agent = build_agent(settings)
        yield

    app = FastAPI(title="agent_platform API", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        agent: ConversationAgent = app.state.agent
        response = await agent.chat(request.message)
        return ChatResponse(response=response)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
