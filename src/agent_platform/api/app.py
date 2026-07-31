from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.base import ToolStreamChunk
from agent_platform.api.schemas import ChatRequest, ChatResponse
from agent_platform.config import Settings, build_agent, get_settings, setup_logging
from agent_platform.core.tracing import configure_tracing


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
        return ChatResponse(
            response=response,
            usage=agent.token_usage,
            estimated_cost=agent.estimated_cost,
        )

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        """Single-turn SSE stream of the agent's response.

        Unlike `/chat`, this does not read from or append to the
        `ConversationAgent`'s persistent history: each call is a fresh,
        stateless turn built directly on `AgentExecutor.run_streaming()`.
        """
        agent: ConversationAgent = app.state.agent
        executor = AgentExecutor(agent, max_iterations=settings.max_iterations)
        conversation_id = getattr(agent, "conversation_id", None)

        async def events() -> AsyncIterator[str]:
            async for event in executor.run_streaming(
                request.message,
                conversation_id=str(conversation_id) if conversation_id else None,
            ):
                if isinstance(event, ToolStreamChunk):
                    payload = {
                        "type": "tool",
                        "tool_call_id": event.tool_call_id,
                        "delta": event.delta,
                        "is_final": event.is_final,
                        "is_error": event.is_error,
                    }
                else:
                    payload = {"type": "text", "delta": event}
                yield f"data: {json.dumps(payload)}\n\n"

            usage_payload = {
                "type": "usage",
                **agent.token_usage.model_dump(),
                "estimated_cost": agent.estimated_cost,
            }
            yield f"data: {json.dumps(usage_payload)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
