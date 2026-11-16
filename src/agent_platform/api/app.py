from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.base import ToolStreamChunk
from agent_platform.api.auth import build_api_key_auth
from agent_platform.api.conversations import (
    ConversationNotFound,
    ConversationStore,
)
from agent_platform.api.schemas import ChatRequest, ChatResponse
from agent_platform.config import (
    Settings,
    build_agent_async,
    build_tracer,
    get_settings,
    setup_logging,
)
from agent_platform.core.errors import ConfigError
from agent_platform.core.observability import set_tracer

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Builds the API.

    Raises:
        ConfigError: no API key is configured and authentication is not
            explicitly disabled.
    """
    settings = settings or get_settings()
    if not settings.api_keys and not settings.api_auth_disabled:
        raise ConfigError(
            "No API key configured: set AGENT_PLATFORM_API_KEYS to a JSON list of "
            "keys, or set AGENT_PLATFORM_API_AUTH_DISABLED=true to run without "
            "authentication"
        )
    setup_logging(getattr(logging, settings.log_level.upper(), logging.INFO))
    set_tracer(build_tracer(settings))
    if settings.api_auth_disabled:
        logger.warning(
            "API authentication is disabled: every caller shares one owner and "
            "one conversation cap"
        )
    authenticate = build_api_key_auth(
        settings.api_keys, disabled=settings.api_auth_disabled
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        template, mcp_clients = await build_agent_async(settings)
        app.state.agent = template
        app.state.conversations = ConversationStore(
            template, settings.api_max_conversations_per_key
        )
        try:
            yield
        finally:
            for client in mcp_clients:
                await client.aclose()

    # The interactive docs list every route to anyone who can reach the server,
    # so they stay off unless authentication is disabled.
    docs_enabled = settings.api_auth_disabled
    app = FastAPI(
        title="agent_platform API",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest, key_id: str = Depends(authenticate)
    ) -> ChatResponse:
        store: ConversationStore = app.state.conversations
        try:
            conversation = (
                store.get(key_id, request.conversation_id)
                if request.conversation_id is not None
                else store.create(key_id)
            )
        except ConversationNotFound:
            raise HTTPException(
                status_code=404, detail="conversation not found"
            ) from None
        agent = conversation.agent
        async with conversation.lock:
            response = await agent.chat(request.message)
        return ChatResponse(
            response=response,
            conversation_id=conversation.id,
            usage=agent.token_usage,
            estimated_cost=agent.estimated_cost,
        )

    @app.post("/chat/stream")
    async def chat_stream(
        request: ChatRequest, key_id: str = Depends(authenticate)
    ) -> StreamingResponse:
        """Single-turn SSE stream of the agent's response.

        Unlike `/chat`, this ignores `conversation_id` and does not read from
        or append to any conversation's history: each call is a fresh,
        stateless turn on a throwaway fork of the template agent, built
        directly on `AgentExecutor.run_streaming()`, so its usage covers this
        turn only and nothing outlives the request.
        """
        template: ConversationAgent = app.state.agent
        agent = template.with_conversation(uuid4())
        executor = AgentExecutor(agent, max_iterations=settings.max_iterations)

        async def events() -> AsyncIterator[str]:
            async for event in executor.run_streaming(
                request.message,
                conversation_id=str(agent.conversation_id),
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

            # emitted last so it reflects the final cumulative totals for the whole turn
            usage_payload = {
                "type": "usage",
                **agent.token_usage.model_dump(),
                "estimated_cost": agent.estimated_cost,
            }
            yield f"data: {json.dumps(usage_payload)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
