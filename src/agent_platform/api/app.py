from __future__ import annotations

import json
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.base import ToolStreamChunk
from agent_platform.api.auth import build_api_key_auth
from agent_platform.api.conversations import (
    ConversationNotFound,
    ConversationStore,
)
from agent_platform.api.errors import ErrorResponse, describe_error
from agent_platform.api.schemas import ChatRequest, ChatResponse
from agent_platform.config import (
    Settings,
    build_agent_async,
    build_tracer,
    get_settings,
    setup_logging,
)
from agent_platform.core.errors import ConfigError, PlatformError
from agent_platform.core.observability import set_tracer

logger = logging.getLogger(__name__)


def _log_failure(exc: Exception, failure: ErrorResponse) -> None:
    """Log a failed request once: a traceback for a server error, one line otherwise."""
    if failure.status >= 500:
        logger.error("Request failed: %s", failure.error, exc_info=exc)
    else:
        logger.warning("Request failed: %s", failure.error)


def _error_response(exc: Exception) -> JSONResponse:
    failure = describe_error(exc)
    _log_failure(exc, failure)
    headers: dict[str, str] = {}
    if (
        failure.status == 429
        and isinstance(exc, PlatformError)
        and exc.retry_after is not None
    ):
        headers["Retry-After"] = str(math.ceil(exc.retry_after))
    return JSONResponse(
        {"error": failure.error, "message": failure.message},
        status_code=failure.status,
        headers=headers,
    )


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

    @app.exception_handler(PlatformError)
    async def handle_platform_error(request: Request, exc: Exception) -> JSONResponse:
        return _error_response(exc)

    # `ServerErrorMiddleware` re-raises after answering, which would log a
    # second traceback, so an unexpected exception is answered here instead.
    @app.middleware("http")
    async def answer_unexpected_errors(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            return _error_response(exc)

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
            # The status line is already sent, so a failure can only end the
            # stream with an error event. A client disconnect raises
            # `asyncio.CancelledError`, a `BaseException`, which passes through.
            try:
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
            except Exception as exc:
                failure = describe_error(exc)
                _log_failure(exc, failure)
                error_payload = {
                    "type": "error",
                    "error": failure.error,
                    "message": failure.message,
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
                return

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
