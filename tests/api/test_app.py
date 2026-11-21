import asyncio
import contextlib
import json
import logging
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.errors import AgentThinkError
from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.api.app import create_app, main
from agent_platform.config.settings import Settings
from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.llm.response import StreamChunk
from agent_platform.core.interfaces.mcp.base import BaseMCPClient
from agent_platform.core.schemas.mcp import MCPToolSpec
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.token_usage import TokenUsageAggregator
from tests.helpers import (
    make_fake_llm_response,
    make_fake_stream,
    make_text_stream_chunks,
    make_tool_call_stream_chunks,
)

KEY_A = "key-a-secret"
KEY_B = "key-b-secret"


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes back"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return "echoed"


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _recording_llm() -> MagicMock:
    """LLM whose `agenerate` answers "reply" and records each prompt's contents."""
    llm = MagicMock()
    llm.sent = []

    async def agenerate(*, prompt, config, tools):
        llm.sent.append([m.content for m in prompt.messages])
        return make_fake_llm_response(content="reply")

    llm.agenerate = agenerate
    llm.stream.side_effect = lambda **kw: make_fake_stream(
        [
            StreamChunk(delta="hi"),
            StreamChunk(delta="", usage=TokenUsage(input_tokens=4, output_tokens=2)),
        ]
    )
    return llm


def _make_app(mocker, llm, *, registry=None, **settings_kwargs) -> FastAPI:
    """Build the app with the agent factory replaced by a `ConversationAgent` over `llm`."""
    template = ConversationAgent(
        name="test",
        llm=llm,
        tool_registry=registry,
        token_usage_aggregator=TokenUsageAggregator(),
    )
    mocker.patch(
        "agent_platform.api.app.build_agent_async",
        return_value=(template, []),
    )
    settings_kwargs.setdefault("api_keys", [KEY_A, KEY_B])
    return create_app(Settings(_env_file=None, **settings_kwargs))


def _sse_events(response) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


class TestHealth:
    def test_returns_ok_without_a_key(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAuth:
    def test_chat_without_header_is_401(self, mocker):
        llm = _recording_llm()
        with TestClient(_make_app(mocker, llm)) as client:
            response = client.post("/chat", json={"message": "hi"})

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert llm.sent == []

    def test_stream_without_header_is_401(self, mocker):
        llm = _recording_llm()
        with TestClient(_make_app(mocker, llm)) as client:
            response = client.post("/chat/stream", json={"message": "hi"})

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        llm.stream.assert_not_called()

    def test_wrong_key_is_401(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            chat = client.post("/chat", json={"message": "hi"}, headers=_auth("nope"))
            stream = client.post(
                "/chat/stream", json={"message": "hi"}, headers=_auth("nope")
            )

        assert chat.status_code == 401
        assert stream.status_code == 401

    def test_non_ascii_key_is_401(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            chat = client.post(
                "/chat",
                json={"message": "hi"},
                headers={"Authorization": "Bearer ké".encode("latin-1")},
            )
            stream = client.post(
                "/chat/stream",
                json={"message": "hi"},
                headers={"Authorization": "Bearer ké".encode("latin-1")},
            )

        assert chat.status_code == 401
        assert stream.status_code == 401

    def test_docs_routes_are_off_with_auth_enabled(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            statuses = [
                client.get(path).status_code
                for path in ("/docs", "/redoc", "/openapi.json")
            ]

        assert statuses == [404, 404, 404]

    def test_disabled_auth_serves_without_header_and_keeps_docs(self, mocker):
        app = _make_app(mocker, _recording_llm(), api_keys=[], api_auth_disabled=True)
        with TestClient(app) as client:
            chat = client.post("/chat", json={"message": "hi"})
            docs = client.get("/docs")

        assert chat.status_code == 200
        assert docs.status_code == 200


class TestChat:
    def test_returns_response_and_conversation_id(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post(
                "/chat", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        body = response.json()
        assert response.status_code == 200
        assert body["response"] == "reply"
        assert body["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "reasoning_tokens": 0,
        }
        assert body["estimated_cost"] is None
        UUID(body["conversation_id"])

    def test_requires_message_field(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post("/chat", json={}, headers=_auth(KEY_A))

        assert response.status_code == 422

    def test_same_key_continues_a_conversation(self, mocker):
        llm = _recording_llm()
        with TestClient(_make_app(mocker, llm)) as client:
            first = client.post(
                "/chat", json={"message": "one"}, headers=_auth(KEY_A)
            ).json()
            second = client.post(
                "/chat",
                json={"message": "two", "conversation_id": first["conversation_id"]},
                headers=_auth(KEY_A),
            ).json()

        assert second["conversation_id"] == first["conversation_id"]
        assert llm.sent[1] == ["one", "reply", "two"]
        assert second["usage"]["input_tokens"] == 20

    def test_unknown_conversation_id_is_404(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post(
                "/chat",
                json={"message": "hi", "conversation_id": str(uuid4())},
                headers=_auth(KEY_A),
            )

        assert response.status_code == 404
        assert response.json() == {"detail": "conversation not found"}

    def test_another_keys_conversation_id_is_404_and_skips_the_llm(self, mocker):
        llm = _recording_llm()
        with TestClient(_make_app(mocker, llm)) as client:
            created = client.post(
                "/chat", json={"message": "secret"}, headers=_auth(KEY_A)
            ).json()
            sent_before = len(llm.sent)
            foreign = client.post(
                "/chat",
                json={"message": "peek", "conversation_id": created["conversation_id"]},
                headers=_auth(KEY_B),
            )
            unknown = client.post(
                "/chat",
                json={"message": "peek", "conversation_id": str(uuid4())},
                headers=_auth(KEY_B),
            )

        assert foreign.status_code == 404
        assert foreign.json() == unknown.json()
        assert len(llm.sent) == sent_before

    def test_two_keys_keep_separate_histories(self, mocker):
        llm = _recording_llm()
        with TestClient(_make_app(mocker, llm)) as client:
            a = client.post(
                "/chat", json={"message": "from a"}, headers=_auth(KEY_A)
            ).json()
            b = client.post(
                "/chat", json={"message": "from b"}, headers=_auth(KEY_B)
            ).json()

        assert llm.sent[1] == ["from b"]
        assert a["conversation_id"] != b["conversation_id"]
        assert b["usage"]["input_tokens"] == 10

    def test_two_conversations_of_one_key_keep_separate_usage(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            first = client.post(
                "/chat", json={"message": "one"}, headers=_auth(KEY_A)
            ).json()
            second = client.post(
                "/chat", json={"message": "two"}, headers=_auth(KEY_A)
            ).json()

        assert first["usage"]["input_tokens"] == 10
        assert second["usage"]["input_tokens"] == 10


class TestChatStream:
    def test_streams_text_events(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post(
                "/chat/stream", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(response)
        assert [e["delta"] for e in events if e["type"] == "text"] == ["hi"]

    def test_usage_event_covers_the_current_turn_only(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            first = client.post(
                "/chat/stream", json={"message": "one"}, headers=_auth(KEY_A)
            )
            second = client.post(
                "/chat/stream", json={"message": "two"}, headers=_auth(KEY_A)
            )

        for response in (first, second):
            events = _sse_events(response)
            assert events[-1]["type"] == "usage"
            assert events[-1]["input_tokens"] == 4

    def test_ignores_conversation_id_and_keeps_no_history(self, mocker):
        llm = _recording_llm()
        with TestClient(_make_app(mocker, llm)) as client:
            created = client.post(
                "/chat", json={"message": "secret"}, headers=_auth(KEY_A)
            ).json()
            response = client.post(
                "/chat/stream",
                json={"message": "hi", "conversation_id": created["conversation_id"]},
                headers=_auth(KEY_B),
            )
            history = client.app.state.agent.history

        assert response.status_code == 200
        sent = llm.stream.call_args.kwargs["prompt"].messages
        assert [m.content for m in sent] == ["hi"]
        assert history == []

    def test_streams_tool_events(self, mocker):
        call_count = 0

        def stream_side_effect(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_stream(
                    make_tool_call_stream_chunks(
                        [{"id": "c1", "name": "echo", "args": {}}]
                    )
                )
            return make_fake_stream(make_text_stream_chunks(["done"]))

        llm = MagicMock()
        llm.stream.side_effect = stream_side_effect
        registry = ToolRegistry()
        registry.register(_EchoTool())

        with TestClient(_make_app(mocker, llm, registry=registry)) as client:
            response = client.post(
                "/chat/stream", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        tool_events = [e for e in _sse_events(response) if e["type"] == "tool"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool_call_id"] == "c1"
        assert tool_events[0]["is_error"] is False

    def test_requires_message_field(self, mocker):
        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post("/chat/stream", json={}, headers=_auth(KEY_A))

        assert response.status_code == 422


_ERROR_STATUSES = [
    ("llm_rate_limited", 429),
    ("llm_timeout", 504),
    ("llm_context_window", 422),
    ("provider_unavailable", 503),
    ("provider_bad_request", 502),
    ("agent_recovery_exhausted", 502),
    ("agent_max_iterations", 500),
]
_VENDOR_TEXT = "vendor echoed the prompt: s3cret-prompt"


def _records_with_traceback(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.exc_info]


def _failing_chat(mocker, error: Exception) -> None:
    mocker.patch.object(ConversationAgent, "chat", side_effect=error)


class TestChatErrors:
    @pytest.mark.parametrize(("code", "status"), _ERROR_STATUSES)
    def test_each_code_maps_to_its_status_and_body(self, mocker, code, status):
        _failing_chat(mocker, AgentThinkError(_VENDOR_TEXT, code=code))

        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post(
                "/chat", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        assert response.status_code == status
        body = response.json()
        assert set(body) == {"error", "message"}
        assert body["error"] == code
        assert body["message"]
        assert _VENDOR_TEXT not in response.text

    def test_a_rate_limit_carries_the_retry_after_header(self, mocker):
        _failing_chat(
            mocker,
            AgentThinkError(
                _VENDOR_TEXT, code="llm_rate_limited", retryable=True, retry_after=6.2
            ),
        )

        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post(
                "/chat", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        assert response.status_code == 429
        assert response.headers["retry-after"] == "7"

    def test_a_rate_limit_without_a_delay_has_no_retry_after_header(self, mocker):
        _failing_chat(mocker, AgentThinkError("x", code="llm_rate_limited"))

        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post(
                "/chat", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        assert response.status_code == 429
        assert "retry-after" not in response.headers

    def test_a_platform_error_without_a_known_code_is_an_internal_error(self, mocker):
        _failing_chat(mocker, MissingCredentialError(_VENDOR_TEXT))

        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post(
                "/chat", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        assert response.status_code == 500
        assert response.json()["error"] == "internal_error"
        assert _VENDOR_TEXT not in response.text

    def test_an_unknown_exception_is_an_internal_error_without_its_text(self, mocker):
        _failing_chat(mocker, RuntimeError(_VENDOR_TEXT))

        with TestClient(_make_app(mocker, _recording_llm())) as client:
            response = client.post(
                "/chat", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        assert response.status_code == 500
        assert response.json()["error"] == "internal_error"
        assert _VENDOR_TEXT not in response.text

    def test_a_provider_5xx_logs_one_traceback(self, mocker, caplog):
        _failing_chat(mocker, AgentThinkError("x", code="provider_unavailable"))

        with caplog.at_level(logging.DEBUG):
            with TestClient(_make_app(mocker, _recording_llm())) as client:
                response = client.post(
                    "/chat", json={"message": "hi"}, headers=_auth(KEY_A)
                )

        assert response.status_code == 503
        assert len(_records_with_traceback(caplog)) == 1

    def test_an_unknown_exception_logs_one_traceback(self, mocker, caplog):
        _failing_chat(mocker, RuntimeError("boom"))

        with caplog.at_level(logging.DEBUG):
            with TestClient(_make_app(mocker, _recording_llm())) as client:
                client.post("/chat", json={"message": "hi"}, headers=_auth(KEY_A))

        assert len(_records_with_traceback(caplog)) == 1

    def test_a_client_error_status_logs_a_warning_without_a_traceback(
        self, mocker, caplog
    ):
        _failing_chat(mocker, AgentThinkError("x", code="llm_rate_limited"))

        with caplog.at_level(logging.DEBUG):
            with TestClient(_make_app(mocker, _recording_llm())) as client:
                client.post("/chat", json={"message": "hi"}, headers=_auth(KEY_A))

        assert _records_with_traceback(caplog) == []
        assert [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_a_real_provider_error_travels_from_the_llm_to_the_status(self, mocker):
        llm = _recording_llm()

        async def agenerate(**kwargs):
            raise ProviderError(
                _VENDOR_TEXT, code="llm_timeout", retryable=True, retry_after=None
            )

        llm.agenerate = agenerate

        with TestClient(_make_app(mocker, llm)) as client:
            response = client.post(
                "/chat", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        assert response.status_code == 504
        assert response.json()["error"] == "llm_timeout"


class TestChatStreamErrors:
    @staticmethod
    def _llm_failing_on_the_second_call(error: BaseException) -> MagicMock:
        calls = 0

        async def failing():
            raise error
            yield  # pragma: no cover

        def stream(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return make_fake_stream(
                    [
                        *make_text_stream_chunks(["checking "])[:-1],
                        *make_tool_call_stream_chunks(
                            [{"id": "c1", "name": "echo", "args": {}}]
                        ),
                    ]
                )
            return failing()

        llm = MagicMock()
        llm.stream.side_effect = stream
        return llm

    @staticmethod
    def _registry() -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(_EchoTool())
        return registry

    def test_a_mid_stream_failure_ends_with_one_error_event_and_no_usage(self, mocker):
        llm = self._llm_failing_on_the_second_call(
            ProviderError(_VENDOR_TEXT, code="llm_timeout", retryable=True)
        )

        with TestClient(_make_app(mocker, llm, registry=self._registry())) as client:
            response = client.post(
                "/chat/stream", json={"message": "hi"}, headers=_auth(KEY_A)
            )

        assert response.status_code == 200
        events = _sse_events(response)
        assert [e["delta"] for e in events if e["type"] == "text"] == ["checking "]
        assert [e["type"] for e in events].count("error") == 1
        assert events[-1]["type"] == "error"
        assert events[-1]["error"] == "llm_timeout"
        assert events[-1]["message"]
        assert "usage" not in [e["type"] for e in events]
        assert _VENDOR_TEXT not in response.text

    def test_an_unknown_mid_stream_exception_is_an_internal_error_event(
        self, mocker, caplog
    ):
        llm = self._llm_failing_on_the_second_call(RuntimeError(_VENDOR_TEXT))

        with caplog.at_level(logging.DEBUG):
            with TestClient(
                _make_app(mocker, llm, registry=self._registry())
            ) as client:
                response = client.post(
                    "/chat/stream", json={"message": "hi"}, headers=_auth(KEY_A)
                )

        assert _sse_events(response)[-1]["error"] == "internal_error"
        assert _VENDOR_TEXT not in response.text
        assert len(_records_with_traceback(caplog)) == 1

    def test_a_cancelled_stream_sends_no_error_event(self, mocker):
        llm = self._llm_failing_on_the_second_call(asyncio.CancelledError())
        sent: list[dict] = []

        with TestClient(_make_app(mocker, llm, registry=self._registry())) as client:
            with contextlib.suppress(BaseException):
                with client.stream(
                    "POST",
                    "/chat/stream",
                    json={"message": "hi"},
                    headers=_auth(KEY_A),
                ) as response:
                    sent.extend(
                        json.loads(line[len("data: ") :])
                        for line in response.iter_lines()
                        if line.startswith("data: ")
                    )

        assert "error" not in [e["type"] for e in sent]


class _FakeMCPClient(BaseMCPClient):
    def __init__(self, tools: list[MCPToolSpec]) -> None:
        self._tools = tools
        self.closed = False

    async def connect(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True

    async def list_tools(self, config: Any = None) -> list[MCPToolSpec]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict, config: Any = None) -> Any:
        return "ok"


class TestLifespan:
    def test_builds_conversation_agent_via_lifespan(self):
        settings = Settings(_env_file=None, api_keys=[KEY_A])
        with TestClient(create_app(settings)) as client:
            assert isinstance(client.app.state.agent, ConversationAgent)

    def test_closes_mcp_clients_on_shutdown(self, mocker):
        fake_client = _FakeMCPClient(
            [MCPToolSpec(name="search_docs", description="", input_schema={})]
        )
        mocker.patch(
            "agent_platform.config.container._build_mcp_client",
            return_value=fake_client,
        )
        settings = Settings(
            _env_file=None,
            api_keys=[KEY_A],
            mcp_stdio_servers={"docs": ["npx", "-y", "server-docs"]},
        )

        with TestClient(create_app(settings)) as client:
            assert "search_docs" in client.app.state.agent.tool_registry.all()
            assert fake_client.closed is False

        assert fake_client.closed is True


class TestMain:
    def test_runs_uvicorn_with_settings_host_and_port(self, mocker):
        mock_uvicorn = MagicMock()
        mocker.patch.dict("sys.modules", {"uvicorn": mock_uvicorn})
        mocker.patch(
            "agent_platform.api.app.get_settings",
            return_value=Settings(
                _env_file=None, api_keys=[KEY_A], api_host="0.0.0.0", api_port=9000
            ),
        )
        main()

        mock_uvicorn.run.assert_called_once()
        args, kwargs = mock_uvicorn.run.call_args
        assert isinstance(args[0], FastAPI)
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000
