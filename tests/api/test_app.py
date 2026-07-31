import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pydantic import BaseModel

from agent_platform.agents.agent import Agent
from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.api.app import create_app, main
from tests.helpers import (
    make_fake_stream,
    make_text_stream_chunks,
    make_tool_call_stream_chunks,
)


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes back"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return "echoed"


class FakeAgent:
    async def chat(self, user_input: str) -> str:
        return f"echo: {user_input}"


class TestHealth:
    def test_returns_ok(self):
        with TestClient(create_app()) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChat:
    def test_returns_agent_response(self):
        app = create_app()
        with TestClient(app) as client:
            app.state.agent = FakeAgent()
            response = client.post("/chat", json={"message": "hi"})

        assert response.status_code == 200
        assert response.json() == {"response": "echo: hi"}

    def test_requires_message_field(self):
        with TestClient(create_app()) as client:
            response = client.post("/chat", json={})

        assert response.status_code == 422


class TestChatStream:
    def test_streams_text_events(self):
        mock_llm = MagicMock()
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["Hel", "lo"])
        )
        agent = Agent(name="streamer", llm=mock_llm)

        app = create_app()
        with TestClient(app) as client:
            app.state.agent = agent
            response = client.post("/chat/stream", json={"message": "hi"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [e["delta"] for e in events if e["type"] == "text"] == ["Hel", "lo"]

    def test_streams_tool_events(self):
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

        mock_llm = MagicMock()
        mock_llm.stream.side_effect = stream_side_effect
        registry = ToolRegistry()
        registry.register(_EchoTool())
        agent = Agent(name="tool-streamer", llm=mock_llm, tool_registry=registry)

        app = create_app()
        with TestClient(app) as client:
            app.state.agent = agent
            response = client.post("/chat/stream", json={"message": "hi"})

        events = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        tool_events = [e for e in events if e["type"] == "tool"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool_call_id"] == "c1"
        assert tool_events[0]["is_error"] is False

    def test_requires_message_field(self):
        with TestClient(create_app()) as client:
            response = client.post("/chat/stream", json={})

        assert response.status_code == 422


class TestMain:
    def test_runs_uvicorn_with_settings_host_and_port(self, mocker):
        mock_uvicorn = MagicMock()
        mocker.patch.dict("sys.modules", {"uvicorn": mock_uvicorn})
        mock_get_settings = mocker.patch("agent_platform.api.app.get_settings")
        mock_get_settings.return_value = MagicMock(api_host="0.0.0.0", api_port=9000)
        main()

        mock_uvicorn.run.assert_called_once()
        _, kwargs = mock_uvicorn.run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000
