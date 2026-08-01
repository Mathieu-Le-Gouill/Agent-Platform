from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from agent_platform.core.errors import ProviderError  # noqa: E402
from agent_platform.integrations.mcp.stdio.provider import StdioMCPClient  # noqa: E402


class _FakeSession:
    def __init__(self) -> None:
        self.initialized = False
        self.list_tools_result = SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="echo",
                    description="Echoes input",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                )
            ]
        )
        self.call_tool_result = SimpleNamespace(
            content=[SimpleNamespace(text="hello")], is_error=False
        )

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self):
        return self.list_tools_result

    async def call_tool(self, name, arguments):
        return self.call_tool_result


@pytest.fixture
def fake_session():
    return _FakeSession()


@pytest.fixture
def client(monkeypatch, fake_session):
    @asynccontextmanager
    async def fake_stdio_client(server_params):
        yield ("read", "write")

    @asynccontextmanager
    async def fake_client_session_ctx(read, write):
        yield fake_session

    def fake_client_session(read, write):
        return fake_client_session_ctx(read, write)

    monkeypatch.setattr(
        "agent_platform.integrations.mcp.stdio.provider.stdio_client",
        fake_stdio_client,
    )
    monkeypatch.setattr(
        "agent_platform.integrations.mcp.stdio.provider.ClientSession",
        fake_client_session,
    )
    return StdioMCPClient(command="fake-server")


class TestStdioMCPClient:
    @pytest.mark.asyncio
    async def test_connect_initializes_session(self, client, fake_session):
        await client.connect()
        assert fake_session.initialized is True
        await client.aclose()

    @pytest.mark.asyncio
    async def test_call_tool_without_connect_raises(self, client):
        with pytest.raises(ProviderError, match="not connected"):
            await client.call_tool("echo", {"text": "hi"})

    @pytest.mark.asyncio
    async def test_list_tools_maps_spec(self, client):
        async with client:
            specs = await client.list_tools()
        assert len(specs) == 1
        assert specs[0].name == "echo"
        assert specs[0].description == "Echoes input"
        assert specs[0].input_schema["required"] == ["text"]

    @pytest.mark.asyncio
    async def test_call_tool_returns_text(self, client):
        async with client:
            result = await client.call_tool("echo", {"text": "hi"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_call_tool_error_raises_provider_error(self, client, fake_session):
        fake_session.call_tool_result = SimpleNamespace(
            content=[SimpleNamespace(text="boom")], is_error=True
        )
        async with client:
            with pytest.raises(ProviderError, match="boom"):
                await client.call_tool("echo", {"text": "hi"})

    @pytest.mark.asyncio
    async def test_aclose_without_connect_is_noop(self, client):
        await client.aclose()
