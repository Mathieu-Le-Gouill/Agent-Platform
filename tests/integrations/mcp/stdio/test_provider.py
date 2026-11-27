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
            ],
            next_cursor=None,
        )
        self.list_tools_pages: dict[str | None, SimpleNamespace] | None = None
        self.list_tools_cursors: list[str | None] = []
        self.call_tool_result = SimpleNamespace(
            content=[SimpleNamespace(text="hello")], is_error=False
        )

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self, *, params=None):
        self.list_tools_cursors.append(params.cursor if params else None)
        if self.list_tools_pages is not None:
            return self.list_tools_pages[params.cursor if params else None]
        return self.list_tools_result

    async def call_tool(self, name, arguments):
        return self.call_tool_result


def _tools_page(names, next_cursor):
    return SimpleNamespace(
        tools=[
            SimpleNamespace(name=name, description=None, input_schema={})
            for name in names
        ],
        next_cursor=next_cursor,
    )


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
    async def test_list_tools_follows_next_cursor_to_the_last_page(
        self, client, fake_session
    ):
        fake_session.list_tools_pages = {
            None: _tools_page(["a", "b"], next_cursor="page-2"),
            "page-2": _tools_page(["c"], next_cursor="page-3"),
            "page-3": _tools_page(["d"], next_cursor=None),
        }
        async with client:
            specs = await client.list_tools()
        assert [spec.name for spec in specs] == ["a", "b", "c", "d"]
        assert fake_session.list_tools_cursors == [None, "page-2", "page-3"]

    @pytest.mark.asyncio
    async def test_list_tools_raises_when_server_repeats_a_cursor(
        self, client, fake_session
    ):
        fake_session.list_tools_pages = {
            None: _tools_page(["a"], next_cursor="loop"),
            "loop": _tools_page(["b"], next_cursor="loop"),
        }
        async with client:
            with pytest.raises(ProviderError, match="repeated cursor"):
                await client.list_tools()

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


class TestConnectFailure:
    @pytest.mark.asyncio
    async def test_failed_initialize_exits_the_stdio_context(
        self, monkeypatch, fake_session
    ):
        events: list[str] = []

        @asynccontextmanager
        async def fake_stdio_client(server_params):
            events.append("started")
            try:
                yield ("read", "write")
            finally:
                events.append("stopped")

        @asynccontextmanager
        async def fake_client_session(read, write):
            yield fake_session

        async def failing_initialize() -> None:
            raise RuntimeError("handshake failed")

        fake_session.initialize = failing_initialize
        monkeypatch.setattr(
            "agent_platform.integrations.mcp.stdio.provider.stdio_client",
            fake_stdio_client,
        )
        monkeypatch.setattr(
            "agent_platform.integrations.mcp.stdio.provider.ClientSession",
            fake_client_session,
        )
        client = StdioMCPClient(command="fake-server")

        with pytest.raises(RuntimeError, match="handshake failed"):
            await client.connect()

        assert events == ["started", "stopped"]
        assert client._exit_stack is None
        assert client._session is None
