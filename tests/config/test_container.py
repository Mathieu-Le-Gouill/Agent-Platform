import asyncio
from typing import Any
from uuid import uuid4

import pytest

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.tools.errors import ToolRegistrationError
from agent_platform.config.container import (
    build_agent,
    build_agent_async,
    build_llm_with_fallback,
    build_provider,
    build_provider_from_model_string,
    build_tracer,
)
from agent_platform.config.settings import Settings
from agent_platform.core.errors import ConfigError
from agent_platform.core.interfaces.llm.fallback import FallbackLLMProvider
from agent_platform.core.interfaces.mcp.base import BaseMCPClient
from agent_platform.core.interfaces.tracing.noop import NoOpTracer
from agent_platform.core.schemas.mcp import MCPToolSpec
from agent_platform.core.schemas.message import Message, ToolCall, UserMessage
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations import llm as llm_module
from agent_platform.integrations.llm.anthropic.provider import AnthropicLLM
from agent_platform.integrations.llm.openai.provider import OpenAILLM


class TestBuildAgent:
    def test_builds_conversation_agent_with_openai(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            agent_name="test-agent",
        )

        agent = build_agent(settings)

        assert isinstance(agent, ConversationAgent)
        assert agent.name == "test-agent"

    def test_registers_image_and_audio_tools(self):
        settings = Settings(_env_file=None)

        agent = build_agent(settings)

        assert set(agent.tool_registry.all()) == {"generate_image", "transcribe"}

    def test_unknown_provider_raises_config_error(self):
        settings = Settings(_env_file=None, default_llm_model="does-not-exist:gpt-4o")

        with pytest.raises(ConfigError):
            build_agent(settings)

    def test_malformed_model_string_raises_config_error(self):
        settings = Settings(_env_file=None, default_llm_model="gpt-4o-mini")

        with pytest.raises(ConfigError):
            build_agent(settings)

    @staticmethod
    async def _kept_user_turns(agent: ConversationAgent, sent: int) -> int:
        history: list[Message] = [UserMessage(content=f"turn {i}") for i in range(sent)]
        assert agent._context_strategy is not None
        trimmed = await agent._context_strategy.trim(history)
        return len(trimmed)

    @pytest.mark.asyncio
    async def test_keeps_the_last_20_user_turns_by_default(self):
        agent = build_agent(Settings(_env_file=None))

        assert await self._kept_user_turns(agent, sent=30) == 20

    @pytest.mark.asyncio
    async def test_max_history_turns_setting_sets_the_limit(self):
        agent = build_agent(Settings(_env_file=None, max_history_turns=3))

        assert await self._kept_user_turns(agent, sent=30) == 3

    @pytest.mark.asyncio
    async def test_a_forked_conversation_keeps_the_limit(self):
        agent = build_agent(Settings(_env_file=None, max_history_turns=3))

        fork = agent.with_conversation(uuid4())

        assert await self._kept_user_turns(fork, sent=30) == 3

    def test_max_history_turns_must_be_positive(self):
        with pytest.raises(ValueError):
            Settings(_env_file=None, max_history_turns=0)

    def test_has_a_token_usage_aggregator_wired_in(self):
        settings = Settings(_env_file=None)

        agent = build_agent(settings)

        assert agent.token_usage == TokenUsage.zero()

    def test_uses_plain_provider_without_fallback_models(self):
        settings = Settings(_env_file=None)

        agent = build_agent(settings)

        assert isinstance(agent._llm, OpenAILLM)

    def test_wraps_in_fallback_provider_when_configured(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            fallback_llm_models=["anthropic:claude-sonnet-4-5"],
        )

        agent = build_agent(settings)

        assert isinstance(agent._llm, FallbackLLMProvider)


class TestBuildLlmWithFallback:
    def test_no_fallback_models_returns_primary_directly(self):
        settings = Settings(_env_file=None, default_llm_model="openai:gpt-4o-mini")

        llm, model = build_llm_with_fallback(settings)

        assert isinstance(llm, OpenAILLM)
        assert model == "gpt-4o-mini"

    def test_fallback_models_build_a_chain(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            fallback_llm_models=["anthropic:claude-sonnet-4-5"],
        )

        llm, model = build_llm_with_fallback(settings)

        assert isinstance(llm, FallbackLLMProvider)
        assert model == "gpt-4o-mini"
        assert isinstance(llm._entries[0].provider, OpenAILLM)
        assert isinstance(llm._entries[1].provider, AnthropicLLM)
        assert llm._entries[1].model == "claude-sonnet-4-5"

    def test_no_rate_limiter_when_llm_rate_limit_unset(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            fallback_llm_models=["anthropic:claude-sonnet-4-5"],
        )

        llm, _ = build_llm_with_fallback(settings)

        assert llm._rate_limiter is None

    def test_wires_rate_limiter_when_configured(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            fallback_llm_models=["anthropic:claude-sonnet-4-5"],
            llm_rate_limit=5.0,
            llm_rate_limit_burst=10.0,
        )

        llm, _ = build_llm_with_fallback(settings)

        assert isinstance(llm, FallbackLLMProvider)
        assert llm._rate_limiter is not None

    def test_the_model_and_the_options_reach_the_provider(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            llm_options={"openai": {"temperature": 0.3}},
        )

        llm, _ = build_llm_with_fallback(settings)

        resolved = llm._resolve(None)
        assert (resolved.model, resolved.temperature) == ("gpt-4o-mini", 0.3)

    def test_options_are_found_when_the_provider_is_named_by_class(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="OpenAILLM:gpt-4o-mini",
            llm_options={"openai": {"temperature": 0.3}},
        )

        llm, _ = build_llm_with_fallback(settings)

        assert llm._resolve(None).temperature == 0.3

    def test_each_fallback_entry_gets_its_own_model_and_options(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            fallback_llm_models=["anthropic:claude-sonnet-4-5"],
            llm_options={
                "openai": {"temperature": 0.1},
                "anthropic": {"effort": "high"},
            },
        )

        llm, _ = build_llm_with_fallback(settings)

        openai_cfg = llm._entries[0].provider._resolve(None)
        anthropic_cfg = llm._entries[1].provider._resolve(None)
        assert (openai_cfg.model, openai_cfg.temperature) == ("gpt-4o-mini", 0.1)
        assert (anthropic_cfg.model, anthropic_cfg.effort) == (
            "claude-sonnet-4-5",
            "high",
        )
        assert not hasattr(openai_cfg, "effort")
        assert anthropic_cfg.temperature is None

    def test_no_options_leave_the_provider_unconfigured(self):
        llm, _ = build_llm_with_fallback(Settings(_env_file=None))

        assert llm._resolve(None).temperature is None

    def test_a_bad_option_key_fails_at_build_agent(self):
        settings = Settings(_env_file=None, llm_options={"openai": {"nope": 1}})

        with pytest.raises(ConfigError, match="nope"):
            build_agent(settings)

    def test_the_model_cannot_be_an_option(self):
        settings = Settings(_env_file=None, llm_options={"openai": {"model": "x"}})

        with pytest.raises(ConfigError, match="model"):
            build_agent(settings)

    def test_an_unknown_provider_slug_fails_at_build_agent(self):
        settings = Settings(_env_file=None, llm_options={"nope": {"a": 1}})

        with pytest.raises(ConfigError, match="nope"):
            build_agent(settings)

    def test_an_entry_for_a_provider_outside_the_chain_is_still_validated(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            llm_options={"anthropic": {"nope": 1}},
        )

        with pytest.raises(ConfigError, match="nope"):
            build_llm_with_fallback(settings)

    def test_an_error_names_the_field_and_never_the_value(self):
        settings = Settings(
            _env_file=None,
            llm_options={"openai": {"temperature": "s3cret-value"}},
        )

        with pytest.raises(ConfigError) as caught:
            build_llm_with_fallback(settings)

        assert "temperature" in str(caught.value)
        assert "s3cret-value" not in str(caught.value)


class TestBuildProvider:
    def test_init_kwargs_reach_the_constructor(self):
        provider = build_provider(llm_module, "openai", model="gpt-4o")

        assert provider._model == "gpt-4o"

    def test_instantiates_provider_by_alias(self):
        provider = build_provider(llm_module, "openai")

        assert isinstance(provider, OpenAILLM)

    def test_instantiates_provider_by_class_name(self):
        provider = build_provider(llm_module, "OpenAILLM")

        assert isinstance(provider, OpenAILLM)

    def test_unknown_provider_raises_config_error(self):
        with pytest.raises(ConfigError):
            build_provider(llm_module, "does-not-exist")


class TestBuildProviderFromModelString:
    def test_resolves_provider_and_model(self):
        provider, model = build_provider_from_model_string(
            llm_module, "openai:gpt-4o-mini"
        )

        assert isinstance(provider, OpenAILLM)
        assert model == "gpt-4o-mini"

    def test_malformed_string_raises_config_error(self):
        with pytest.raises(ConfigError):
            build_provider_from_model_string(llm_module, "gpt-4o-mini")

    def test_unknown_provider_raises_config_error(self):
        with pytest.raises(ConfigError):
            build_provider_from_model_string(llm_module, "does-not-exist:gpt-4o-mini")


class TestBuildMCPClient:
    @pytest.mark.asyncio
    async def test_splits_command_and_args(self):
        pytest.importorskip("mcp")
        from agent_platform.config.container import _build_mcp_client
        from agent_platform.integrations.mcp.stdio.provider import StdioMCPClient

        client = await _build_mcp_client(["npx", "-y", "server-docs"])

        assert isinstance(client, StdioMCPClient)
        assert client._server_params.command == "npx"
        assert client._server_params.args == ["-y", "server-docs"]


class _FakeMCPClient(BaseMCPClient):
    def __init__(self, tools: list[MCPToolSpec]) -> None:
        self._tools = tools
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def aclose(self) -> None:
        self.closed = True

    async def list_tools(self, config: Any = None) -> list[MCPToolSpec]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict, config: Any = None) -> Any:
        return f"{name} called"


class TestBuildAgentAsync:
    @pytest.mark.asyncio
    async def test_no_mcp_servers_behaves_like_build_agent(self):
        settings = Settings(_env_file=None)

        agent, clients = await build_agent_async(settings)

        assert isinstance(agent, ConversationAgent)
        assert clients == []
        assert set(agent.tool_registry.all()) == {"generate_image", "transcribe"}

    @pytest.mark.asyncio
    async def test_connects_and_registers_mcp_tools(self, mocker):
        fake_client = _FakeMCPClient(
            [
                MCPToolSpec(
                    name="search_docs", description="Search docs", input_schema={}
                )
            ]
        )
        mocker.patch(
            "agent_platform.config.container._build_mcp_client",
            return_value=fake_client,
        )
        settings = Settings(
            _env_file=None, mcp_stdio_servers={"docs": ["npx", "-y", "server-docs"]}
        )

        agent, clients = await build_agent_async(settings)

        assert clients == [fake_client]
        assert fake_client.connected is True
        assert "search_docs" in agent.tool_registry.all()

    @pytest.mark.asyncio
    async def test_passes_command_and_args_to_client_factory(self, mocker):
        build_client = mocker.patch(
            "agent_platform.config.container._build_mcp_client",
            return_value=_FakeMCPClient([]),
        )
        settings = Settings(
            _env_file=None, mcp_stdio_servers={"docs": ["npx", "-y", "server-docs"]}
        )

        await build_agent_async(settings)

        build_client.assert_awaited_once_with(["npx", "-y", "server-docs"])

    @pytest.mark.asyncio
    async def test_multiple_servers_all_connected_and_returned(self, mocker):
        client_a = _FakeMCPClient([])
        client_b = _FakeMCPClient([])
        mocker.patch(
            "agent_platform.config.container._build_mcp_client",
            side_effect=[client_a, client_b],
        )
        settings = Settings(
            _env_file=None,
            mcp_stdio_servers={"a": ["cmd-a"], "b": ["cmd-b"]},
        )

        _, clients = await build_agent_async(settings)

        assert clients == [client_a, client_b]
        assert client_a.connected is True
        assert client_b.connected is True

    @pytest.mark.asyncio
    async def test_failed_second_connect_closes_the_first_client(self, mocker):
        client_a = _FakeMCPClient([])
        client_b = _FakeMCPClient([])
        client_b.connect = mocker.AsyncMock(side_effect=RuntimeError("no server"))
        mocker.patch(
            "agent_platform.config.container._build_mcp_client",
            side_effect=[client_a, client_b],
        )
        settings = Settings(
            _env_file=None, mcp_stdio_servers={"a": ["cmd-a"], "b": ["cmd-b"]}
        )

        with pytest.raises(RuntimeError, match="no server"):
            await build_agent_async(settings)

        assert client_a.closed is True
        assert client_b.closed is False

    @pytest.mark.asyncio
    async def test_tool_name_clash_closes_every_started_client(self, mocker):
        spec = MCPToolSpec(name="search_docs", description="Search", input_schema={})
        client_a = _FakeMCPClient([spec])
        client_b = _FakeMCPClient([spec])
        mocker.patch(
            "agent_platform.config.container._build_mcp_client",
            side_effect=[client_a, client_b],
        )
        settings = Settings(
            _env_file=None, mcp_stdio_servers={"a": ["cmd-a"], "b": ["cmd-b"]}
        )

        with pytest.raises(ToolRegistrationError):
            await build_agent_async(settings)

        assert client_a.closed is True
        assert client_b.closed is True

    @pytest.mark.asyncio
    async def test_failing_close_does_not_mask_the_original_error(self, mocker):
        client_a = _FakeMCPClient([])
        client_a.aclose = mocker.AsyncMock(side_effect=OSError("close failed"))
        client_b = _FakeMCPClient([])
        client_b.connect = mocker.AsyncMock(side_effect=RuntimeError("no server"))
        mocker.patch(
            "agent_platform.config.container._build_mcp_client",
            side_effect=[client_a, client_b],
        )
        settings = Settings(
            _env_file=None, mcp_stdio_servers={"a": ["cmd-a"], "b": ["cmd-b"]}
        )

        with pytest.raises(RuntimeError, match="no server"):
            await build_agent_async(settings)

    @pytest.mark.asyncio
    async def test_mcp_tools_are_registered_with_the_configured_timeout(self, mocker):
        client = _FakeMCPClient(
            [MCPToolSpec(name="slow", description="Slow", input_schema={})]
        )
        mocker.patch(
            "agent_platform.config.container._build_mcp_client",
            return_value=client,
        )
        settings = Settings(
            _env_file=None,
            mcp_stdio_servers={"docs": ["cmd"]},
            mcp_tool_timeout=0.05,
        )

        agent, _ = await build_agent_async(settings)

        async def hang(name: str, arguments: dict, config: Any = None) -> Any:
            await asyncio.sleep(5)

        client.call_tool = hang  # type: ignore[method-assign]
        message = await agent.tool_registry.call_and_wrap(
            ToolCall(id="c1", name="slow", arguments={})
        )

        assert message.result.is_error is True
        assert "timed out after 0.05s" in message.result.content


class TestBuildTracer:
    def test_defaults_to_the_langfuse_tracer(self):
        from agent_platform.integrations.tracing.langfuse.provider import (
            LangfuseTracer,
        )

        tracer = build_tracer(Settings(_env_file=None))

        assert isinstance(tracer, LangfuseTracer)

    def test_none_disables_tracing(self):
        tracer = build_tracer(Settings(_env_file=None, tracing_provider="none"))

        assert isinstance(tracer, NoOpTracer)

    def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError):
            build_tracer(Settings(_env_file=None, tracing_provider="does-not-exist"))

    def test_falls_back_to_noop_when_the_sdk_is_missing(self, mocker):
        mocker.patch(
            "agent_platform.config.container.build_provider",
            side_effect=ImportError("no langfuse"),
        )

        tracer = build_tracer(Settings(_env_file=None))

        assert isinstance(tracer, NoOpTracer)
