from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from agent_platform.agents.errors import AgentRecoveryExhausted, AgentThinkError
from agent_platform.agents.guardrails import Guardrail, GuardrailContext
from agent_platform.agents.tools.base import ToolStreamChunk
from agent_platform.agents.tools.registry import (
    TOOL_CALL_VALIDATION_ERROR_KEY,
    ToolRegistry,
)
from agent_platform.agents.validation import retry_once_on_invalid
from agent_platform.core.cost import CostEstimator
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import ResponseFormat, StreamChunk
from agent_platform.core.middleware import MiddlewarePipeline
from agent_platform.core.schemas.message import (
    AssistantMessage,
    Message,
    Prompt,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.token_usage import TokenUsageAggregator

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        *,
        name: str,
        llm: BaseLLMProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        model: str = "default",
        generation_config: GenerationConfig | None = None,
        guardrails: list[Guardrail] | None = None,
        response_schema: type[BaseModel] | None = None,
        token_usage_aggregator: TokenUsageAggregator | None = None,
        usage_key: str | None = None,
        cost_estimator: CostEstimator | None = None,
    ) -> None:
        if not name:
            raise ValueError("Agent name must not be empty")
        self._name = name
        self._llm = llm
        self._tool_registry = tool_registry or ToolRegistry()
        self._system_prompt = system_prompt
        self._model = model
        self._generation_config = generation_config
        self._guardrail_pipeline = (
            MiddlewarePipeline(list(guardrails)) if guardrails else None
        )
        self._response_schema = response_schema
        self._usage_aggregator = token_usage_aggregator
        self._usage_key = usage_key or name
        self._cost_estimator = cost_estimator

    @property
    def name(self) -> str:
        return self._name

    @property
    def token_usage(self) -> TokenUsage:
        """Total `TokenUsage` recorded under this agent's `usage_key` so far.

        `TokenUsage.zero()` when no `token_usage_aggregator` was injected, so
        callers can read this unconditionally instead of checking for `None`
        first.
        """
        if self._usage_aggregator is None:
            return TokenUsage.zero()
        return self._usage_aggregator.total_for(self._usage_key)

    @property
    def estimated_cost(self) -> float | None:
        """Estimated dollar cost of `token_usage` so far, or `None` when no
        `cost_estimator` was injected (as opposed to `0.0`, a priced-but-free
        result, which a missing estimator can't produce)."""
        if self._cost_estimator is None:
            return None
        return self._cost_estimator.cost_for(self._model, self.token_usage)

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def response_schema(self) -> type[BaseModel] | None:
        return self._response_schema

    def _build_prompt(self, messages: list[Message]) -> Prompt:
        return Prompt.build(system=self._system_prompt, history=messages)

    def _build_generation_config(self) -> GenerationConfig:
        config = self._generation_config or GenerationConfig()
        if not config.model:
            config = config.model_copy(update={"model": self._model})
        if (
            self._response_schema is not None
            and config.response_format is ResponseFormat.TEXT
        ):
            config = config.model_copy(
                update={
                    "response_format": ResponseFormat.JSON_SCHEMA,
                    "json_schema": self._response_schema.model_json_schema(),
                }
            )
        return config

    def _build_call_args(
        self, messages: list[Message]
    ) -> tuple[Prompt, GenerationConfig, list | None]:
        tool_list = list(self._tool_registry)
        return (
            self._build_prompt(messages),
            self._build_generation_config(),
            tool_list or None,
        )

    async def _generate(self, messages: list[Message]) -> AssistantMessage:
        prompt, config, tools = self._build_call_args(messages)

        try:
            response = await self._llm.agenerate(
                prompt=prompt, config=config, tools=tools
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("LLM generation failed in agent '%s'", self._name)
            raise AgentThinkError(f"LLM generation failed: {exc}") from exc

        if response.message is None:
            raise AgentThinkError("LLM returned empty response")

        if self._usage_aggregator is not None:
            self._usage_aggregator.record(self._usage_key, response.usage)

        return response.message

    async def _generate_with_guardrails(
        self, messages: list[Message]
    ) -> AssistantMessage:
        if self._guardrail_pipeline is None:
            return await self._generate(messages)

        ctx = GuardrailContext(messages=messages)
        return await self._guardrail_pipeline.run(
            ctx, lambda c: self._generate(c.messages)
        )

    def _check_response_schema(self, assistant_msg: AssistantMessage) -> str | None:
        if assistant_msg.tool_calls or self._response_schema is None:
            return None
        try:
            self._response_schema.model_validate_json(assistant_msg.text)
        except PydanticValidationError as exc:
            return str(exc)
        return None

    async def think(self, messages: list[Message]) -> AssistantMessage:
        if self._response_schema is None:
            return await self._generate_with_guardrails(messages)

        working_messages = messages
        last_msg: AssistantMessage | None = None

        async def attempt() -> AssistantMessage:
            nonlocal last_msg
            last_msg = await self._generate_with_guardrails(working_messages)
            return last_msg

        async def correct(error: str) -> None:
            nonlocal working_messages
            assert last_msg is not None
            working_messages = [
                *messages,
                last_msg,
                UserMessage(
                    content=(
                        "Your previous response did not match the required JSON "
                        f"schema: {error}. Respond again with valid JSON only, "
                        "matching the schema exactly."
                    )
                ),
            ]

        try:
            return await retry_once_on_invalid(
                attempt=attempt,
                check=self._check_response_schema,
                correct=correct,
            )
        except AgentRecoveryExhausted as exc:
            result: AssistantMessage = exc.last_result
            return result

    async def think_stream(self, messages: list[Message]) -> AsyncIterator[StreamChunk]:
        prompt, config, tools = self._build_call_args(messages)

        try:
            async for chunk in self._llm.stream(
                prompt=prompt, config=config, tools=tools
            ):
                if chunk.usage is not None and self._usage_aggregator is not None:
                    self._usage_aggregator.record(self._usage_key, chunk.usage)
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("LLM streaming failed in agent '%s'", self._name)
            raise AgentThinkError(f"LLM streaming failed: {exc}") from exc

    async def act(self, assistant_message: AssistantMessage) -> list[ToolMessage]:
        return list(
            await asyncio.gather(
                *(
                    self._tool_registry.call_and_wrap(tc)
                    for tc in assistant_message.tool_calls
                )
            )
        )

    async def act_stream(
        self, assistant_message: AssistantMessage
    ) -> AsyncIterator[ToolStreamChunk | ToolMessage]:
        for tc in assistant_message.tool_calls:
            content = ""
            is_error = False
            is_validation_error = False
            async for chunk in self._tool_registry.call_and_stream(tc):
                yield chunk
                if chunk.is_final:
                    is_error = chunk.is_error
                    is_validation_error = chunk.is_validation_error
                    if is_error:
                        # final error chunk carries the whole message, not a fragment
                        content = chunk.delta
                    else:
                        content += chunk.delta
                elif not chunk.is_error:
                    content += chunk.delta

            yield ToolMessage(
                result=ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=content,
                    is_error=is_error,
                ),
                metadata=(
                    {TOOL_CALL_VALIDATION_ERROR_KEY: True}
                    if is_validation_error
                    else {}
                ),
            )

    async def step(
        self, messages: list[Message]
    ) -> tuple[AssistantMessage, list[ToolMessage]]:
        assistant_msg = await self.think(messages)
        if not assistant_msg.tool_calls:
            return assistant_msg, []
        tool_messages = await self.act(assistant_msg)
        return assistant_msg, tool_messages
