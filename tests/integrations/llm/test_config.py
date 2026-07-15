import pytest

from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.core.interfaces.llm.response import ResponseFormat


class TestResponseFormat:
    def test_values(self):
        assert ResponseFormat.TEXT.value == "text"
        assert ResponseFormat.JSON.value == "json"
        assert ResponseFormat.JSON_SCHEMA.value == "json_schema"


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens is None
        assert cfg.top_p is None
        assert cfg.top_k is None
        assert cfg.stop_sequences == []
        assert cfg.seed is None
        assert cfg.frequency_penalty is None
        assert cfg.presence_penalty is None
        assert cfg.timeout is None
        assert cfg.max_retries is None
        assert cfg.response_format == ResponseFormat.TEXT
        assert cfg.json_schema is None

    def test_construction(self):
        cfg = GenerationConfig(
            temperature=0.1,
            max_tokens=100,
            top_p=0.9,
            top_k=50,
            stop_sequences=["stop"],
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            timeout=30.0,
            max_retries=5,
            response_format=ResponseFormat.JSON,
            json_schema={"type": "object"},
        )
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 100
        assert cfg.top_p == 0.9
        assert cfg.top_k == 50
        assert cfg.stop_sequences == ["stop"]
        assert cfg.seed == 42
        assert cfg.frequency_penalty == 0.5
        assert cfg.presence_penalty == 0.5
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 5
        assert cfg.response_format == ResponseFormat.JSON
        assert cfg.json_schema == {"type": "object"}


class TestAnthropicGenerationConfig:
    def test_defaults(self):
        cfg = AnthropicGenerationConfig()
        assert cfg.temperature == 0.7
        assert cfg.thinking is False
        assert cfg.thinking_budget is None
        assert cfg.cache_control is False

    def test_construction(self):
        cfg = AnthropicGenerationConfig(
            thinking=True,
            thinking_budget=10000,
            cache_control=True,
        )
        assert cfg.thinking is True
        assert cfg.thinking_budget == 10000
        assert cfg.cache_control is True


class TestOpenAIGenerationConfig:
    def test_defaults(self):
        cfg = OpenAIGenerationConfig()
        assert cfg.temperature == 0.7
        assert cfg.reasoning_effort is None
        assert cfg.parallel_tool_calls is True

    def test_construction(self):
        cfg = OpenAIGenerationConfig(
            reasoning_effort="high",
            parallel_tool_calls=False,
        )
        assert cfg.reasoning_effort == "high"
        assert cfg.parallel_tool_calls is False
