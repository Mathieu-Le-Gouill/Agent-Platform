from uuid import uuid4

import pytest
from pydantic import BaseModel, SecretStr

from agent_platform.agents.tools.base import Tool
from agent_platform.integrations.llm.config import (
    GenerationConfig,
    AnthropicConfig,
    OpenAIConfig,
    ResponseFormat,
)
from agent_platform.integrations.llm.providers.anthropic import (
    AnthropicLLM,
    _to_langchain_anthropic,
)
from agent_platform.integrations.llm.providers.openai import (
    OpenAILLM,
    _to_langchain_openai,
)
from agent_platform.integrations.llm.providers.mistral import (
    MistralLLM,
    _to_langchain_mistral,
)
from agent_platform.integrations.llm.providers.ollama import (
    OllamaLLM,
    _to_langchain_ollama,
)


class TestToLangchainAnthropic:
    def test_none_config(self):
        result = _to_langchain_anthropic(None)
        assert result == {"max_tokens": 1024}

    def test_default_config(self):
        result = _to_langchain_anthropic(GenerationConfig())
        assert result == {
            "max_tokens": 1024,
            "temperature": 0.7,
            "max_retries": 3,
        }

    def test_with_max_tokens(self):
        cfg = GenerationConfig(max_tokens=500)
        result = _to_langchain_anthropic(cfg)
        assert result["max_tokens"] == 500
        assert result["temperature"] == 0.7

    def test_with_all_fields(self):
        cfg = GenerationConfig(
            temperature=0.1,
            max_tokens=200,
            top_p=0.9,
            top_k=40,
            stop_sequences=["stop1"],
            timeout=30.0,
            max_retries=5,
        )
        result = _to_langchain_anthropic(cfg)
        assert result["temperature"] == 0.1
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["stop_sequences"] == ["stop1"]
        assert result["timeout"] == 30.0
        assert result["max_retries"] == 5

    def test_anthropic_config_thinking(self):
        cfg = AnthropicConfig(thinking=True)
        result = _to_langchain_anthropic(cfg)
        assert result["thinking"] == {
            "type": "enabled",
            "budget_tokens": 5000,
        }

    def test_anthropic_config_thinking_with_budget(self):
        cfg = AnthropicConfig(thinking=True, thinking_budget=10000)
        result = _to_langchain_anthropic(cfg)
        assert result["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }

    def test_anthropic_config_cache_control(self):
        cfg = AnthropicConfig(cache_control=True)
        result = _to_langchain_anthropic(cfg)
        assert result["cache_control"] == {"type": "ephemeral"}

    def test_anthropic_config_thinking_disabled(self):
        cfg = AnthropicConfig(thinking=False, cache_control=True)
        result = _to_langchain_anthropic(cfg)
        assert "thinking" not in result
        assert result["cache_control"] == {"type": "ephemeral"}


class TestToLangchainOpenAI:
    def test_none_config(self):
        result = _to_langchain_openai(None)
        assert result == {}

    def test_default_config(self):
        result = _to_langchain_openai(GenerationConfig())
        assert result == {"temperature": 0.7, "max_retries": 3}

    def test_with_fields(self):
        cfg = GenerationConfig(
            temperature=0.2,
            max_tokens=100,
            top_p=0.8,
            stop_sequences=["stop"],
            seed=42,
            timeout=15.0,
            max_retries=3,
            frequency_penalty=0.3,
            presence_penalty=0.4,
        )
        result = _to_langchain_openai(cfg)
        assert result["temperature"] == 0.2
        assert result["max_tokens"] == 100
        assert result["top_p"] == 0.8
        assert result["stop"] == ["stop"]
        assert result["seed"] == 42
        assert result["timeout"] == 15.0
        assert result["max_retries"] == 3
        assert result["model_kwargs"]["frequency_penalty"] == 0.3
        assert result["model_kwargs"]["presence_penalty"] == 0.4

    def test_openai_config_reasoning_effort(self):
        cfg = OpenAIConfig(reasoning_effort="high")
        result = _to_langchain_openai(cfg)
        assert result["model_kwargs"]["reasoning_effort"] == "high"

    def test_openai_config_parallel_tool_calls_false(self):
        cfg = OpenAIConfig(parallel_tool_calls=False)
        result = _to_langchain_openai(cfg)
        assert result["model_kwargs"]["parallel_tool_calls"] is False

    def test_openai_config_parallel_tool_calls_default(self):
        cfg = OpenAIConfig()
        result = _to_langchain_openai(cfg)
        assert "parallel_tool_calls" not in result.get("model_kwargs", {})

    def test_response_format_json(self):
        cfg = GenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_langchain_openai(cfg)
        assert result["model_kwargs"]["response_format"] == {"type": "json_object"}

    def test_response_format_json_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cfg = GenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
        )
        result = _to_langchain_openai(cfg)
        assert result["model_kwargs"]["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }

    def test_json_schema_missing_raises(self):
        cfg = GenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_langchain_openai(cfg)


class TestToLangchainMistral:
    def test_none_config(self):
        result = _to_langchain_mistral(None)
        assert result == {}

    def test_default_config(self):
        result = _to_langchain_mistral(GenerationConfig())
        assert result == {"temperature": 0.7, "max_retries": 3}

    def test_with_fields(self):
        cfg = GenerationConfig(
            temperature=0.3,
            max_tokens=200,
            top_p=0.9,
            stop_sequences=["stop"],
            seed=123,
            timeout=20.0,
            max_retries=4,
            frequency_penalty=0.2,
            presence_penalty=0.1,
        )
        result = _to_langchain_mistral(cfg)
        assert result["temperature"] == 0.3
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["stop"] == ["stop"]
        assert result["random_seed"] == 123
        assert result["timeout"] == 20.0
        assert result["max_retries"] == 4
        assert result["model_kwargs"]["frequency_penalty"] == 0.2
        assert result["model_kwargs"]["presence_penalty"] == 0.1

    def test_response_format_json(self):
        cfg = GenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_langchain_mistral(cfg)
        assert result["model_kwargs"]["response_format"] == {"type": "json_object"}

    def test_response_format_json_schema(self):
        schema = {"type": "object"}
        cfg = GenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
        )
        result = _to_langchain_mistral(cfg)
        assert result["model_kwargs"]["response_format"] == {
            "type": "json_schema",
            "json_schema": schema,
        }

    def test_json_schema_missing_raises(self):
        cfg = GenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_langchain_mistral(cfg)


class TestToLangchainOllama:
    def test_none_config(self):
        result = _to_langchain_ollama(None)
        assert result == {}

    def test_default_config(self):
        result = _to_langchain_ollama(GenerationConfig())
        assert result == {"temperature": 0.7}

    def test_with_fields(self):
        cfg = GenerationConfig(
            temperature=0.5,
            max_tokens=500,
            top_p=0.95,
            top_k=40,
            seed=99,
            stop_sequences=["stop"],
            frequency_penalty=0.3,
            timeout=60.0,
        )
        result = _to_langchain_ollama(cfg)
        assert result["temperature"] == 0.5
        assert result["num_predict"] == 500
        assert result["top_p"] == 0.95
        assert result["top_k"] == 40
        assert result["seed"] == 99
        assert result["stop"] == ["stop"]
        assert result["repeat_penalty"] == 0.3
        assert result["timeout"] == 60.0

    def test_response_format_json(self):
        cfg = GenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_langchain_ollama(cfg)
        assert result["format"] == "json"

    def test_response_format_json_schema(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        cfg = GenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
        )
        result = _to_langchain_ollama(cfg)
        assert result["format"] == schema

    def test_json_schema_missing_raises(self):
        cfg = GenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_langchain_ollama(cfg)


class TestAnthropicLLMConstruction:
    def test_construct(self):
        provider = AnthropicLLM(api_key=SecretStr("sk-ant-123"))
        assert provider._api_key.get_secret_value() == "sk-ant-123"


class TestOpenAILLMConstruction:
    def test_construct(self):
        provider = OpenAILLM(api_key=SecretStr("sk-123"))
        assert provider._api_key.get_secret_value() == "sk-123"


class TestMistralLLMConstruction:
    def test_construct(self):
        provider = MistralLLM(api_key=SecretStr("mist-123"))
        assert provider._api_key.get_secret_value() == "mist-123"


class TestOllamaLLMConstruction:
    def test_construct(self):
        provider = OllamaLLM()
        assert hasattr(provider, "_client")


class _SchemaTool(Tool):
    name = "test_tool"
    description = "A test tool"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return None


class OpenAISchemaTest:
    def test_openai_schema(self):
        provider = OpenAILLM(api_key=SecretStr("sk-test"))
        tool = _SchemaTool()
        schema = provider._tool_to_schema(tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert "parameters" in schema["function"]

    def test_anthropic_schema(self):
        provider = AnthropicLLM(api_key=SecretStr("sk-ant-test"))
        tool = _SchemaTool()
        schema = provider._tool_to_schema(tool)
        assert schema["name"] == "test_tool"
        assert "input_schema" in schema

    def test_mistral_schema(self):
        provider = MistralLLM(api_key=SecretStr("sk-mist-test"))
        tool = _SchemaTool()
        schema = provider._tool_to_schema(tool)
        assert schema["type"] == "function"

    def test_ollama_schema(self):
        provider = OllamaLLM()
        tool = _SchemaTool()
        schema = provider._tool_to_schema(tool)
        assert schema["type"] == "function"
