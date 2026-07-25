import pytest
from pydantic import BaseModel, SecretStr

pytest.importorskip("langchain_anthropic")
pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_mistralai")
pytest.importorskip("langchain_ollama")

from agent_platform.agents.tools.base import Tool
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.integrations.credentials import (
    AnthropicCredentials,
    MistralCredentials,
    OllamaCredentials,
    OpenAICredentials,
)
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.anthropic.provider import (
    AnthropicLLM,
    _to_langchain_anthropic,
)
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.llm.mistral.provider import (
    MistralLLM,
    _to_langchain_mistral,
)
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.llm.ollama.provider import (
    OllamaLLM,
    _to_langchain_ollama,
)
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.integrations.llm.openai.provider import (
    OpenAILLM,
    _to_langchain_openai,
)


class TestToLangchainAnthropic:
    def _creds(self):
        return AnthropicCredentials(api_key=SecretStr("test"))

    def test_default_config(self):
        cfg = AnthropicGenerationConfig()
        result = _to_langchain_anthropic(cfg, self._creds())
        assert result == {
            "max_tokens": 1024,
            "temperature": 0.7,
            "max_retries": 3,
        }

    def test_with_max_tokens(self):
        cfg = AnthropicGenerationConfig(max_tokens=500)
        result = _to_langchain_anthropic(cfg, self._creds())
        assert result["max_tokens"] == 500
        assert result["temperature"] == 0.7

    def test_with_all_fields(self):
        cfg = AnthropicGenerationConfig(
            temperature=0.1,
            max_tokens=200,
            top_p=0.9,
            top_k=40,
            stop_sequences=["stop1"],
            timeout=30.0,
            max_retries=5,
        )
        result = _to_langchain_anthropic(cfg, self._creds())
        assert result["temperature"] == 0.1
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["stop_sequences"] == ["stop1"]
        assert result["timeout"] == 30.0
        assert result["max_retries"] == 5

    def test_anthropic_config_thinking(self):
        cfg = AnthropicGenerationConfig(thinking=True)
        result = _to_langchain_anthropic(cfg, self._creds())
        # Default max_tokens must widen so budget_tokens < max_tokens holds
        # (Anthropic 400s otherwise) — the bug this fix addresses.
        assert result["max_tokens"] == 8192
        assert result["thinking"] == {
            "type": "enabled",
            "budget_tokens": 5000,
        }
        assert result["thinking"]["budget_tokens"] < result["max_tokens"]

    def test_anthropic_config_thinking_with_budget(self):
        cfg = AnthropicGenerationConfig(thinking=True, thinking_budget=10000)
        result = _to_langchain_anthropic(cfg, self._creds())
        assert result["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }
        assert result["thinking"]["budget_tokens"] < result["max_tokens"]

    def test_anthropic_config_thinking_budget_exceeds_explicit_max_tokens(self):
        # Explicit max_tokens smaller than the default thinking_budget must not
        # produce budget_tokens >= max_tokens (the original 400-on-defaults bug).
        cfg = AnthropicGenerationConfig(thinking=True, max_tokens=1024)
        result = _to_langchain_anthropic(cfg, self._creds())
        assert result["max_tokens"] == 1024
        assert result["thinking"]["budget_tokens"] < result["max_tokens"]

    def test_anthropic_config_effort(self):
        cfg = AnthropicGenerationConfig(effort="high")
        result = _to_langchain_anthropic(cfg, self._creds())
        assert result["effort"] == "high"
        assert "thinking" not in result
        assert result["max_tokens"] == 1024

    def test_anthropic_config_effort_takes_precedence_over_thinking(self):
        cfg = AnthropicGenerationConfig(effort="max", thinking=True)
        result = _to_langchain_anthropic(cfg, self._creds())
        assert result["effort"] == "max"
        assert "thinking" not in result

    def test_anthropic_config_cache_control(self):
        cfg = AnthropicGenerationConfig(cache_control=True)
        result = _to_langchain_anthropic(cfg, self._creds())
        # cache_control is not a real ChatAnthropic constructor field; it must
        # be routed through model_kwargs, not passed top-level.
        assert result["model_kwargs"] == {"cache_control": {"type": "ephemeral"}}
        assert "cache_control" not in result

    def test_anthropic_config_thinking_disabled(self):
        cfg = AnthropicGenerationConfig(thinking=False, cache_control=True)
        result = _to_langchain_anthropic(cfg, self._creds())
        assert "thinking" not in result
        assert result["model_kwargs"] == {"cache_control": {"type": "ephemeral"}}


class TestToLangchainOpenAI:
    def _creds(self):
        return OpenAICredentials(api_key=SecretStr("test"))

    def test_default_config(self):
        cfg = OpenAIGenerationConfig()
        result = _to_langchain_openai(cfg, self._creds())
        # temperature equals the field default -> omitted, not sent unconditionally.
        assert result == {"max_retries": 3}

    def test_with_fields(self):
        cfg = OpenAIGenerationConfig(
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
        result = _to_langchain_openai(cfg, self._creds())
        assert result["temperature"] == 0.2
        assert result["max_tokens"] == 100
        assert result["top_p"] == 0.8
        assert result["stop"] == ["stop"]
        assert result["seed"] == 42
        assert result["timeout"] == 15.0
        assert result["max_retries"] == 3
        assert result["model_kwargs"]["frequency_penalty"] == 0.3
        assert result["model_kwargs"]["presence_penalty"] == 0.4

    def test_temperature_at_default_is_omitted(self):
        cfg = OpenAIGenerationConfig(temperature=0.7)
        result = _to_langchain_openai(cfg, self._creds())
        assert "temperature" not in result

    def test_temperature_non_default_non_reasoning_model_is_sent(self):
        cfg = OpenAIGenerationConfig(model="gpt-4.1", temperature=0.2)
        result = _to_langchain_openai(cfg, self._creds())
        assert result["temperature"] == 0.2

    @pytest.mark.parametrize(
        "model", ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"]
    )
    def test_temperature_dropped_for_reasoning_models(self, model):
        cfg = OpenAIGenerationConfig(model=model, temperature=0.2)
        result = _to_langchain_openai(cfg, self._creds())
        assert "temperature" not in result

    def test_temperature_kept_for_gpt5_chat(self):
        cfg = OpenAIGenerationConfig(model="gpt-5-chat", temperature=0.2)
        result = _to_langchain_openai(cfg, self._creds())
        assert result["temperature"] == 0.2

    def test_openai_config_reasoning_effort(self):
        cfg = OpenAIGenerationConfig(model="o3", reasoning_effort="high")
        result = _to_langchain_openai(cfg, self._creds())
        assert result["model_kwargs"]["reasoning_effort"] == "high"

    def test_openai_config_reasoning_effort_dropped_for_non_reasoning_model(self):
        cfg = OpenAIGenerationConfig(model="gpt-4.1", reasoning_effort="high")
        result = _to_langchain_openai(cfg, self._creds())
        assert "reasoning_effort" not in result.get("model_kwargs", {})

    def test_openai_config_parallel_tool_calls_false(self):
        cfg = OpenAIGenerationConfig(parallel_tool_calls=False)
        result = _to_langchain_openai(cfg, self._creds())
        assert result["model_kwargs"]["parallel_tool_calls"] is False

    def test_openai_config_parallel_tool_calls_default(self):
        cfg = OpenAIGenerationConfig()
        result = _to_langchain_openai(cfg, self._creds())
        assert "parallel_tool_calls" not in result.get("model_kwargs", {})

    def test_response_format_json(self):
        cfg = OpenAIGenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_langchain_openai(cfg, self._creds())
        assert result["model_kwargs"]["response_format"] == {"type": "json_object"}

    def test_response_format_json_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
        )
        result = _to_langchain_openai(cfg, self._creds())
        assert result["model_kwargs"]["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }

    def test_response_format_json_schema_strict_false(self):
        schema = {"type": "object"}
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
            strict=False,
        )
        result = _to_langchain_openai(cfg, self._creds())
        assert (
            result["model_kwargs"]["response_format"]["json_schema"]["strict"] is False
        )

    def test_json_schema_missing_raises(self):
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_langchain_openai(cfg, self._creds())


class TestToLangchainMistral:
    def _creds(self):
        return MistralCredentials(api_key=SecretStr("test"))

    def test_default_config(self):
        cfg = MistralGenerationConfig()
        result = _to_langchain_mistral(cfg, self._creds())
        assert result == {"temperature": 0.7, "max_retries": 3}

    def test_with_fields(self):
        cfg = MistralGenerationConfig(
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
        result = _to_langchain_mistral(cfg, self._creds())
        assert result["temperature"] == 0.3
        assert result["max_tokens"] == 200
        assert result["top_p"] == 0.9
        assert result["stop"] == ["stop"]
        assert result["random_seed"] == 123
        assert result["timeout"] == 20
        assert result["max_retries"] == 4
        assert result["model_kwargs"]["frequency_penalty"] == 0.2
        assert result["model_kwargs"]["presence_penalty"] == 0.1

    def test_response_format_json(self):
        cfg = MistralGenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_langchain_mistral(cfg, self._creds())
        assert result["model_kwargs"]["response_format"] == {"type": "json_object"}

    def test_response_format_json_schema(self):
        schema = {"type": "object"}
        cfg = MistralGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
        )
        result = _to_langchain_mistral(cfg, self._creds())
        assert result["model_kwargs"]["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "schema": schema,
                "name": "response",
                "strict": True,
            },
        }

    def test_response_format_json_schema_custom_name_and_strict(self):
        schema = {"type": "object"}
        cfg = MistralGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
            json_schema_name="my_schema",
            json_schema_strict=False,
        )
        result = _to_langchain_mistral(cfg, self._creds())
        assert result["model_kwargs"]["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "schema": schema,
                "name": "my_schema",
                "strict": False,
            },
        }

    def test_json_schema_missing_raises(self):
        cfg = MistralGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_langchain_mistral(cfg, self._creds())


class TestToLangchainOllama:
    def _creds(self):
        return OllamaCredentials()

    def test_default_config(self):
        cfg = OllamaGenerationConfig()
        result = _to_langchain_ollama(cfg, self._creds())
        assert result == {"temperature": 0.7}

    def test_with_fields(self):
        cfg = OllamaGenerationConfig(
            temperature=0.5,
            max_tokens=500,
            top_p=0.95,
            top_k=40,
            seed=99,
            stop_sequences=["stop"],
            timeout=60.0,
        )
        result = _to_langchain_ollama(cfg, self._creds())
        assert result["temperature"] == 0.5
        assert result["num_predict"] == 500
        assert result["top_p"] == 0.95
        assert result["top_k"] == 40
        assert result["seed"] == 99
        assert result["stop"] == ["stop"]
        assert result["timeout"] == 60.0

    def test_frequency_penalty_is_not_remapped_to_repeat_penalty(self):
        # frequency_penalty (additive, OpenAI/Anthropic-shaped) must NOT be
        # cross-mapped into Ollama's repeat_penalty (multiplicative) -- the
        # semantics don't correspond.
        cfg = OllamaGenerationConfig(frequency_penalty=0.3)
        result = _to_langchain_ollama(cfg, self._creds())
        assert "repeat_penalty" not in result

    def test_repeat_penalty_native_field(self):
        cfg = OllamaGenerationConfig(repeat_penalty=1.2)
        result = _to_langchain_ollama(cfg, self._creds())
        assert result["repeat_penalty"] == 1.2

    def test_mirostat_fields(self):
        cfg = OllamaGenerationConfig(mirostat=2, mirostat_tau=5.0, mirostat_eta=0.1)
        result = _to_langchain_ollama(cfg, self._creds())
        assert result["mirostat"] == 2
        assert result["mirostat_tau"] == 5.0
        assert result["mirostat_eta"] == 0.1

    def test_num_ctx_field(self):
        cfg = OllamaGenerationConfig(num_ctx=4096)
        result = _to_langchain_ollama(cfg, self._creds())
        assert result["num_ctx"] == 4096

    def test_optional_fields_omitted_when_unset(self):
        cfg = OllamaGenerationConfig()
        result = _to_langchain_ollama(cfg, self._creds())
        for key in (
            "repeat_penalty",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "num_ctx",
        ):
            assert key not in result

    def test_response_format_json(self):
        cfg = OllamaGenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_langchain_ollama(cfg, self._creds())
        assert result["format"] == "json"

    def test_response_format_json_schema(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        cfg = OllamaGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA,
            json_schema=schema,
        )
        result = _to_langchain_ollama(cfg, self._creds())
        assert result["format"] == schema

    def test_json_schema_missing_raises(self):
        cfg = OllamaGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_langchain_ollama(cfg, self._creds())


class TestAnthropicLLMConstruction:
    def test_construct(self):
        provider = AnthropicLLM(AnthropicCredentials(api_key=SecretStr("sk-ant-123")))
        assert provider._credentials.api_key.get_secret_value() == "sk-ant-123"

    def test_default_config(self):
        provider = AnthropicLLM()
        cfg = provider._default_config()
        assert isinstance(cfg, AnthropicGenerationConfig)
        assert cfg.model == "claude-sonnet-4-6"


class TestOpenAILLMConstruction:
    def test_construct(self):
        provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("sk-123")))
        assert provider._credentials.api_key.get_secret_value() == "sk-123"

    def test_default_config(self):
        provider = OpenAILLM()
        cfg = provider._default_config()
        assert isinstance(cfg, OpenAIGenerationConfig)
        assert cfg.model == "gpt-4.1"


class TestMistralLLMConstruction:
    def test_construct(self):
        provider = MistralLLM(MistralCredentials(api_key=SecretStr("mist-123")))
        assert provider._credentials.api_key.get_secret_value() == "mist-123"

    def test_default_config(self):
        provider = MistralLLM()
        cfg = provider._default_config()
        assert isinstance(cfg, MistralGenerationConfig)
        assert cfg.model == "mistral-medium-latest"


class TestOllamaLLMConstruction:
    def test_construct(self):
        provider = OllamaLLM(OllamaCredentials())
        assert hasattr(provider, "_credentials")

    def test_default_config(self):
        provider = OllamaLLM()
        cfg = provider._default_config()
        assert isinstance(cfg, OllamaGenerationConfig)
        assert cfg.model == "llama3.2"


class _SchemaTool(Tool):
    name = "test_tool"
    description = "A test tool"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return None


class OpenAISchemaTest:
    def test_openai_schema(self):
        provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("sk-test")))
        tool = _SchemaTool()
        schema = provider._tool_to_schema(tool)
        assert schema["name"] == "test_tool"
        assert "parameters" in schema

    def test_anthropic_schema(self):
        provider = AnthropicLLM(AnthropicCredentials(api_key=SecretStr("sk-ant-test")))
        tool = _SchemaTool()
        schema = provider._tool_to_schema(tool)
        assert schema["name"] == "test_tool"
        assert "input_schema" in schema

    def test_mistral_schema(self):
        provider = MistralLLM(MistralCredentials(api_key=SecretStr("sk-mist-test")))
        tool = _SchemaTool()
        schema = provider._tool_to_schema(tool)
        assert schema["type"] == "function"

    def test_ollama_schema(self):
        provider = OllamaLLM(OllamaCredentials())
        tool = _SchemaTool()
        schema = provider._tool_to_schema(tool)
        assert schema["type"] == "function"
