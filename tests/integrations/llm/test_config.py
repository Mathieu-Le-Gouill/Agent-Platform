from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.huggingface.config import (
    HuggingFaceGenerationConfig,
)
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig


class TestResponseFormat:
    def test_values(self):
        assert ResponseFormat.TEXT.value == "text"
        assert ResponseFormat.JSON.value == "json"
        assert ResponseFormat.JSON_SCHEMA.value == "json_schema"


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.temperature is None
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
        assert cfg.temperature is None
        assert cfg.thinking is False
        assert cfg.thinking_budget is None
        assert cfg.effort is None
        assert cfg.cache_control is False

    def test_construction(self):
        cfg = AnthropicGenerationConfig(
            thinking=True,
            thinking_budget=10000,
            cache_control=True,
            effort="high",
        )
        assert cfg.thinking is True
        assert cfg.thinking_budget == 10000
        assert cfg.cache_control is True
        assert cfg.effort == "high"


class TestOpenAIGenerationConfig:
    def test_defaults(self):
        cfg = OpenAIGenerationConfig()
        assert cfg.temperature is None
        assert cfg.reasoning_effort is None
        assert cfg.parallel_tool_calls is True
        assert cfg.strict is True

    def test_construction(self):
        cfg = OpenAIGenerationConfig(
            reasoning_effort="high",
            parallel_tool_calls=False,
            strict=False,
        )
        assert cfg.reasoning_effort == "high"
        assert cfg.parallel_tool_calls is False
        assert cfg.strict is False


class TestMistralGenerationConfig:
    def test_defaults(self):
        cfg = MistralGenerationConfig()
        assert cfg.model == "mistral-medium-latest"
        assert cfg.json_schema_name == "response"
        assert cfg.json_schema_strict is True

    def test_construction(self):
        cfg = MistralGenerationConfig(
            json_schema_name="custom",
            json_schema_strict=False,
        )
        assert cfg.json_schema_name == "custom"
        assert cfg.json_schema_strict is False


class TestHuggingFaceGenerationConfig:
    def test_defaults(self):
        cfg = HuggingFaceGenerationConfig()
        assert cfg.provider == "auto"
        assert cfg.task == "text-generation"
        assert cfg.repo_id == "deepseek-ai/DeepSeek-R1-0528"
        assert cfg.repetition_penalty is None
        assert cfg.do_sample is None
        assert cfg.typical_p is None
        assert cfg.return_full_text is None
        assert not hasattr(cfg, "device")

    def test_provider_default_is_not_corrupted(self):
        # Regression test: config.py previously had provider: str = "auto" glued
        # to an unterminated docstring via string-literal concatenation, so the
        # actual default was "auto" + ~70 lines of prose.
        cfg = HuggingFaceGenerationConfig()
        assert cfg.provider == "auto"
        assert len(cfg.provider) == len("auto")

    def test_construction(self):
        cfg = HuggingFaceGenerationConfig(
            repetition_penalty=1.1,
            do_sample=True,
            typical_p=0.9,
            return_full_text=False,
        )
        assert cfg.repetition_penalty == 1.1
        assert cfg.do_sample is True
        assert cfg.typical_p == 0.9
        assert cfg.return_full_text is False


class TestOllamaGenerationConfig:
    def test_defaults(self):
        cfg = OllamaGenerationConfig()
        assert cfg.repeat_penalty is None
        assert cfg.mirostat is None
        assert cfg.mirostat_tau is None
        assert cfg.mirostat_eta is None
        assert cfg.num_ctx is None

    def test_construction(self):
        cfg = OllamaGenerationConfig(
            repeat_penalty=1.2,
            mirostat=2,
            mirostat_tau=5.0,
            mirostat_eta=0.1,
            num_ctx=4096,
        )
        assert cfg.repeat_penalty == 1.2
        assert cfg.mirostat == 2
        assert cfg.mirostat_tau == 5.0
        assert cfg.mirostat_eta == 0.1
        assert cfg.num_ctx == 4096
