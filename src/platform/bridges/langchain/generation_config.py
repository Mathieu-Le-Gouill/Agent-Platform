from typing import Any
from integrations.llm.config import AnthropicConfig, GenerationConfig, OpenAIConfig, ResponseFormat


def to_langchain_anthropic(config: GenerationConfig | None) -> dict[str, Any]:
    if config is None:
        return {"max_tokens": 1024}

    params: dict[str, Any] = {
        "max_tokens": config.max_tokens or 1024,
        "temperature": config.temperature,
    }

    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.top_k is not None:
        params["top_k"] = config.top_k
    if config.stop_sequences:
        params["stop_sequences"] = config.stop_sequences
    if config.timeout is not None:
        params["timeout"] = config.timeout
    if config.max_retries is not None:
        params["max_retries"] = config.max_retries

    if isinstance(config, AnthropicConfig):
        if config.thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.thinking_budget or 5000,
            }
        if config.cache_control:
            params["cache_control"] = {"type": "ephemeral"}

    return params


def to_langchain_openai(config: GenerationConfig | None) -> dict[str, Any]:
    if config is None:
        return {}

    params: dict[str, Any] = {
        "temperature": config.temperature,
    }

    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["seed"] = config.seed
    if config.timeout is not None:
        params["timeout"] = config.timeout
    if config.max_retries is not None:
        params["max_retries"] = config.max_retries

    model_kwargs: dict[str, Any] = {}

    if config.frequency_penalty is not None:
        model_kwargs["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        model_kwargs["presence_penalty"] = config.presence_penalty

    if isinstance(config, OpenAIConfig):
        if config.reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = config.reasoning_effort
        if not config.parallel_tool_calls:
            model_kwargs["parallel_tool_calls"] = config.parallel_tool_calls

    match config.response_format:
        case ResponseFormat.JSON:
            model_kwargs["response_format"] = {"type": "json_object"}
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError("json_schema is required for JSON_SCHEMA response format")
            model_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": config.json_schema},
            }

    if model_kwargs:
        params["model_kwargs"] = model_kwargs

    return params


def to_langchain_mistral(config: GenerationConfig | None) -> dict[str, Any]:
    if config is None:
        return {}

    params: dict[str, Any] = {
        "temperature": config.temperature,
    }

    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["random_seed"] = config.seed
    if config.timeout is not None:
        params["timeout"] = config.timeout
    if config.max_retries is not None:
        params["max_retries"] = config.max_retries

    model_kwargs: dict[str, Any] = {}

    if config.frequency_penalty is not None:
        model_kwargs["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        model_kwargs["presence_penalty"] = config.presence_penalty

    match config.response_format:
        case ResponseFormat.JSON:
            model_kwargs["response_format"] = {"type": "json_object"}
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError("json_schema is required for JSON_SCHEMA response format")
            model_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": config.json_schema,
            }

    if model_kwargs:
        params["model_kwargs"] = model_kwargs

    return params


def to_langchain_ollama(config: GenerationConfig | None) -> dict[str, Any]:
    if config is None:
        return {}

    params: dict[str, Any] = {
        "temperature": config.temperature,
    }

    if config.max_tokens is not None:
        params["num_predict"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.top_k is not None:
        params["top_k"] = config.top_k
    if config.seed is not None:
        params["seed"] = config.seed
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.frequency_penalty is not None:
        params["repeat_penalty"] = config.frequency_penalty  # Ollama's equivalent
    if config.timeout is not None:
        params["timeout"] = config.timeout

    match config.response_format:
        case ResponseFormat.JSON:
            params["format"] = "json"
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError("json_schema is required for JSON_SCHEMA response format")
            params["format"] = config.json_schema

    return params