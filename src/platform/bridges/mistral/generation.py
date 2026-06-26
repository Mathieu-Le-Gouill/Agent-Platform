from typing import Any

from providers.llm.config import GenerationConfig, ResponseFormat


def to_mistral(
    config: GenerationConfig | None,
) -> dict[str, Any]:
    if config is None:
        return {}

    params: dict[str, Any] = {}

    if config.temperature is not None:
        params["temperature"] = config.temperature

    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens

    if config.top_p is not None:
        params["top_p"] = config.top_p

    if config.stop_sequences:
        params["stop"] = config.stop_sequences

    if config.seed is not None:
        params["random_seed"] = config.seed

    match config.response_format:
        case ResponseFormat.JSON:
            params["response_format"] = {
                "type": "json_object"
            }

        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )

            params["response_format"] = {
                "type": "json_schema",
                "json_schema": config.json_schema,
            }

    return params


"""
MISTRAL PARAMS:
    model: str,
    messages: List[ChatCompletionRequestMessage] | List[ChatCompletionRequestMessageTypedDict],
    temperature: OptionalNullable[float] = UNSET,
    top_p: OptionalNullable[float] = UNSET,
    max_tokens: OptionalNullable[int] = UNSET,
    stream: bool | None = False,
    stop: OptionalNullable[ChatCompletionRequestStop] = UNSET,
    random_seed: OptionalNullable[int] = UNSET,
    metadata: OptionalNullable[Dict[str, Any]] = UNSET,
    response_format: ResponseFormat | ResponseFormatTypedDict | None = None,
    tools: OptionalNullable[List[ChatCompletionRequestTool] | List[ChatCompletionRequestToolTypedDict]] = UNSET,
    tool_choice: ChatCompletionRequestToolChoice | ChatCompletionRequestToolChoiceTypedDict | None = None,
    presence_penalty: OptionalNullable[float] = UNSET,
    frequency_penalty: OptionalNullable[float] = UNSET,
    n: OptionalNullable[int] = UNSET,
    prediction: Prediction | PredictionTypedDict | None = None,
    parallel_tool_calls: bool | None = None,
    reasoning_effort: OptionalNullable[ReasoningEffort] = UNSET,
    prompt_mode: OptionalNullable[MistralPromptMode] = UNSET,
    guardrails: OptionalNullable[List[GuardrailConfig] | List[GuardrailConfigTypedDict]] = UNSET,
    prompt_cache_key: OptionalNullable[str] = UNSET,
    safe_prompt: bool | None = None,
    retries: OptionalNullable[RetryConfig] = UNSET,
    server_url: str | None = None,
    timeout_ms: int | None = None,
    http_headers: Mapping[str, str] | None = None
"""