from typing import Any

from adapters.llm.config import GenerationConfig, ResponseFormat


def to_openai(
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
        params["seed"] = config.seed

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
                "json_schema": {
                    "name": "response",
                    "schema": config.json_schema,
                },
            }

    return params


"""
OPENAI PARAMS:
    messages: Iterable[ChatCompletionMessageParam],
    model: ChatModel | str,
    audio: ChatCompletionAudioParam | Omit | None = omit,
    frequency_penalty: float | Omit | None = omit,
    function_call: FunctionCall | Omit = omit,
    functions: Iterable[Function] | Omit = omit,
    logit_bias: Dict[str, int] | Omit | None = omit,
    logprobs: bool | Omit | None = omit,
    max_completion_tokens: int | Omit | None = omit,
    max_tokens: int | Omit | None = omit,
    metadata: Metadata | Omit | None = omit,
    modalities: List[Literal['text', 'audio']] | Omit | None = omit,
    moderation: Moderation | Omit | None = omit,
    n: int | Omit | None = omit,
    parallel_tool_calls: bool | Omit = omit,
    prediction: ChatCompletionPredictionContentParam | Omit | None = omit,
    presence_penalty: float | Omit | None = omit,
    prompt_cache_key: str | Omit = omit,
    prompt_cache_retention: Omit | Literal['in_memory', '24h'] | None = omit,
    reasoning_effort: ReasoningEffort | Omit = omit,
    response_format: ResponseFormat | Omit = omit,
    safety_identifier: str | Omit = omit,
    seed: int | Omit | None = omit,
    service_tier: Omit | Literal['auto', 'default', 'flex', 'scale', 'priority'] | None = omit,
    stop: str | SequenceNotStr[str] | Omit | None = omit,
    store: bool | Omit | None = omit,
    stream: Omit | Literal[False] | None = omit,
    stream_options: ChatCompletionStreamOptionsParam | Omit | None = omit,
    temperature: float | Omit | None = omit,
    tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
    tools: Iterable[ChatCompletionToolUnionParam] | Omit = omit,
    top_logprobs: int | Omit | None = omit,
    top_p: float | Omit | None = omit,
    user: str | Omit = omit,
    verbosity: Omit | Literal['low', 'medium', 'high'] | None = omit,
    web_search_options: WebSearchOptions | Omit = omit,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | Timeout | NotGiven | None = not_given
"""