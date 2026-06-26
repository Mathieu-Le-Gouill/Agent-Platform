from typing import Any

from adapters.llm.config import GenerationConfig, ResponseFormat


def to_anthropic(
    config: GenerationConfig | None,
) -> dict[str, Any]:

    if config is None:
        return {
            "max_tokens": 1024,
        }

    params: dict[str, Any] = {
        "max_tokens": config.max_tokens or 1024,
        "temperature": config.temperature,
    }

    if config.top_p is not None:
        params["top_p"] = config.top_p

    if config.stop_sequences:
        params["stop_sequences"] = config.stop_sequences

    if config.response_format == ResponseFormat.JSON:
        params["output_format"] = {
            "type": "json"
        }

    elif config.response_format == ResponseFormat.JSON_SCHEMA:
        if config.json_schema is None:
            raise ValueError(
                "json_schema must be provided when using JSON_SCHEMA response format"
            )

        params["output_format"] = {
            "type": "json_schema",
            "schema": config.json_schema,
        }

    return params


"""
ANTHROPIC PARAMS:
    max_tokens: int,
    messages: Iterable[MessageParam],
    model: ModelParam,
    cache_control: CacheControlEphemeralParam | Omit | None = omit,
    container: str | Omit | None = omit,
    inference_geo: str | Omit | None = omit,
    metadata: MetadataParam | Omit = omit,
    output_config: OutputConfigParam | Omit = omit,
    service_tier: Omit | Literal['auto', 'standard_only'] = omit,
    stop_sequences: SequenceNotStr[str] | Omit = omit,
    stream: Omit | Literal[False] = omit,
    system: str | Iterable[TextBlockParam] | Omit = omit,
    temperature: float | Omit = omit,
    thinking: ThinkingConfigParam | Omit = omit,
    tool_choice: ToolChoiceParam | Omit = omit,
    tools: Iterable[ToolUnionParam] | Omit = omit,
    top_k: int | Omit = omit,
    top_p: float | Omit = omit,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | Timeout | NotGiven | None = not_given
"""