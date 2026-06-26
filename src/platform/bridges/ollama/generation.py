from typing import Any

from providers.llm.config import GenerationConfig, ResponseFormat


def to_ollama(
    config: GenerationConfig | None,
) -> dict[str, Any]:
    if config is None:
        return {}

    params: dict[str, Any] = {}
    options: dict[str, Any] = {}

    if config.temperature is not None:
        options["temperature"] = config.temperature

    if config.top_p is not None:
        options["top_p"] = config.top_p

    if config.seed is not None:
        options["seed"] = config.seed

    if config.stop_sequences:
        options["stop"] = config.stop_sequences

    if options:
        params["options"] = options

    match config.response_format:
        case ResponseFormat.JSON:
            params["format"] = "json"

        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )

            params["format"] = config.json_schema

    return params


"""
OLLAMA PARAMS:
    model: str = '',
    messages: Sequence[Mapping[str, Any] | Message] | None = None,
    *,
    tools: Sequence[Mapping[str, Any] | Tool | ((...) -> Unknown)] | None = None,
    stream: Literal[False] = False,
    think: bool | Literal['low', 'medium', 'high'] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    format: JsonSchemaValue | Literal['', 'json'] | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None
"""