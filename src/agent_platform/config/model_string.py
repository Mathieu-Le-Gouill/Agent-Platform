from __future__ import annotations

from agent_platform.core.errors import ConfigError

__all__ = ["parse_model_string"]


def parse_model_string(value: str) -> tuple[str, str]:
    """Split a `"<provider>:<model>"` string into `(provider, model)`.

    `:` is the separator rather than `/` because Hugging Face model ids
    themselves contain `/` (e.g. `meta-llama/Llama-3-8B`).
    """
    provider, sep, model = value.partition(":")
    if not sep or not provider or not model:
        raise ConfigError(
            f"Invalid model string {value!r}, expected '<provider>:<model>'"
        )
    return provider, model
