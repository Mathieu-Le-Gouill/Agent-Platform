from __future__ import annotations

from typing import Any

from agent_platform.integrations.image_generation.dalle.config import DalleConfig

__all__ = ["build_request_kwargs", "default_size", "validate_n", "validate_size"]

_SIZE_MAP: dict[str, tuple[str, ...]] = {
    "dall-e-2": ("256x256", "512x512", "1024x1024"),
    "dall-e-3": ("1024x1024", "1792x1024", "1024x1792"),
    "gpt-image-1": ("1024x1024", "1536x1024", "1024x1536", "auto"),
}

# dall-e-3 only ever returns a single image per request; dall-e-2 and
# gpt-image-1 accept a batch of up to 10: https://platform.openai.com/docs/api-reference/images/create#images-create-n
_N_LIMITS: dict[str, tuple[int, int]] = {
    "dall-e-2": (1, 10),
    "dall-e-3": (1, 1),
    "gpt-image-1": (1, 10),
}

# gpt-image-1 always returns base64 and rejects the response_format param outright;
# only dall-e-2/dall-e-3 accept it: https://platform.openai.com/docs/api-reference/images/create#images-create-response_format
_RESPONSE_FORMAT_MODELS = frozenset({"dall-e-2", "dall-e-3"})


def default_size(model: str) -> str:
    return _SIZE_MAP[model][0]


def validate_size(model: str, size: str) -> None:
    valid_sizes = _SIZE_MAP.get(model, ())
    if size not in valid_sizes:
        raise ValueError(
            f"Invalid size '{size}' for {model}. Valid sizes: {valid_sizes}"
        )


def validate_n(model: str, n: int) -> None:
    limits = _N_LIMITS.get(model)
    if limits is None:
        return
    lo, hi = limits
    if not (lo <= n <= hi):
        raise ValueError(f"Invalid n={n} for {model}. Valid range: {lo}-{hi}.")


def build_request_kwargs(
    config: DalleConfig, prompt: str, size: str, n: int
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "prompt": prompt,
        "size": size,
        "n": n,
    }
    if config.model != "dall-e-2":
        kwargs["quality"] = config.quality
    if config.model in _RESPONSE_FORMAT_MODELS:
        kwargs["response_format"] = "b64_json"
    if config.model == "dall-e-3" and config.style is not None:
        kwargs["style"] = config.style
    return kwargs
