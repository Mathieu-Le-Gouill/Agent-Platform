from __future__ import annotations

import base64
from typing import Any

from openai import AsyncOpenAI

from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.core.interfaces.image_generation.base import BaseImageGenerator
from agent_platform.integrations.image_generation.dalle.config import DalleConfig
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.schemas.document import DocumentMetadata, ImageDocument
from agent_platform.core.schemas.enums import ImageFormat


_MODEL_MAP: dict[str, str] = {
    "dall-e-2": "dall-e-2",
    "dall-e-3": "dall-e-3",
    "gpt-image-1": "gpt-image-1",
}

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


class DallEImageGenerator(BaseImageGenerator[DalleConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else OpenAICredentials()
        )

    def _default_config(self) -> DalleConfig:
        return DalleConfig()

    @error_logged(re_raise=ProviderError, message="Image generation failed")
    @with_retry()
    async def generate(
        self,
        prompt: str,
        config: DalleConfig | None = None,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> ImageDocument:
        config = config or self._default_config()
        if config.model not in _MODEL_MAP:
            raise ValueError(
                f"Unsupported model: {config.model}. Use dall-e-2, dall-e-3, or gpt-image-1."
            )
        client = AsyncOpenAI(api_key=self._credentials.api_key.get_secret_value())

        size = size or _SIZE_MAP[config.model][0]
        _validate_size(config.model, size)
        _validate_n(config.model, 1)

        response = await client.images.generate(
            **_build_request_kwargs(config, prompt, size, n=1)
        )

        if response.data is None:
            raise ProviderError(
                "OpenAI returned no image data", code="dalle.empty_response"
            )

        data = response.data[0]
        if data.b64_json is None:
            raise ProviderError(
                "OpenAI returned no b64_json in image data", code="dalle.no_b64"
            )
        image_bytes = base64.b64decode(data.b64_json)

        return ImageDocument(
            content=image_bytes,
            format=format,
            metadata=DocumentMetadata(
                description=prompt,
                extra={
                    "provider": "openai",
                    "model": config.model,
                    "size": size,
                    "revised_prompt": data.revised_prompt,
                },
            ),
        )

    @error_logged(re_raise=ProviderError, message="Image generation failed")
    @with_retry()
    async def generate_many(
        self,
        prompt: str,
        n: int = 1,
        config: DalleConfig | None = None,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> list[ImageDocument]:
        config = config or self._default_config()
        if config.model not in _MODEL_MAP:
            raise ValueError(
                f"Unsupported model: {config.model}. Use dall-e-2, dall-e-3, or gpt-image-1."
            )
        client = AsyncOpenAI(api_key=self._credentials.api_key.get_secret_value())

        size = size or _SIZE_MAP[config.model][0]
        _validate_size(config.model, size)
        _validate_n(config.model, n)

        response = await client.images.generate(
            **_build_request_kwargs(config, prompt, size, n=n)
        )

        if response.data is None:
            raise ProviderError(
                "OpenAI returned no image data", code="dalle.empty_response"
            )

        documents: list[ImageDocument] = []
        for data in response.data:
            if data.b64_json is None:
                continue
            image_bytes = base64.b64decode(data.b64_json)
            documents.append(
                ImageDocument(
                    content=image_bytes,
                    format=format,
                    metadata=DocumentMetadata(
                        description=prompt,
                        extra={
                            "provider": "openai",
                            "model": config.model,
                            "size": size,
                            "revised_prompt": data.revised_prompt,
                        },
                    ),
                )
            )

        return documents


# --- Utils ---


def _validate_size(model: str, size: str) -> None:
    valid_sizes = _SIZE_MAP.get(model, ())
    if size not in valid_sizes:
        raise ValueError(
            f"Invalid size '{size}' for {model}. Valid sizes: {valid_sizes}"
        )


def _validate_n(model: str, n: int) -> None:
    limits = _N_LIMITS.get(model)
    if limits is None:
        return
    lo, hi = limits
    if not (lo <= n <= hi):
        raise ValueError(f"Invalid n={n} for {model}. Valid range: {lo}-{hi}.")


def _build_request_kwargs(
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
