from __future__ import annotations

import base64

from openai import AsyncOpenAI

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, error_logged, require_secret
from agent_platform.core.interfaces.image_generation.base import BaseImageGenerator
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.document import DocumentMetadata, ImageDocument
from agent_platform.core.schemas.enums import ImageFormat
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.image_generation.dalle.config import DalleConfig
from agent_platform.integrations.image_generation.dalle.mappers import (
    build_request_kwargs,
    default_size,
    validate_n,
    validate_size,
)

_MODEL_MAP: dict[str, str] = {
    "dall-e-2": "dall-e-2",
    "dall-e-3": "dall-e-3",
    "gpt-image-1": "gpt-image-1",
}


class DallEImageGenerator(BaseImageGenerator[DalleConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)

    def _default_config(self) -> DalleConfig:
        return DalleConfig()

    def _api_key(self) -> str:
        api_key = require_secret(
            self._credentials.api_key, "OpenAI API key is required"
        )
        return api_key.get_secret_value()

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
        client = AsyncOpenAI(api_key=self._api_key())

        size = size or default_size(config.model)
        validate_size(config.model, size)
        validate_n(config.model, 1)

        response = await client.images.generate(
            **build_request_kwargs(config, prompt, size, n=1)
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
        client = AsyncOpenAI(api_key=self._api_key())

        size = size or default_size(config.model)
        validate_size(config.model, size)
        validate_n(config.model, n)

        response = await client.images.generate(
            **build_request_kwargs(config, prompt, size, n=n)
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
