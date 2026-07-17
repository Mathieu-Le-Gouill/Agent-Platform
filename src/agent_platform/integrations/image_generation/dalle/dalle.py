from __future__ import annotations

import base64

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
}

_SIZE_MAP: dict[str, tuple[str, ...]] = {
    "dall-e-2": ("256x256", "512x512", "1024x1024"),
    "dall-e-3": ("1024x1024", "1792x1024", "1024x1792"),
}


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
                f"Unsupported model: {config.model}. Use dall-e-2 or dall-e-3."
            )
        client = AsyncOpenAI(api_key=self._credentials.api_key.get_secret_value())

        size = size or _SIZE_MAP[config.model][0]
        _validate_size(config.model, size)

        response = await client.images.generate(
            model=config.model,
            prompt=prompt,
            size=size,
            quality=config.quality,
            n=1,
            response_format="b64_json",
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
                f"Unsupported model: {config.model}. Use dall-e-2 or dall-e-3."
            )
        client = AsyncOpenAI(api_key=self._credentials.api_key.get_secret_value())

        size = size or _SIZE_MAP[config.model][0]
        _validate_size(config.model, size)

        response = await client.images.generate(
            model=config.model,
            prompt=prompt,
            size=size,
            quality=config.quality,
            n=n,
            response_format="b64_json",
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
