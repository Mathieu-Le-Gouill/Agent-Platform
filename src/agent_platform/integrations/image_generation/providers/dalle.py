from __future__ import annotations

import base64
from typing import Literal

from openai import AsyncOpenAI

from agent_platform.integrations.image_generation.base import BaseImageGenerator
from agent_platform.models.document import DocumentMetadata, ImageDocument
from agent_platform.models.enums import ImageFormat


_MODEL_MAP: dict[str, str] = {
    "dall-e-2": "dall-e-2",
    "dall-e-3": "dall-e-3",
}

_SIZE_MAP: dict[str, tuple[str, ...]] = {
    "dall-e-2": ("256x256", "512x512", "1024x1024"),
    "dall-e-3": ("1024x1024", "1792x1024", "1024x1792"),
}


class DallEImageGenerator(BaseImageGenerator):
    def __init__(
        self,
        api_key: str,
        model: str = "dall-e-3",
        quality: Literal["standard", "hd"] = "standard",
    ) -> None:

        if model not in _MODEL_MAP:
            raise ValueError(f"Unsupported model: {model}. Use dall-e-2 or dall-e-3.")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._quality: Literal["standard", "hd"] = quality

    async def generate(
        self,
        prompt: str,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> ImageDocument:

        size = size or _SIZE_MAP[self._model][0]
        _validate_size(self._model, size)

        response = await self._client.images.generate(
            model=self._model,
            prompt=prompt,
            size=size,
            quality=self._quality,
            n=1,
            response_format="b64_json",
        )

        if response.data is None:
            raise RuntimeError("OpenAI returned no image data")

        data = response.data[0]
        if data.b64_json is None:
            raise RuntimeError("OpenAI returned no b64_json in image data")
        image_bytes = base64.b64decode(data.b64_json)

        return ImageDocument(
            content=image_bytes,
            format=format,
            metadata=DocumentMetadata(
                description=prompt,
                extra={
                    "provider": "openai",
                    "model": self._model,
                    "size": size,
                    "revised_prompt": data.revised_prompt,
                },
            ),
        )

    async def generate_many(
        self,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> list[ImageDocument]:

        size = size or _SIZE_MAP[self._model][0]
        _validate_size(self._model, size)

        response = await self._client.images.generate(
            model=self._model,
            prompt=prompt,
            size=size,
            quality=self._quality,
            n=n,
            response_format="b64_json",
        )

        if response.data is None:
            raise RuntimeError("OpenAI returned no image data")

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
                            "model": self._model,
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
