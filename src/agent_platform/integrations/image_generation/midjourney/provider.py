from __future__ import annotations

from math import gcd
from typing import Any

import httpx

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import (
    ProviderError,
    error_logged,
    require_secret,
    with_retry,
)
from agent_platform.core.interfaces.image_generation.base import BaseImageGenerator
from agent_platform.core.schemas.document import DocumentMetadata, ImageDocument
from agent_platform.core.schemas.enums import ImageFormat
from agent_platform.integrations.credentials import MidjourneyCredentials
from agent_platform.integrations.image_generation.midjourney.config import (
    MidjourneyConfig,
)


class MidjourneyGenerator(BaseImageGenerator[MidjourneyConfig]):
    def __init__(self, credentials: MidjourneyCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, MidjourneyCredentials)

    def _default_config(self) -> MidjourneyConfig:
        return MidjourneyConfig()

    def _api_key(self) -> str:
        api_key = require_secret(
            self._credentials.api_key, "Midjourney API key is required"
        )
        return api_key.get_secret_value()

    @error_logged(re_raise=ProviderError, message="Image generation failed")
    @with_retry()
    async def generate(
        self,
        prompt: str,
        config: MidjourneyConfig | None = None,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> ImageDocument:
        config = config or self._default_config()

        payload = {
            "prompt": prompt,
            "aspect_ratio": _size_to_aspect(size),
            "process_mode": config.process_mode,
        }

        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"{config.api_url}/imagine",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key()}"},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            image_url = data.get("image_url") or data.get("uri")
            if not image_url:
                raise ProviderError(
                    "Midjourney proxy response missing image_url/uri",
                    code="midjourney.no_image_url",
                )
            image_bytes = await self._fetch_image(client, image_url)

        return ImageDocument(
            content=image_bytes,
            format=format,
            metadata=DocumentMetadata(
                description=prompt,
                extra={
                    "provider": "midjourney",
                    "size": size,
                    "job_id": data.get("job_id"),
                },
            ),
        )

    @error_logged(re_raise=ProviderError, message="Image generation failed")
    @with_retry()
    async def generate_many(
        self,
        prompt: str,
        n: int = 1,
        config: MidjourneyConfig | None = None,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> list[ImageDocument]:
        config = config or self._default_config()

        payload = {
            "prompt": prompt,
            "aspect_ratio": _size_to_aspect(size),
            "process_mode": config.process_mode,
        }

        documents: list[ImageDocument] = []
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"{config.api_url}/imagine",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key()}"},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            image_urls: list[str] = data.get("image_urls") or [
                data.get("image_url") or data.get("uri") or ""
            ]
            image_urls = image_urls[:n]

            if not image_urls or any(not url for url in image_urls):
                raise ProviderError(
                    "Midjourney proxy response missing image_url(s)/uri",
                    code="midjourney.no_image_url",
                )

            for url in image_urls:
                image_bytes = await self._fetch_image(client, url)
                documents.append(
                    ImageDocument(
                        content=image_bytes,
                        format=format,
                        metadata=DocumentMetadata(
                            description=prompt,
                            extra={
                                "provider": "midjourney",
                                "size": size,
                            },
                        ),
                    )
                )
        return documents

    @staticmethod
    async def _fetch_image(client: httpx.AsyncClient, url: str) -> bytes:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


# --- Utils ---


def _size_to_aspect(size: str | None) -> str:
    if size is None:
        return "1:1"
    parts = size.lower().split("x")
    if len(parts) != 2:
        return "1:1"
    try:
        w, h = int(parts[0]), int(parts[1])
        g = gcd(w, h)
        return f"{w // g}:{h // g}"
    except (ValueError, ZeroDivisionError):
        return "1:1"
