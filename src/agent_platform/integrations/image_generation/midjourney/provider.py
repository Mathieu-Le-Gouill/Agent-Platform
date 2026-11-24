from __future__ import annotations

import functools
from typing import Any

import httpx

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.core.errors import ProviderError, error_logged, require_secret
from agent_platform.core.interfaces.image_generation.base import BaseImageGenerator
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.document import DocumentMetadata, ImageDocument
from agent_platform.core.schemas.enums import ImageFormat
from agent_platform.integrations.credentials import MidjourneyCredentials
from agent_platform.integrations.image_generation.midjourney.config import (
    MidjourneyConfig,
)
from agent_platform.integrations.image_generation.midjourney.mappers import (
    size_to_aspect,
)
from agent_platform.integrations.sdk_errors import translate_sdk_error
from agent_platform.utils.env import from_env

_DEFAULT_BASE_URL = "http://localhost:8080"
_DEFAULT_TIMEOUT = 120

_ERROR_MESSAGE = "Image generation failed"

# The `/imagine` POST starts the billed job, so only a status that means the
# proxy rejected the request before working on it is retried.
_translate_imagine = functools.partial(
    translate_sdk_error,
    operation=_ERROR_MESSAGE,
    retry_statuses=frozenset({429, 503}),
)
_translate_download = functools.partial(translate_sdk_error, operation=_ERROR_MESSAGE)


class MidjourneyGenerator(BaseImageGenerator[MidjourneyConfig]):
    def __init__(
        self,
        credentials: MidjourneyCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, MidjourneyCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> MidjourneyConfig:
        return MidjourneyConfig()

    def _api_key(self) -> str:
        api_key = require_secret(
            self._credentials.api_key, "Midjourney API key is required"
        )
        return api_key.get_secret_value()

    def _base_url(self) -> str:
        return (
            self._client_options.base_url
            or from_env("MIDJOURNEY_API_URL")
            or _DEFAULT_BASE_URL
        )

    @with_retry(translate=_translate_imagine)
    async def _imagine(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST the generation request, which starts the billed job, and return its JSON."""
        response = await client.post(
            f"{self._base_url()}/imagine",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key()}"},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    @error_logged(re_raise=ProviderError, message=_ERROR_MESSAGE)
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
            "aspect_ratio": size_to_aspect(size),
            "process_mode": config.process_mode,
        }

        timeout = (
            resolve_timeout(config.timeout, self._client_options) or _DEFAULT_TIMEOUT
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            data = await self._imagine(client, payload)

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

    @error_logged(re_raise=ProviderError, message=_ERROR_MESSAGE)
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
            "aspect_ratio": size_to_aspect(size),
            "process_mode": config.process_mode,
        }

        documents: list[ImageDocument] = []
        timeout = (
            resolve_timeout(config.timeout, self._client_options) or _DEFAULT_TIMEOUT
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            data = await self._imagine(client, payload)

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

    # Retried on its own: a download is idempotent, and a failed one must never
    # send the generation request again.
    @staticmethod
    @with_retry(translate=_translate_download)
    async def _fetch_image(client: httpx.AsyncClient, url: str) -> bytes:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content
