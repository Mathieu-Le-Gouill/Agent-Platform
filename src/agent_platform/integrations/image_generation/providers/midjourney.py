from __future__ import annotations

from math import gcd
from typing import Any

import httpx

from agent_platform.integrations.image_generation.base import BaseImageGenerator
from agent_platform.models.document import DocumentMetadata, ImageDocument
from agent_platform.models.enums import ImageFormat


class MidjourneyGenerator(BaseImageGenerator):

    def __init__(
        self,
        api_url: str,
        api_key: str,
        timeout: float = 120.0,
    ) -> None:
        
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self,
        prompt: str,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> ImageDocument:
    
        payload = {
            "prompt": prompt,
            "aspect_ratio": _size_to_aspect(size),
            "process_mode": "fast",
        }

        response = await self._client.post(
            f"{self._api_url}/imagine",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        image_url = data.get("image_url") or data.get("uri", "")
        image_bytes = await self._fetch_image(image_url)

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

    async def generate_many(
        self,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> list[ImageDocument]:
        
        payload = {
            "prompt": prompt,
            "aspect_ratio": _size_to_aspect(size),
            "process_mode": "fast",
        }

        response = await self._client.post(
            f"{self._api_url}/imagine",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        image_urls: list[str] = data.get("image_urls", [data.get("image_url", "")])
        image_urls = image_urls[:n]

        documents: list[ImageDocument] = []
        for url in image_urls:
            image_bytes = await self._fetch_image(url)
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

    async def _fetch_image(self, url: str) -> bytes:
        resp = await self._client.get(url)
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
