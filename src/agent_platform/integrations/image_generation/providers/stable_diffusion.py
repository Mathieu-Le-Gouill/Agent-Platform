from __future__ import annotations

import asyncio
import io
from typing import Any

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from agent_platform.integrations.image_generation.base import BaseImageGenerator
from agent_platform.models.document import DocumentMetadata, ImageDocument
from agent_platform.models.enums import ImageFormat


class StableDiffusionGenerator(BaseImageGenerator):
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        safety_checker: bool = True,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._pipeline: StableDiffusionPipeline | None = None
        self._safety_checker = safety_checker

    async def _load(self) -> None:
        if self._pipeline is not None:
            return

        loop = asyncio.get_event_loop()

        def _load_pipeline() -> StableDiffusionPipeline:
            load_kwargs: dict[str, Any] = {"torch_dtype": self._dtype}
            if not self._safety_checker:
                load_kwargs["safety_checker"] = None
            pipe = StableDiffusionPipeline.from_pretrained(
                self._model_id, **load_kwargs
            )
            pipe = pipe.to(self._device)
            return pipe

        self._pipeline = await loop.run_in_executor(None, _load_pipeline)

    async def generate(
        self,
        prompt: str,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> ImageDocument:

        await self._load()
        pipeline = self._pipeline

        if pipeline is None:
            raise RuntimeError("Failed to load Stable Diffusion pipeline")

        height, width = _parse_size(size)
        loop = asyncio.get_event_loop()

        def _infer() -> Image.Image:
            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_images_per_prompt=1,
                )
            return _pluck_images(output)[0]

        pil_image = await loop.run_in_executor(None, _infer)

        buf = io.BytesIO()
        pil_image.save(buf, format=format.value.upper())
        image_bytes = buf.getvalue()

        return ImageDocument(
            content=image_bytes,
            format=format,
            width=pil_image.width,
            height=pil_image.height,
            metadata=DocumentMetadata(
                description=prompt,
                extra={
                    "provider": "stable_diffusion",
                    "model": self._model_id,
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
        await self._load()
        pipeline = self._pipeline

        if pipeline is None:
            raise RuntimeError("Failed to load Stable Diffusion pipeline")

        height, width = _parse_size(size)
        loop = asyncio.get_event_loop()

        def _infer() -> list[bytes]:
            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_images_per_prompt=n,
                )

            documents: list[bytes] = []
            for img in _pluck_images(output):
                buf = io.BytesIO()
                img.save(buf, format=format.value.upper())
                documents.append(buf.getvalue())
            return documents

        image_bytes_list = await loop.run_in_executor(None, _infer)

        docs: list[ImageDocument] = []
        for img_bytes in image_bytes_list:
            docs.append(
                ImageDocument(
                    content=img_bytes,
                    format=format,
                    metadata=DocumentMetadata(
                        description=prompt,
                        extra={
                            "provider": "stable_diffusion",
                            "model": self._model_id,
                        },
                    ),
                )
            )
        return docs


def _pluck_images(
    output: Any,
) -> list[Image.Image]:
    if isinstance(output, tuple):
        return list(output[0])
    return list(output.images)


def _parse_size(size: str | None) -> tuple[int, int]:
    if size is None:
        return 512, 512
    parts = size.lower().split("x")
    if len(parts) != 2:
        return 512, 512
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return 512, 512
