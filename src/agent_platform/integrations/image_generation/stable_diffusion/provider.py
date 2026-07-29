from __future__ import annotations

import asyncio
import io
from typing import Any

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from PIL import Image

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.image_generation.base import BaseImageGenerator
from agent_platform.core.schemas.dimensions import Dimensions
from agent_platform.core.schemas.document import DocumentMetadata, ImageDocument
from agent_platform.core.schemas.enums import ImageFormat
from agent_platform.integrations.image_generation.stable_diffusion.config import (
    StableDiffusionConfig,
)
from agent_platform.integrations.image_generation.stable_diffusion.mappers import (
    build_generator,
    parse_size,
    pluck_images,
)


class StableDiffusionGenerator(BaseImageGenerator[StableDiffusionConfig]):
    def __init__(
        self,
        config: StableDiffusionConfig | None = None,
    ) -> None:
        self._config = config or StableDiffusionConfig()
        self._dtype = self._resolve_dtype(self._config.dtype)
        self._pipeline: StableDiffusionPipeline | None = None

    def _default_config(self) -> StableDiffusionConfig:
        return StableDiffusionConfig()

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype_str, torch.float32)

    async def _load(self) -> None:
        if self._pipeline is not None:
            return

        loop = asyncio.get_event_loop()

        def _load_pipeline() -> StableDiffusionPipeline:
            load_kwargs: dict[str, Any] = {"torch_dtype": self._dtype}
            if not self._config.safety_checker:
                load_kwargs["safety_checker"] = None
            pipe = StableDiffusionPipeline.from_pretrained(
                self._config.model, **load_kwargs
            )
            pipe = pipe.to(self._config.device)
            return pipe

        self._pipeline = await loop.run_in_executor(None, _load_pipeline)

    async def generate(
        self,
        prompt: str,
        config: StableDiffusionConfig | None = None,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
        negative_prompt: str | None = None,
    ) -> ImageDocument:
        config = config or self._config
        await self._load()
        pipeline = self._pipeline

        if pipeline is None:
            raise ProviderError(
                "Failed to load Stable Diffusion pipeline",
                code="sd.pipeline_not_loaded",
            )

        height, width = parse_size(size)
        negative_prompt = (
            negative_prompt if negative_prompt is not None else config.negative_prompt
        )
        generator = build_generator(config)
        loop = asyncio.get_event_loop()

        def _infer() -> Image.Image:
            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_images_per_prompt": 1,
                "guidance_scale": config.guidance_scale,
                "num_inference_steps": config.num_inference_steps,
            }
            if negative_prompt:
                call_kwargs["negative_prompt"] = negative_prompt
            if generator is not None:
                call_kwargs["generator"] = generator
            with torch.no_grad():
                output = pipeline(**call_kwargs)
            return pluck_images(output)[0]

        pil_image = await loop.run_in_executor(None, _infer)

        buf = io.BytesIO()
        pil_image.save(buf, format=format.value.upper())
        image_bytes = buf.getvalue()

        return ImageDocument(
            content=image_bytes,
            format=format,
            dimensions=Dimensions(width=pil_image.width, height=pil_image.height),
            metadata=DocumentMetadata(
                description=prompt,
                extra={
                    "provider": "stable_diffusion",
                    "model": config.model,
                },
            ),
        )

    async def generate_many(
        self,
        prompt: str,
        n: int = 1,
        config: StableDiffusionConfig | None = None,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
        negative_prompt: str | None = None,
    ) -> list[ImageDocument]:
        config = config or self._config
        await self._load()
        pipeline = self._pipeline

        if pipeline is None:
            raise ProviderError(
                "Failed to load Stable Diffusion pipeline",
                code="sd.pipeline_not_loaded",
            )

        height, width = parse_size(size)
        negative_prompt = (
            negative_prompt if negative_prompt is not None else config.negative_prompt
        )
        generator = build_generator(config)
        loop = asyncio.get_event_loop()

        def _infer() -> list[bytes]:
            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_images_per_prompt": n,
                "guidance_scale": config.guidance_scale,
                "num_inference_steps": config.num_inference_steps,
            }
            if negative_prompt:
                call_kwargs["negative_prompt"] = negative_prompt
            if generator is not None:
                call_kwargs["generator"] = generator
            with torch.no_grad():
                output = pipeline(**call_kwargs)

            documents: list[bytes] = []
            for img in pluck_images(output):
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
                            "model": config.model,
                        },
                    ),
                )
            )
        return docs
