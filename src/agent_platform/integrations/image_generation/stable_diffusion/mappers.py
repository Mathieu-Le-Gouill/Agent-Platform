from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from agent_platform.integrations.image_generation.stable_diffusion.config import (
    StableDiffusionConfig,
)

__all__ = ["build_generator", "parse_size", "pluck_images"]


def build_generator(config: StableDiffusionConfig) -> torch.Generator | None:
    if config.seed is None:
        return None
    return torch.Generator(device=config.device).manual_seed(config.seed)


def parse_size(size: str | None) -> tuple[int, int]:
    if size is None:
        return 512, 512
    parts = size.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size '{size}'. Expected format 'WIDTHxHEIGHT'.")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid size '{size}'. Expected format 'WIDTHxHEIGHT'."
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size '{size}'. Width and height must be positive.")
    return width, height


def pluck_images(output: Any) -> list[Image.Image]:
    if isinstance(output, tuple):
        return list(output[0])
    return list(output.images)
