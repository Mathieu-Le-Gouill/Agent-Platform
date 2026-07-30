from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.image_generation.config import ImageGenConfig


class MidjourneyConfig(ImageGenConfig):
    # Speed/cost tradeoff forwarded to the proxy's "processMode"-style field, matching the
    # cost tiers Midjourney's own Discord bot exposes.
    process_mode: Literal["fast", "relax", "turbo"] = "fast"


"""
sources: https://github.com/novicezk/midjourney-proxy
         https://docs.midjourney.com/hc/en-us/articles/32859204029709-Parameter-List (Fast/Relax/Turbo Mode)
"""
