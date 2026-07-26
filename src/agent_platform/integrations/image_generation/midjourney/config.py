from __future__ import annotations

from typing import Literal

from pydantic import Field

from agent_platform.core.interfaces.image_generation.config import ImageGenConfig


class MidjourneyConfig(ImageGenConfig):
    # Base URL of an unofficial self-hosted Midjourney proxy (e.g. a midjourney-proxy-style wrapper
    # around the Discord bot) — there is no official Midjourney API. Field shape is a common community
    # convention, not a documented contract.
    api_url: str = Field(default="http://localhost:8080", alias="MIDJOURNEY_API_URL")
    # HTTP timeout in seconds for calls to the proxy above.
    timeout: int = 120
    # Speed/cost tradeoff forwarded to the proxy's "processMode"-style field, matching the
    # cost tiers Midjourney's own Discord bot exposes.
    process_mode: Literal["fast", "relax", "turbo"] = "fast"


"""
sources: https://github.com/novicezk/midjourney-proxy
         https://docs.midjourney.com/hc/en-us/articles/32859204029709-Parameter-List (Fast/Relax/Turbo Mode)
"""
