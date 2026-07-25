from __future__ import annotations

from agent_platform.core.config import ProviderConfig


class ClassificationConfig(ProviderConfig):
    multi_label: bool = False
