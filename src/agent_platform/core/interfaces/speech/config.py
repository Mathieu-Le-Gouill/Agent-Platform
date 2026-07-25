from agent_platform.core.config import ProviderConfig


class SpeechConfig(ProviderConfig):
    # No sensible provider-agnostic default; every provider overrides this.
    model: str = ""
