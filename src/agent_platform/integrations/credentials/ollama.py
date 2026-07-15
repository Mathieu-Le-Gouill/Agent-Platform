from pydantic import Field
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import from_env


class OllamaCredentials(ProviderCredentials, frozen=True):
    base_url: str | None = Field(
        default_factory=lambda: from_env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
