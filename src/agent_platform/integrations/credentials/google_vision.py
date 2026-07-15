from pydantic import Field
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import from_env


class GoogleVisionCredentials(ProviderCredentials, frozen=True):
    credentials_path: str | None = Field(
        default_factory=lambda: from_env("GOOGLE_APPLICATION_CREDENTIALS")
    )
