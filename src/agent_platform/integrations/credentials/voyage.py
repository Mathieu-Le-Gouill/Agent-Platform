from pydantic import Field, SecretStr
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import secret_from_env


class VoyageCredentials(ProviderCredentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("VOYAGE_API_KEY")
    )
