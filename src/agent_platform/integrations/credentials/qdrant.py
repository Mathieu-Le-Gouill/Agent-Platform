from pydantic import Field, SecretStr
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import secret_from_env


class QdrantCredentials(ProviderCredentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("QDRANT_API_KEY")
    )
