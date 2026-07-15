from pydantic import Field, SecretStr
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import from_env, secret_from_env


class WeaviateCredentials(ProviderCredentials, frozen=True):
    url: str = Field(
        default_factory=lambda: from_env("WEAVIATE_URL") or "http://localhost:8080"
    )
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("WEAVIATE_API_KEY")
    )
