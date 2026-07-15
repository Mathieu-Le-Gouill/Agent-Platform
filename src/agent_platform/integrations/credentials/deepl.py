from pydantic import Field, SecretStr
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import secret_from_env


class DeepLCredentials(ProviderCredentials, frozen=True):
    auth_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("DEEPL_AUTH_KEY")
    )
