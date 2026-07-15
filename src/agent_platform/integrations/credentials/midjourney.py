from pydantic import Field, SecretStr
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import secret_from_env


class MidjourneyCredentials(ProviderCredentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("MIDJOURNEY_API_KEY")
    )
