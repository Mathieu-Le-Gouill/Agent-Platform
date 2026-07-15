from pydantic import Field, SecretStr
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import secret_from_env, from_env


class OpenAICredentials(ProviderCredentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("OPENAI_API_KEY")
    )
    organization: str | None = Field(
        default_factory=lambda: from_env(["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"])
    )
