from pydantic import Field, SecretStr
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import from_env, secret_from_env


class AzureTranslatorCredentials(ProviderCredentials, frozen=True):
    api_key: SecretStr | None = Field(
        default_factory=lambda: secret_from_env("AZURE_TRANSLATOR_KEY")
    )
    region: str | None = Field(
        default_factory=lambda: from_env("AZURE_TRANSLATOR_REGION")
    )
    endpoint: str = "https://api.cognitive.microsofttranslator.com"
