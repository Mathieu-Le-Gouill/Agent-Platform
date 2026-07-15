from pydantic import Field
from agent_platform.core.credentials import ProviderCredentials
from agent_platform.utils.utils import from_env


class AWSTextractCredentials(ProviderCredentials, frozen=True):
    aws_access_key_id: str | None = Field(
        default_factory=lambda: from_env("AWS_ACCESS_KEY_ID")
    )
    aws_secret_access_key: str | None = Field(
        default_factory=lambda: from_env("AWS_SECRET_ACCESS_KEY")
    )
