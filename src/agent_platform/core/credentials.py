from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, SecretStr


class BaseCredentials(BaseModel, frozen=True):
    model_config = ConfigDict(extra="ignore")


class ProviderCredentials(BaseCredentials, frozen=True):
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int = 3


# --- Utils ---

_CredT = TypeVar("_CredT", bound=BaseCredentials)


def resolve_credentials(
    credentials: _CredT | None, credentials_cls: type[_CredT]
) -> _CredT:
    return credentials if credentials is not None else credentials_cls()


def resolve_timeout(
    config_timeout: float | None, credentials: ProviderCredentials
) -> float | None:
    return config_timeout if config_timeout is not None else credentials.timeout


def resolve_max_retries(
    config_max_retries: int | None, credentials: ProviderCredentials
) -> int:
    return (
        config_max_retries
        if config_max_retries is not None
        else credentials.max_retries
    )
