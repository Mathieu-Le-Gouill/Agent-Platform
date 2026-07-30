from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, SecretStr


class Credentials(BaseModel, frozen=True):
    model_config = ConfigDict(extra="ignore")

    api_key: SecretStr | None = None


class ClientOptions(BaseModel, frozen=True):
    model_config = ConfigDict(extra="ignore")

    base_url: str | None = None
    timeout: float | None = None
    max_retries: int = 3


# --- Utils ---

_CredT = TypeVar("_CredT", bound=Credentials)


def resolve_credentials(
    credentials: _CredT | None, credentials_cls: type[_CredT]
) -> _CredT:
    return credentials if credentials is not None else credentials_cls()


def resolve_client_options(client_options: ClientOptions | None) -> ClientOptions:
    return client_options if client_options is not None else ClientOptions()


def resolve_timeout(
    config_timeout: float | None, client_options: ClientOptions
) -> float | None:
    return config_timeout if config_timeout is not None else client_options.timeout


def resolve_max_retries(
    config_max_retries: int | None, client_options: ClientOptions
) -> int:
    return (
        config_max_retries
        if config_max_retries is not None
        else client_options.max_retries
    )
