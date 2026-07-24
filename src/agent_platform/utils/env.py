import os
from collections.abc import Iterable

from pydantic import SecretStr


def secret_from_env(
    keys: str | Iterable[str], default: str | None = None
) -> SecretStr | None:

    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        value = os.environ.get(key)
        if value is not None:
            return SecretStr(value)

    return SecretStr(default) if default is not None else None


def from_env(keys: str | Iterable[str], default: str | None = None) -> str | None:

    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        value = os.environ.get(key)
        if value is not None:
            return value

    return default
