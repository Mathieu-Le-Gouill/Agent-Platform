from __future__ import annotations

import hmac
from collections.abc import Callable, Sequence
from hashlib import sha256
from typing import Annotated

from fastapi import Header, HTTPException
from pydantic import SecretStr

ANONYMOUS_KEY_ID = "anonymous"

_BEARER = "bearer"


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def build_api_key_auth(
    api_keys: Sequence[SecretStr], *, disabled: bool = False
) -> Callable[[str | None], str]:
    """Returns a FastAPI dependency that authenticates a bearer key.

    The dependency returns the caller's key id: the SHA-256 hex digest of the
    matching key, so the raw key is never stored or logged. It raises 401 with
    `WWW-Authenticate: Bearer` on a missing header, another scheme or an
    unknown key. With `disabled`, it returns `ANONYMOUS_KEY_ID` and never reads
    the header.

    The presented key is compared against every configured digest with
    `hmac.compare_digest` and no early exit, so the response time does not
    reveal which key matched or how long a matching prefix is.
    """
    digests = [sha256(key.get_secret_value().encode()).digest() for key in api_keys]

    def authenticate(
        authorization: Annotated[str | None, Header()] = None,
    ) -> str:
        if disabled:
            return ANONYMOUS_KEY_ID
        if authorization is None:
            raise _unauthorized()
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() != _BEARER:
            raise _unauthorized()
        # Starlette decodes header bytes as latin-1; re-encoding recovers the
        # raw bytes the client sent, which match a UTF-8 configured key.
        presented = sha256(credentials.encode("latin-1")).digest()
        matched = False
        for digest in digests:
            matched |= hmac.compare_digest(presented, digest)
        if not matched:
            raise _unauthorized()
        return presented.hex()

    return authenticate


__all__ = ["ANONYMOUS_KEY_ID", "build_api_key_auth"]
