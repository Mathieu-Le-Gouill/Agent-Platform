import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import boto3
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from botocore.exceptions import ConnectionError as BotoConnectionError

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.core.errors import ProviderError, error_logged
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import AWSTextractCredentials
from agent_platform.integrations.ocr.aws_textract.config import AWSTextractConfig
from agent_platform.integrations.ocr.aws_textract.mappers import from_textract
from agent_platform.integrations.ocr.sources import load_bytes
from agent_platform.integrations.sdk_errors import translate_sdk_error

_ERROR_MESSAGE = "OCR extraction failed"

# Textract reports throttling as an HTTP 400 with one of these error codes, so
# the status alone would call it permanent.
_THROTTLING_CODES = frozenset(
    {"ThrottlingException", "ProvisionedThroughputExceededException"}
)


def _translate(exc: Exception) -> Exception:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code")
        if code in _THROTTLING_CODES:
            return ProviderError(
                f"{_ERROR_MESSAGE}: {exc}",
                code="provider_unavailable",
                retryable=True,
            )
    return translate_sdk_error(
        exc, operation=_ERROR_MESSAGE, transient_types=(BotoConnectionError,)
    )


class AWSTextractOCR(BaseOCR[AWSTextractConfig]):
    def __init__(
        self,
        credentials: AWSTextractCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, AWSTextractCredentials)
        self._client_options = resolve_client_options(client_options)
        self._clients: dict[tuple[str, float | None], Any] = {}

    def _default_config(self) -> AWSTextractConfig:
        return AWSTextractConfig()

    def _get_client(self, region_name: str, config: AWSTextractConfig) -> BaseClient:
        timeout = resolve_timeout(config.timeout, self._client_options)
        cache_key = (region_name, timeout)

        client = self._clients.get(cache_key)
        if client is None:
            secret_key = self._credentials.aws_secret_access_key
            kwargs: dict[str, Any] = {
                "region_name": region_name,
                "aws_access_key_id": self._credentials.aws_access_key_id,
                "aws_secret_access_key": (
                    secret_key.get_secret_value() if secret_key is not None else None
                ),
            }
            if self._client_options.base_url:
                kwargs["endpoint_url"] = self._client_options.base_url

            # No SDK retries: the platform's `with_retry` is the only retry layer.
            boto_config_kwargs: dict[str, Any] = {"retries": {"max_attempts": 0}}
            if timeout is not None:
                boto_config_kwargs["connect_timeout"] = timeout
                boto_config_kwargs["read_timeout"] = timeout
            kwargs["config"] = BotoConfig(**boto_config_kwargs)

            client = boto3.client("textract", **kwargs)
            self._clients[cache_key] = client
        return client

    @error_logged(re_raise=ProviderError, message=_ERROR_MESSAGE)
    @with_retry(translate=_translate)
    async def extract(
        self,
        source: str,
        config: AWSTextractConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        doc_id = document_id or uuid4()
        document_bytes = await asyncio.to_thread(load_bytes, source)

        client = self._get_client(config.region_name, config)

        response = await asyncio.to_thread(
            client.detect_document_text,
            Document={"Bytes": document_bytes},
        )

        return from_textract(
            response, document_id=doc_id, min_confidence=config.min_confidence
        )
