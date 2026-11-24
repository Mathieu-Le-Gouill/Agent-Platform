from __future__ import annotations

import pytest

from agent_platform.core.errors import PlatformError, ProviderError
from agent_platform.integrations.sdk_errors import translate_sdk_error


class _Err(Exception):
    def __init__(self, **attrs):
        super().__init__("boom")
        for name, value in attrs.items():
            setattr(self, name, value)


class _Response:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


@pytest.mark.parametrize(
    "attribute", ["status_code", "status", "http_status", "http_status_code", "code"]
)
class TestStatusShapes:
    def test_a_503_is_retryable(self, attribute):
        error = translate_sdk_error(_Err(**{attribute: 503}))

        assert error.retryable is True
        assert error.code == "provider_unavailable"

    def test_a_400_is_permanent(self, attribute):
        error = translate_sdk_error(_Err(**{attribute: 400}))

        assert error.retryable is False
        assert error.code == "provider_bad_request"


class TestStatusSources:
    def test_response_status_code(self):
        error = translate_sdk_error(_Err(response=_Response(502)))

        assert error.retryable is True

    def test_botocore_response_dict(self):
        exc = _Err(response={"ResponseMetadata": {"HTTPStatusCode": 503}})

        assert translate_sdk_error(exc).retryable is True

    def test_a_non_integer_code_is_ignored(self):
        error = translate_sdk_error(_Err(code="rate_limit_exceeded"))

        assert error.retryable is False
        assert error.code is None

    def test_a_401_is_permanent(self):
        error = translate_sdk_error(_Err(status_code=401))

        assert error.retryable is False
        assert error.code == "provider_bad_request"

    def test_should_retry_flag(self):
        assert translate_sdk_error(_Err(should_retry=True)).retryable is True
        assert translate_sdk_error(_Err(should_retry=False)).retryable is False

    def test_retry_after_header(self):
        exc = _Err(response=_Response(429, {"retry-after": "7"}))

        assert translate_sdk_error(exc).retry_after == 7.0


class TestConnectionErrors:
    @pytest.mark.parametrize(
        ("module", "build"),
        [
            pytest.param(None, lambda _: TimeoutError("slow"), id="timeout"),
            pytest.param(None, lambda _: ConnectionResetError("reset"), id="reset"),
            pytest.param("httpx", lambda m: m.ConnectError("down"), id="httpx-connect"),
            pytest.param(
                "requests", lambda m: m.ConnectionError("down"), id="requests-connect"
            ),
            pytest.param(
                "requests", lambda m: m.Timeout("slow"), id="requests-timeout"
            ),
            pytest.param(
                "aiohttp",
                lambda m: m.ClientConnectionError("down"),
                id="aiohttp-connect",
            ),
            pytest.param(
                "azure.core.exceptions",
                lambda m: m.ServiceRequestError("down"),
                id="azure-request",
            ),
        ],
    )
    def test_each_connection_class_is_retryable(self, module, build):
        exc = build(pytest.importorskip(module) if module else None)

        error = translate_sdk_error(exc)

        assert error.retryable is True
        assert error.code == "provider_unavailable"

    def test_a_chained_connection_error_is_found(self):
        httpx = pytest.importorskip("httpx")
        try:
            try:
                raise httpx.ConnectError("down")
            except httpx.ConnectError as cause:
                raise RuntimeError("wrapped") from cause
        except RuntimeError as exc:
            error = translate_sdk_error(exc)

        assert error.retryable is True

    def test_a_self_referencing_chain_ends(self):
        exc = RuntimeError("loop")
        exc.__cause__ = exc

        assert translate_sdk_error(exc).retryable is False

    def test_transient_types_extend_the_rule(self):
        class Custom(Exception):
            pass

        assert translate_sdk_error(Custom()).retryable is False
        assert translate_sdk_error(Custom(), transient_types=(Custom,)).retryable


class TestRetryStatuses:
    def test_a_listed_status_is_retryable(self):
        error = translate_sdk_error(
            _Err(status_code=503), retry_statuses=frozenset({429, 503})
        )

        assert error.retryable is True

    def test_an_unlisted_transient_status_is_not(self):
        error = translate_sdk_error(
            _Err(status_code=500), retry_statuses=frozenset({429, 503})
        )

        assert error.retryable is False
        assert error.code == "provider_bad_request"

    def test_a_timeout_is_not_retryable_but_stays_unavailable(self):
        error = translate_sdk_error(
            TimeoutError("slow"), retry_statuses=frozenset({429, 503})
        )

        assert error.retryable is False
        assert error.code == "provider_unavailable"


class TestOtherErrors:
    def test_a_local_error_is_permanent_with_no_code(self):
        error = translate_sdk_error(FileNotFoundError("missing"))

        assert error.retryable is False
        assert error.code is None

    def test_a_platform_error_passes_through(self):
        original = ProviderError("x", retryable=True)

        assert translate_sdk_error(original, operation="op") is original
        assert isinstance(translate_sdk_error(original), PlatformError)

    def test_operation_prefixes_the_message(self):
        error = translate_sdk_error(_Err(status_code=400), operation="Embedding failed")

        assert str(error) == "Embedding failed: boom"
        assert str(translate_sdk_error(_Err(status_code=400))) == "boom"
