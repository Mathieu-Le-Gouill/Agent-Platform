"""Shared harness for the table-driven retry-rule tests of remote providers.

A case names the operation, the optional SDK modules it needs, how to make
the mocked SDK call raise, and how to reach it through the provider's public
method. `retry_rule_tests(cases)` turns a case table into the three checks
every remote provider must pass: a 503 or a connection error costs 3 calls,
a 400 costs 1 and raises a non-retryable `ProviderError` naming the operation.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_platform.core.errors import ProviderError

REQUEST = httpx.Request("POST", "https://vendor.test/v1")

Patch = Callable[[Any, Exception], Any]
# Receives the test's tmp_path, for cases whose input is a file.
Call = Callable[[Path], Awaitable[object]]


@dataclass(frozen=True)
class Case:
    operation: str
    # Top-level or dotted modules the case imports; it is skipped without them.
    requires: tuple[str, ...]
    patch: Patch
    call: Call
    status_error: Callable[[int], Exception]
    connection_error: Callable[[], Exception]


def installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ImportError:
        return False


def skip_unless_installed(*modules: str) -> pytest.MarkDecorator:
    missing = [module for module in modules if not installed(module)]
    return pytest.mark.skipif(bool(missing), reason=f"{missing} not installed")


def http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=REQUEST)


def raised_from(error: Exception, cause: Exception) -> Exception:
    """Chain `error` to `cause` as `raise error from cause` does in an SDK."""
    error.__cause__ = cause
    return error


def httpx_connection() -> Exception:
    return httpx.ConnectError("connection refused", request=REQUEST)


def google_api_status(status: int) -> Exception:
    import google.api_core.exceptions

    error_class = {
        400: google.api_core.exceptions.InvalidArgument,
        503: google.api_core.exceptions.ServiceUnavailable,
    }[status]
    return error_class("vendor said no")


def requests_connection() -> Exception:
    import requests

    return requests.ConnectionError("connection refused")


def retry_rule_tests(cases: Mapping[str, Case]) -> type:
    """Build the test class checking the retry rule on every case.

    Assign the result to a `Test*` name in a test module so pytest collects it.
    """
    parametrize = pytest.mark.parametrize(
        "case",
        [
            pytest.param(case, id=name, marks=skip_unless_installed(*case.requires))
            for name, case in cases.items()
        ],
    )

    async def _expect(
        case: Case, mocker: Any, tmp_path: Path, error: Exception
    ) -> tuple[Any, ProviderError]:
        sdk_call = case.patch(mocker, error)
        with pytest.raises(ProviderError) as caught:
            await case.call(tmp_path)
        assert str(caught.value).startswith(case.operation)
        return sdk_call, caught.value

    class RetryRule:
        @parametrize
        async def test_a_503_costs_three_calls(
            self, case: Case, mocker, no_retry_sleep, tmp_path
        ):
            sdk_call, error = await _expect(
                case, mocker, tmp_path, case.status_error(503)
            )
            assert sdk_call.call_count == 3
            assert error.retryable is True

        @parametrize
        async def test_a_connection_error_costs_three_calls(
            self, case: Case, mocker, no_retry_sleep, tmp_path
        ):
            sdk_call, error = await _expect(
                case, mocker, tmp_path, case.connection_error()
            )
            assert sdk_call.call_count == 3
            assert error.retryable is True

        @parametrize
        async def test_a_400_costs_one_call_and_is_not_retryable(
            self, case: Case, mocker, no_retry_sleep, tmp_path
        ):
            sdk_call, error = await _expect(
                case, mocker, tmp_path, case.status_error(400)
            )
            assert sdk_call.call_count == 1
            assert error.retryable is False

    return RetryRule
