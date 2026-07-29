import logging

import pytest
from pydantic import SecretStr

from agent_platform.agents.errors import (
    AgentActError,
    AgentError,
    AgentMaxIterations,
    AgentThinkError,
)
from agent_platform.agents.tools.errors import ToolError
from agent_platform.core.errors import (
    ConfigError,
    MissingCredentialError,
    NotFoundError,
    PlatformError,
    ProviderError,
    ValidationError,
    catch_noraise,
    error_logged,
    require_secret,
)
from agent_platform.core.interfaces.llm.errors import (
    LLMError,
    LLMGenerationError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class TestErrorHierarchy:
    def test_platform_error_is_base(self):
        assert issubclass(ProviderError, PlatformError)
        assert issubclass(ConfigError, PlatformError)
        assert issubclass(NotFoundError, PlatformError)
        assert issubclass(ValidationError, PlatformError)
        assert issubclass(LLMError, PlatformError)

    def test_llm_error_hierarchy(self):
        assert issubclass(LLMGenerationError, LLMError)
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)

    def test_isinstance_checks(self):
        assert isinstance(ProviderError(), PlatformError)
        assert isinstance(ConfigError(), PlatformError)
        assert isinstance(NotFoundError(), PlatformError)
        assert isinstance(ValidationError(), PlatformError)

    def test_isinstance_llm_chain(self):
        exc = LLMGenerationError()
        assert isinstance(exc, LLMError)
        assert isinstance(exc, PlatformError)
        assert isinstance(exc, Exception)

    def test_isinstance_timeout_chain(self):
        exc = LLMTimeoutError()
        assert isinstance(exc, LLMError)
        assert isinstance(exc, PlatformError)

    def test_isinstance_rate_limit_chain(self):
        exc = LLMRateLimitError()
        assert isinstance(exc, LLMError)
        assert isinstance(exc, PlatformError)

    def test_provider_error_not_llm(self):
        assert not issubclass(ProviderError, LLMError)

    def test_platform_error_is_exception(self):
        assert issubclass(PlatformError, Exception)

    def test_errors_are_distinct(self):
        assert ProviderError is not ConfigError
        assert ConfigError is not NotFoundError
        assert NotFoundError is not ValidationError

    def test_llm_errors_are_distinct(self):
        assert LLMGenerationError is not LLMTimeoutError
        assert LLMTimeoutError is not LLMRateLimitError

    def test_can_raise_and_catch_platform_error(self):
        with pytest.raises(PlatformError):
            raise ProviderError("test")

    def test_can_raise_and_catch_llm_generation(self):
        with pytest.raises(LLMError):
            raise LLMGenerationError("gen failed")

    def test_can_raise_and_catch_specific_llm(self):
        with pytest.raises(LLMTimeoutError):
            raise LLMTimeoutError("timeout")

    def test_agent_error_hierarchy(self):
        assert issubclass(AgentThinkError, AgentError)
        assert issubclass(AgentActError, AgentError)
        assert issubclass(AgentMaxIterations, AgentError)
        assert issubclass(AgentError, PlatformError)

    def test_tool_error_hierarchy(self):
        assert issubclass(ToolError, PlatformError)

    def test_missing_credential_error(self):
        assert issubclass(MissingCredentialError, PlatformError)


class TestRequireSecret:
    def test_returns_secret_when_present(self):
        secret = SecretStr("value")
        assert require_secret(secret, "missing") is secret

    def test_raises_missing_credential_error_when_none(self):
        with pytest.raises(MissingCredentialError, match="missing"):
            require_secret(None, "missing")


class TestStructuredFields:
    def test_defaults(self):
        exc = PlatformError("fail")
        assert exc.code is None
        assert exc.retryable is False
        assert exc.context == {}

    def test_code_field(self):
        exc = PlatformError("fail", code="test.code")
        assert exc.code == "test.code"

    def test_retryable_field(self):
        exc = PlatformError("fail", retryable=True)
        assert exc.retryable is True

    def test_context_field(self):
        exc = PlatformError("fail", context={"key": "val"})
        assert exc.context == {"key": "val"}

    def test_subclass_inherits_fields(self):
        exc = ProviderError("fail", code="provider.err", retryable=True)
        assert exc.code == "provider.err"
        assert exc.retryable is True

    def test_tool_error_with_fields(self):
        exc = ToolError("tool fail", code="tool.err", retryable=True)
        assert exc.code == "tool.err"
        assert exc.retryable is True


class TestErrorLoggedDecorator:
    def test_re_raise_wraps_to_domain(self):
        @error_logged(re_raise=LLMError, message="LLM call failed")
        async def failing_func():
            raise ValueError("underlying error")

        with pytest.raises(LLMError, match="LLM call failed"):
            import asyncio

            asyncio.run(failing_func())

    def test_platform_error_passthrough(self):
        @error_logged(re_raise=LLMError)
        async def raises_platform():
            raise ProviderError("already a platform error")

        with pytest.raises(ProviderError):
            import asyncio

            asyncio.run(raises_platform())

    def test_sync_wrapper(self):
        @error_logged(re_raise=LLMError, message="sync fail")
        def failing_sync():
            raise ValueError("oops")

        with pytest.raises(LLMError, match="sync fail"):
            failing_sync()


class TestErrorLoggedNoReRaise:
    @pytest.mark.asyncio
    async def test_async_success_passthrough(self):
        @error_logged()
        async def succeeds():
            return "ok"

        assert await succeeds() == "ok"

    @pytest.mark.asyncio
    async def test_async_platform_error_passthrough(self):
        @error_logged()
        async def raises_platform():
            raise ProviderError("already a platform error")

        with pytest.raises(ProviderError):
            await raises_platform()

    @pytest.mark.asyncio
    async def test_async_raw_exception_reraised_unwrapped(self):
        @error_logged()
        async def raises_raw():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await raises_raw()

    def test_sync_success_passthrough(self):
        @error_logged()
        def succeeds():
            return "ok"

        assert succeeds() == "ok"

    def test_sync_platform_error_passthrough(self):
        @error_logged()
        def raises_platform():
            raise ConfigError("already a platform error")

        with pytest.raises(ConfigError):
            raises_platform()

    def test_sync_raw_exception_reraised_unwrapped(self):
        @error_logged()
        def raises_raw():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            raises_raw()


class TestCatchNoraise:
    def test_passthrough_on_success(self):
        logger = logging.getLogger("test")
        with catch_noraise(logger, fallback=42, context_msg="parse"):
            pass

    def test_does_not_leak_exception(self):
        logger = logging.getLogger("test")
        with catch_noraise(logger, fallback=42, context_msg="parse"):
            raise ValueError("bad")
