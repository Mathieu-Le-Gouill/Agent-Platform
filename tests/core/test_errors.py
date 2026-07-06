import pytest

from agent_platform.core.errors import (
    PlatformError,
    ProviderError,
    ConfigError,
    NotFoundError,
    ValidationError,
    LLMError,
    LLMGenerationError,
    LLMTimeoutError,
    LLMRateLimitError,
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
