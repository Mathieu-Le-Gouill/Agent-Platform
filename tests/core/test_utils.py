import os

from agent_platform.utils.env import from_env, secret_from_env


class TestSecretFromEnv:
    def test_single_key_found(self):
        os.environ["_TEST_SECRET"] = "my-secret"
        result = secret_from_env("_TEST_SECRET")
        assert result is not None
        assert result.get_secret_value() == "my-secret"
        del os.environ["_TEST_SECRET"]

    def test_multi_key_first_found(self):
        os.environ["_TEST_SECRET_B"] = "found-b"
        result = secret_from_env(["_TEST_SECRET_A", "_TEST_SECRET_B"])
        assert result is not None
        assert result.get_secret_value() == "found-b"
        del os.environ["_TEST_SECRET_B"]

    def test_no_key_returns_none(self):
        result = secret_from_env("_TEST_SECRET_NONEXISTENT")
        assert result is None

    def test_no_key_with_default(self):
        result = secret_from_env("_TEST_SECRET_NONEXISTENT", default="fallback")
        assert result is not None
        assert result.get_secret_value() == "fallback"


class TestFromEnv:
    def test_single_key_found(self):
        os.environ["_TEST_VAR"] = "value"
        result = from_env("_TEST_VAR")
        assert result == "value"
        del os.environ["_TEST_VAR"]

    def test_multi_key_first_found(self):
        os.environ["_TEST_VAR_B"] = "found-b"
        result = from_env(["_TEST_VAR_A", "_TEST_VAR_B"])
        assert result == "found-b"
        del os.environ["_TEST_VAR_B"]

    def test_no_key_returns_default(self):
        result = from_env("_TEST_VAR_NONEXISTENT", default="fallback")
        assert result == "fallback"

    def test_no_key_no_default(self):
        result = from_env("_TEST_VAR_NONEXISTENT")
        assert result is None
