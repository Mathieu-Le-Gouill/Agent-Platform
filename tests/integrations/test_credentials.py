from pydantic import SecretStr

from agent_platform.core.credentials import (
    BaseCredentials,
    ProviderCredentials,
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.integrations.credentials import (
    AnthropicCredentials,
    AWSTextractCredentials,
    CohereCredentials,
    DeepgramCredentials,
    DeepLCredentials,
    GoogleVisionCredentials,
    HuggingFaceCredentials,
    JinaCredentials,
    MidjourneyCredentials,
    MistralCredentials,
    OllamaCredentials,
    OpenAICredentials,
    PicoVoiceCredentials,
    QdrantCredentials,
)


class TestBaseCredentials:
    def test_can_be_instantiated(self):
        creds = BaseCredentials()
        assert isinstance(creds, BaseCredentials)

    def test_ignores_extra_fields(self):
        creds = BaseCredentials(unknown_field="value", another="test")
        assert isinstance(creds, BaseCredentials)


class TestProviderCredentials:
    def test_defaults(self):
        creds = ProviderCredentials()
        assert creds.api_key is None
        assert creds.base_url is None
        assert creds.timeout is None
        assert creds.max_retries == 3

    def test_custom_values(self):
        creds = ProviderCredentials(
            api_key="sk-123",
            base_url="https://api.example.com",
            timeout=30.0,
            max_retries=5,
        )
        assert creds.api_key.get_secret_value() == "sk-123"
        assert creds.base_url == "https://api.example.com"
        assert creds.timeout == 30.0
        assert creds.max_retries == 5

    def test_api_key_is_secret_str_type(self):
        creds = ProviderCredentials(api_key="my-key")
        assert isinstance(creds.api_key, SecretStr)


class TestOpenAICredentials:
    def test_can_be_instantiated_without_api_key(self):
        creds = OpenAICredentials()
        assert isinstance(creds, ProviderCredentials)

    def test_custom_api_key(self):
        creds = OpenAICredentials(api_key="custom-key")
        assert creds.api_key.get_secret_value() == "custom-key"

    def test_organization(self):
        creds = OpenAICredentials(organization="my-org")
        assert creds.organization == "my-org"


class TestMistralCredentials:
    def test_can_be_instantiated(self):
        creds = MistralCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestOllamaCredentials:
    def test_can_be_instantiated(self):
        creds = OllamaCredentials()
        assert isinstance(creds, ProviderCredentials)

    def test_default_base_url(self):
        creds = OllamaCredentials()
        assert creds.base_url == "http://localhost:11434"


class TestAnthropicCredentials:
    def test_can_be_instantiated(self):
        creds = AnthropicCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestHuggingFaceCredentials:
    def test_can_be_instantiated(self):
        creds = HuggingFaceCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestDeepgramCredentials:
    def test_can_be_instantiated(self):
        creds = DeepgramCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestDeepLCredentials:
    def test_can_be_instantiated(self):
        creds = DeepLCredentials()
        assert isinstance(creds, ProviderCredentials)

    def test_auth_key_field(self):
        creds = DeepLCredentials(auth_key="deepL-key")
        assert creds.auth_key.get_secret_value() == "deepL-key"


class TestCohereCredentials:
    def test_can_be_instantiated(self):
        creds = CohereCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestJinaCredentials:
    def test_can_be_instantiated(self):
        creds = JinaCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestMidjourneyCredentials:
    def test_can_be_instantiated(self):
        creds = MidjourneyCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestQdrantCredentials:
    def test_can_be_instantiated(self):
        creds = QdrantCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestPicoVoiceCredentials:
    def test_can_be_instantiated(self):
        creds = PicoVoiceCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestGoogleVisionCredentials:
    def test_can_be_instantiated(self):
        creds = GoogleVisionCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestAWSTextractCredentials:
    def test_can_be_instantiated(self):
        creds = AWSTextractCredentials()
        assert isinstance(creds, ProviderCredentials)


class TestResolveCredentials:
    def test_returns_given_credentials_when_not_none(self):
        creds = OpenAICredentials(api_key=SecretStr("key"))
        result = resolve_credentials(creds, OpenAICredentials)
        assert result is creds

    def test_builds_default_when_none(self):
        result = resolve_credentials(None, OpenAICredentials)
        assert isinstance(result, OpenAICredentials)
        assert result.api_key is None


class TestResolveTimeout:
    def test_returns_config_timeout_when_not_none(self):
        result = resolve_timeout(30.0, ProviderCredentials(timeout=60.0))
        assert result == 30.0

    def test_falls_back_to_credentials_timeout_when_config_timeout_is_none(self):
        result = resolve_timeout(None, ProviderCredentials(timeout=60.0))
        assert result == 60.0

    def test_returns_none_when_both_are_none(self):
        result = resolve_timeout(None, ProviderCredentials())
        assert result is None


class TestResolveMaxRetries:
    def test_returns_config_max_retries_when_not_none(self):
        result = resolve_max_retries(5, ProviderCredentials(max_retries=3))
        assert result == 5

    def test_falls_back_to_credentials_max_retries_when_config_max_retries_is_none(
        self,
    ):
        result = resolve_max_retries(None, ProviderCredentials(max_retries=7))
        assert result == 7

    def test_returns_3_when_both_are_none(self):
        result = resolve_max_retries(None, ProviderCredentials())
        assert result == 3
