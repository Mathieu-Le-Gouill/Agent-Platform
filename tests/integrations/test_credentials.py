from pydantic import SecretStr

from agent_platform.core.credentials import (
    ClientOptions,
    Credentials,
    resolve_client_options,
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


class TestCredentials:
    def test_can_be_instantiated(self):
        creds = Credentials()
        assert isinstance(creds, Credentials)
        assert creds.api_key is None

    def test_ignores_extra_fields(self):
        creds = Credentials(unknown_field="value", another="test")
        assert isinstance(creds, Credentials)

    def test_api_key_is_secret_str_type(self):
        creds = Credentials(api_key="my-key")
        assert isinstance(creds.api_key, SecretStr)


class TestClientOptions:
    def test_defaults(self):
        options = ClientOptions()
        assert options.base_url is None
        assert options.timeout is None
        assert options.max_retries == 3

    def test_custom_values(self):
        options = ClientOptions(
            base_url="https://api.example.com",
            timeout=30.0,
            max_retries=5,
        )
        assert options.base_url == "https://api.example.com"
        assert options.timeout == 30.0
        assert options.max_retries == 5


class TestOpenAICredentials:
    def test_can_be_instantiated_without_api_key(self):
        creds = OpenAICredentials()
        assert isinstance(creds, Credentials)

    def test_custom_api_key(self):
        creds = OpenAICredentials(api_key="custom-key")
        assert creds.api_key.get_secret_value() == "custom-key"

    def test_organization(self):
        creds = OpenAICredentials(organization="my-org")
        assert creds.organization == "my-org"


class TestMistralCredentials:
    def test_can_be_instantiated(self):
        creds = MistralCredentials()
        assert isinstance(creds, Credentials)


class TestOllamaCredentials:
    def test_can_be_instantiated(self):
        creds = OllamaCredentials()
        assert isinstance(creds, Credentials)


class TestAnthropicCredentials:
    def test_can_be_instantiated(self):
        creds = AnthropicCredentials()
        assert isinstance(creds, Credentials)


class TestHuggingFaceCredentials:
    def test_can_be_instantiated(self):
        creds = HuggingFaceCredentials()
        assert isinstance(creds, Credentials)


class TestDeepgramCredentials:
    def test_can_be_instantiated(self):
        creds = DeepgramCredentials()
        assert isinstance(creds, Credentials)


class TestDeepLCredentials:
    def test_can_be_instantiated(self):
        creds = DeepLCredentials()
        assert isinstance(creds, Credentials)

    def test_auth_key_field(self):
        creds = DeepLCredentials(auth_key="deepL-key")
        assert creds.auth_key.get_secret_value() == "deepL-key"


class TestCohereCredentials:
    def test_can_be_instantiated(self):
        creds = CohereCredentials()
        assert isinstance(creds, Credentials)


class TestJinaCredentials:
    def test_can_be_instantiated(self):
        creds = JinaCredentials()
        assert isinstance(creds, Credentials)


class TestMidjourneyCredentials:
    def test_can_be_instantiated(self):
        creds = MidjourneyCredentials()
        assert isinstance(creds, Credentials)


class TestQdrantCredentials:
    def test_can_be_instantiated(self):
        creds = QdrantCredentials()
        assert isinstance(creds, Credentials)


class TestPicoVoiceCredentials:
    def test_can_be_instantiated(self):
        creds = PicoVoiceCredentials()
        assert isinstance(creds, Credentials)


class TestGoogleVisionCredentials:
    def test_can_be_instantiated(self):
        creds = GoogleVisionCredentials()
        assert isinstance(creds, Credentials)


class TestAWSTextractCredentials:
    def test_can_be_instantiated(self):
        creds = AWSTextractCredentials()
        assert isinstance(creds, Credentials)


class TestResolveCredentials:
    def test_returns_given_credentials_when_not_none(self):
        creds = OpenAICredentials(api_key=SecretStr("key"))
        result = resolve_credentials(creds, OpenAICredentials)
        assert result is creds

    def test_builds_default_when_none(self):
        result = resolve_credentials(None, OpenAICredentials)
        assert isinstance(result, OpenAICredentials)
        assert result.api_key is None


class TestResolveClientOptions:
    def test_returns_given_options_when_not_none(self):
        options = ClientOptions(base_url="https://custom")
        result = resolve_client_options(options)
        assert result is options

    def test_builds_default_when_none(self):
        result = resolve_client_options(None)
        assert isinstance(result, ClientOptions)
        assert result.base_url is None


class TestResolveTimeout:
    def test_returns_config_timeout_when_not_none(self):
        result = resolve_timeout(30.0, ClientOptions(timeout=60.0))
        assert result == 30.0

    def test_falls_back_to_client_options_timeout_when_config_timeout_is_none(self):
        result = resolve_timeout(None, ClientOptions(timeout=60.0))
        assert result == 60.0

    def test_returns_none_when_both_are_none(self):
        result = resolve_timeout(None, ClientOptions())
        assert result is None


class TestResolveMaxRetries:
    def test_returns_config_max_retries_when_not_none(self):
        result = resolve_max_retries(5, ClientOptions(max_retries=3))
        assert result == 5

    def test_falls_back_to_client_options_max_retries_when_config_max_retries_is_none(
        self,
    ):
        result = resolve_max_retries(None, ClientOptions(max_retries=7))
        assert result == 7

    def test_returns_3_when_both_are_none(self):
        result = resolve_max_retries(None, ClientOptions())
        assert result == 3
