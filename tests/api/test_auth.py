import logging
from hashlib import sha256

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from agent_platform.api.app import create_app
from agent_platform.api.auth import ANONYMOUS_KEY_ID, build_api_key_auth
from agent_platform.config.settings import Settings
from agent_platform.core.errors import ConfigError

KEY = "correct-horse"


def _client(*keys: str, disabled: bool = False) -> TestClient:
    authenticate = build_api_key_auth([SecretStr(k) for k in keys], disabled=disabled)
    app = FastAPI()

    @app.get("/whoami")
    def whoami(key_id: str = Depends(authenticate)) -> dict[str, str]:
        return {"key_id": key_id}

    return TestClient(app)


class TestAuthenticate:
    def test_valid_key_returns_its_sha256_digest_as_id(self):
        response = _client("other", KEY).get(
            "/whoami", headers={"Authorization": f"Bearer {KEY}"}
        )

        assert response.status_code == 200
        assert response.json() == {"key_id": sha256(KEY.encode()).hexdigest()}

    def test_scheme_is_case_insensitive(self):
        response = _client(KEY).get(
            "/whoami", headers={"Authorization": f"bearer {KEY}"}
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": KEY},
            {"Authorization": f"Basic {KEY}"},
            {"Authorization": "Bearer wrong"},
            {"Authorization": "Bearer "},
            {"Authorization": f"Bearer {KEY[:-1]}"},
            {"Authorization": f"Bearer {KEY}x"},
        ],
    )
    def test_rejects_with_401_and_challenge(self, headers):
        response = _client(KEY).get("/whoami", headers=headers)

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    def test_non_ascii_key_is_401_not_500(self):
        response = _client(KEY).get(
            "/whoami",
            headers={"Authorization": "Bearer ké€".encode("latin-1", "ignore")},
        )

        assert response.status_code == 401

    def test_non_ascii_configured_key_authenticates(self):
        key = "clé-€"
        response = _client(key).get(
            "/whoami", headers={"Authorization": f"Bearer {key}".encode()}
        )

        assert response.status_code == 200
        assert response.json() == {"key_id": sha256(key.encode()).hexdigest()}

    def test_disabled_returns_anonymous_without_reading_the_header(self):
        response = _client(disabled=True).get("/whoami")

        assert response.json() == {"key_id": ANONYMOUS_KEY_ID}

    def test_response_never_contains_the_raw_key(self):
        response = _client(KEY).get(
            "/whoami", headers={"Authorization": f"Bearer {KEY}"}
        )

        assert KEY not in response.text


class TestStartupCheck:
    def test_no_keys_and_auth_enabled_raises_config_error(self):
        with pytest.raises(ConfigError) as exc:
            create_app(Settings(_env_file=None))

        assert "AGENT_PLATFORM_API_KEYS" in str(exc.value)
        assert "AGENT_PLATFORM_API_AUTH_DISABLED" in str(exc.value)

    def test_disabled_auth_starts_and_warns_once(self, caplog):
        with caplog.at_level(logging.WARNING, logger="agent_platform.api.app"):
            create_app(Settings(_env_file=None, api_auth_disabled=True))

        warnings = [r for r in caplog.records if r.name == "agent_platform.api.app"]
        assert len(warnings) == 1
        assert "disabled" in warnings[0].getMessage()
