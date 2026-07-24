from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from agent_platform.api.app import create_app, main


class FakeAgent:
    async def chat(self, user_input: str) -> str:
        return f"echo: {user_input}"


class TestHealth:
    def test_returns_ok(self):
        with TestClient(create_app()) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChat:
    def test_returns_agent_response(self):
        app = create_app()
        with TestClient(app) as client:
            app.state.agent = FakeAgent()
            response = client.post("/chat", json={"message": "hi"})

        assert response.status_code == 200
        assert response.json() == {"response": "echo: hi"}

    def test_requires_message_field(self):
        with TestClient(create_app()) as client:
            response = client.post("/chat", json={})

        assert response.status_code == 422


class TestMain:
    def test_runs_uvicorn_with_settings_host_and_port(self, mocker):
        mock_uvicorn = MagicMock()
        mocker.patch.dict("sys.modules", {"uvicorn": mock_uvicorn})
        mock_get_settings = mocker.patch("agent_platform.api.app.get_settings")
        mock_get_settings.return_value = MagicMock(api_host="0.0.0.0", api_port=9000)
        main()

        mock_uvicorn.run.assert_called_once()
        _, kwargs = mock_uvicorn.run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000
