from fastapi.testclient import TestClient

from agent_platform.api.app import create_app


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
