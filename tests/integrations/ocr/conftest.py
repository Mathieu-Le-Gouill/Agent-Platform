import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


@dataclass(frozen=True)
class _Route:
    status: int
    body: bytes = b""
    location: str | None = None


@dataclass
class LocalHTTPServer:
    """A real HTTP server on 127.0.0.1 that answers configured routes.

    Unknown paths answer 404. Every requested path is appended to
    ``requested``, so a test can prove a redirect target was never fetched.
    """

    base_url: str
    routes: dict[str, _Route] = field(default_factory=dict)
    requested: list[str] = field(default_factory=list)

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def ok(self, path: str, body: bytes) -> None:
        self.routes[path] = _Route(status=200, body=body)

    def redirect(self, path: str, status: int, location: str) -> None:
        self.routes[path] = _Route(status=status, location=location)


def _handler_for(server: LocalHTTPServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            server.requested.append(self.path)
            route = server.routes.get(self.path, _Route(status=404))
            self.send_response(route.status)
            if route.location is not None:
                self.send_header("Location", route.location)
            self.send_header("Content-Length", str(len(route.body)))
            self.end_headers()
            self.wfile.write(route.body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


@pytest.fixture
def http_server() -> Iterator[LocalHTTPServer]:
    """Starts a local HTTP server on a free port and stops it after the test."""
    httpd = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    host, port = httpd.server_address[:2]
    server = LocalHTTPServer(base_url=f"http://{host!s}:{port}")
    httpd.RequestHandlerClass = _handler_for(server)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
