"""Serve the static frontend and proxy API routes through one local port."""

from __future__ import annotations

import argparse
import mimetypes
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO / "frontend"
API_ROUTES = {
    "/health",
    "/ready",
    "/stats",
    "/entities",
    "/route",
    "/query",
    "/deep-research",
    "/metrics",
}


def make_handler(frontend_dir: Path, api_base: str) -> type[BaseHTTPRequestHandler]:
    api_base = api_base.rstrip("/")

    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "NornikelDemo/1.0"

        def do_GET(self) -> None:
            if self._is_api_path():
                self._proxy()
                return
            self._serve_static()

        def do_HEAD(self) -> None:
            if self._is_api_path():
                self._proxy(head_only=True)
                return
            self._serve_static(head_only=True)

        def do_POST(self) -> None:
            if self._is_api_path():
                self._proxy()
                return
            self.send_error(404)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def _is_api_path(self) -> bool:
            path = self.path.split("?", 1)[0]
            return path in API_ROUTES

        def _serve_static(self, head_only: bool = False) -> None:
            raw_path = self.path.split("?", 1)[0].lstrip("/")
            rel = Path(raw_path or "index.html")
            target = (frontend_dir / rel).resolve()
            root = frontend_dir.resolve()
            if not str(target).startswith(str(root)) or not target.exists() or target.is_dir():
                target = root / "index.html"
            content = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if not head_only:
                self.wfile.write(content)

        def _proxy(self, head_only: bool = False) -> None:
            body = None
            if self.command in {"POST", "PUT", "PATCH"}:
                length = int(self.headers.get("Content-Length") or "0")
                body = self.rfile.read(length) if length else b""
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "connection"}
            }
            req = urllib.request.Request(
                api_base + self.path,
                data=body,
                headers=headers,
                method=self.command,
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                    self.send_response(resp.status)
                    self._cors()
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    if not head_only:
                        self.wfile.write(data)
            except urllib.error.HTTPError as exc:
                data = exc.read()
                self.send_response(exc.code)
                self._cors()
                self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(data)
            except Exception as exc:
                data = f'{{"detail":"proxy_error: {type(exc).__name__}: {exc}"}}'.encode("utf-8")
                self.send_response(502)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(data)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type,Authorization,X-API-Key")

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.address_string()} - {fmt % args}")

    return DemoHandler


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve frontend and API through one local port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--frontend", type=Path, default=FRONTEND_DIR)
    args = parser.parse_args(argv)

    handler = make_handler(args.frontend, args.api)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Demo server: http://{args.host}:{args.port} -> {args.frontend} + {args.api}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping demo server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
