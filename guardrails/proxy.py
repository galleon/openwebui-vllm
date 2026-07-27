#!/usr/bin/env python3
"""
Thin HTTP proxy for NeMo Guardrails.

Routes:
  GET  /v1/models  -> vLLM directly  (model discovery — guardrails has no /v1/models)
  *    /*          -> nemoguardrails  (all chat traffic stays guarded)

Also rewrites NeMo Guardrails' blocked-response format:
  {"messages": [{"role": "assistant", "content": "..."}]}
to the OpenAI-compatible format Open WebUI expects:
  {"choices": [{"message": {"role": "assistant", "content": "..."}, ...}]}

Runs on port 8001 (external). Nemoguardrails runs on port 8002 (internal).
Pure stdlib — no extra packages needed.
"""
import http.client
import http.server
import json
import os
import time
import urllib.parse
import uuid

_vllm = urllib.parse.urlparse(os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1"))
VLLM_HOST = _vllm.hostname or "vllm"
VLLM_PORT = _vllm.port or 8000
RAILS_HOST = "127.0.0.1"
RAILS_PORT = 8002
LISTEN_PORT = 8001


def _to_openai(body: bytes, model: str) -> bytes | None:
    """If body is NeMo's {messages:[...]} format, rewrite to OpenAI choices format."""
    try:
        d = json.loads(body)
        if "messages" in d and "choices" not in d:
            last = d["messages"][-1]
            return json.dumps({
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": last, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }).encode()
    except Exception:
        pass
    return None


class _Proxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: D102
        pass  # keep logs quiet; nemoguardrails already logs requests

    def _forward(self, host: str, port: int, path: str, rewrite_model: str | None = None) -> None:
        body_len = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(body_len) if body_len else None
        fwd_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "connection", "transfer-encoding")
        }
        try:
            conn = http.client.HTTPConnection(host, port, timeout=600)
            conn.request(self.command, path, body=body, headers=fwd_headers)
            resp = conn.getresponse()
        except Exception as exc:
            self.send_error(502, f"Upstream error: {exc}")
            return

        ct = resp.getheader("Content-Type", "")
        is_json = "application/json" in ct
        is_stream = "text/event-stream" in ct

        if rewrite_model and is_json and not is_stream:
            # Read full body so we can inspect and potentially rewrite it.
            raw = resp.read()
            rewritten = _to_openai(raw, rewrite_model)
            out = rewritten if rewritten is not None else raw
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        else:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(32768)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                conn.close()

    def _route(self) -> None:
        if self.command == "GET" and self.path.rstrip("/") in ("/v1/models", "/v1/models/"):
            self._forward(VLLM_HOST, VLLM_PORT, "/v1/models")
            return

        # Extract model name from request body for potential response rewriting.
        model = ""
        body_len = int(self.headers.get("Content-Length", 0) or 0)
        if body_len and self.path.startswith("/v1/chat"):
            raw = self.rfile.read(body_len)
            try:
                model = json.loads(raw).get("model", "")
            except Exception:
                pass
            # Reconstruct rfile so _forward can re-read the body.
            import io
            self.rfile = io.BytesIO(raw)
            self.headers["Content-Length"] = str(body_len)

        self._forward(RAILS_HOST, RAILS_PORT, self.path, rewrite_model=model or None)

    def do_GET(self):    self._route()  # noqa: E704
    def do_POST(self):   self._route()  # noqa: E704
    def do_HEAD(self):   self._route()  # noqa: E704
    def do_PUT(self):    self._route()  # noqa: E704
    def do_DELETE(self): self._route()  # noqa: E704
    def do_OPTIONS(self): self._route()  # noqa: E704


if __name__ == "__main__":
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), _Proxy)
    print(f"[proxy] listening on 0.0.0.0:{LISTEN_PORT}", flush=True)
    print(f"[proxy] /v1/models  -> {VLLM_HOST}:{VLLM_PORT}", flush=True)
    print(f"[proxy] everything  -> {RAILS_HOST}:{RAILS_PORT}", flush=True)
    server.serve_forever()
