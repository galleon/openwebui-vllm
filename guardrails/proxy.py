#!/usr/bin/env python3
"""
Thin HTTP proxy for NeMo Guardrails.

Routes:
  GET  /v1/models  -> vLLM directly  (model discovery — guardrails has no /v1/models)
  *    /*          -> nemoguardrails  (all chat traffic stays guarded)

NeMo Guardrails uses its own SSE and JSON formats that differ from OpenAI's:
  - Streaming SSE: data: {"messages":[{"role":"...", "content":"..."}]}
  - Blocked JSON:  {"messages":[{"role":"assistant","content":"..."}]}

This proxy rewrites both to the OpenAI-compatible format Open WebUI expects:
  - Streaming SSE: data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{...}}]}
  - Non-streaming:  {"choices":[{"message":{...},"finish_reason":"stop"}],...}

Additionally: when the client requested a streaming response (stream:true in the
request body) but nemoguardrails returned a non-streaming JSON (e.g. for blocked
messages), the proxy wraps the JSON in SSE so locust TTFT/ITL/TPS metrics fire.

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


# ── Format rewriters ──────────────────────────────────────────────────────────

def _to_openai(body: bytes, model: str) -> bytes | None:
    """Rewrite NeMo's {messages:[...]} non-streaming format to OpenAI choices format."""
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


def _rewrite_sse_line(line: bytes, model: str, chunk_id: str) -> bytes:
    """Rewrite a single SSE data line from NeMo format to OpenAI chunk format.

    NeMo sends:  data: {"messages":[{"role":"assistant","content":"..."}]}
    We produce:  data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{...}}]}
    """
    if not line.startswith(b"data: "):
        return line
    payload = line[6:].strip()
    if payload == b"[DONE]" or not payload:
        return line
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        return line
    if "messages" in d and "choices" not in d:
        last = d["messages"][-1]
        rewritten = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": last, "finish_reason": "stop"}],
        }
        return b"data: " + json.dumps(rewritten).encode()
    return line


def _json_to_sse(body: bytes, model: str) -> bytes:
    """Wrap a non-streaming (possibly NeMo-format) JSON body in SSE format.

    Used when the client requested stream:true but the backend returned JSON.
    Produces a minimal two-event SSE stream: one data chunk then [DONE].
    """
    rewritten = _to_openai(body, model)
    try:
        full = json.loads(rewritten or body)
        msg = full.get("choices", [{}])[0].get("message") or {}
        chunk = {
            "id": full.get("id") or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": full.get("created") or int(time.time()),
            "model": full.get("model") or model,
            "choices": [{"index": 0, "delta": msg, "finish_reason": "stop"}],
        }
        payload = json.dumps(chunk).encode()
    except Exception:
        payload = rewritten or body
    return b"data: " + payload + b"\n\ndata: [DONE]\n\n"


# ── Proxy handler ─────────────────────────────────────────────────────────────

class _Proxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: D102
        pass  # keep logs quiet; nemoguardrails already logs requests

    def _forward(
        self,
        host: str,
        port: int,
        path: str,
        rewrite_model: str | None = None,
        stream_requested: bool = False,
    ) -> None:
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

        if rewrite_model and is_stream:
            # ── Streaming: rewrite NeMo SSE events to OpenAI chunk format ────
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            buf = b""
            try:
                while True:
                    data = resp.read(4096)
                    if not data:
                        break
                    buf += data
                    # Flush complete newline-terminated SSE lines immediately.
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        out = _rewrite_sse_line(line, rewrite_model, chunk_id)
                        self.wfile.write(out + b"\n")
                    self.wfile.flush()
                if buf:
                    out = _rewrite_sse_line(buf, rewrite_model, chunk_id)
                    self.wfile.write(out)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                conn.close()

        elif rewrite_model and is_json and not is_stream:
            # ── Non-streaming JSON from nemoguardrails ──────────────────────
            raw = resp.read()
            conn.close()
            if stream_requested:
                # Client wanted SSE but backend returned JSON (common for blocked
                # messages when streaming is disabled at the guardrails config
                # level).  Wrap in SSE so locust can fire TTFT/TPS metrics.
                out = _json_to_sse(raw, rewrite_model)
                self.send_response(200)
                for k, v in resp.getheaders():
                    if k.lower() not in (
                        "transfer-encoding", "connection",
                        "content-type", "content-length",
                    ):
                        self.send_header(k, v)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            else:
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
            # ── Pass-through (GET /v1/models → vLLM, or unknown format) ─────
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

        # Extract model name and stream flag from the request body.
        model = ""
        stream_requested = False
        body_len = int(self.headers.get("Content-Length", 0) or 0)
        if body_len and self.path.startswith("/v1/chat"):
            raw = self.rfile.read(body_len)
            try:
                parsed = json.loads(raw)
                model = parsed.get("model", "")
                stream_requested = bool(parsed.get("stream", False))
            except Exception:
                pass
            # Reconstruct rfile so _forward can re-read the body.
            import io
            self.rfile = io.BytesIO(raw)
            self.headers["Content-Length"] = str(body_len)

        self._forward(
            RAILS_HOST, RAILS_PORT, self.path,
            rewrite_model=model or None,
            stream_requested=stream_requested,
        )

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
