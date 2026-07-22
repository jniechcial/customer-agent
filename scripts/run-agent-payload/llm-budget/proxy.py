"""Metered LLM egress proxy for sandboxed agent runs.

Runs as a sidecar container beside the sandbox. The sandbox policy gives the
project's venv Python no route to api.openai.com / api.anthropic.com, so this
process is the only way out — and it is outside the sandbox, which means the
agent cannot raise its own limit.

Two jobs:
  1. Cap the number of upstream LLM calls. Past the cap every request gets a
     429 with an explicit reason, so an over-eager eval fails loudly instead of
     quietly spending money.
  2. Hold the real API keys. The sandbox gets a per-run token in place of a key;
     the Authorization header is rewritten here, so a prompt-injected agent
     cannot exfiltrate a usable credential.

That token is also what keeps budgets private. Every proxy listens on the
gateway's shared bridge network, so without it a concurrent run could spend a
neighbour's budget just by resolving `llm-budget-<other-run>:4000`. Requests
must present the run's own token as the API key, and each run's token is
generated at launch and known only inside that run's sandbox.

Routing is by first path segment:
    OPENAI_BASE_URL=http://llm-budget-<run>:4000/openai/v1
    ANTHROPIC_BASE_URL=http://llm-budget-<run>:4000/anthropic

GET /_budget returns the tally for the host to read at teardown.

Stdlib only — no build step, nothing to keep up to date.
"""

from __future__ import annotations

import hmac
import json
import os
import ssl
import sys
import threading
from http.client import HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 0 means unlimited (run-agent --unlimited-calls).
MAX_CALLS = int(os.environ.get("MAX_CALLS", "25"))
PORT = int(os.environ.get("PORT", "4000"))
RUN_TOKEN = os.environ.get("RUN_TOKEN", "")

# first path segment -> (upstream host, how to attach the real credential)
UPSTREAMS = {
    "openai": ("api.openai.com", "bearer", "OPENAI_API_KEY"),
    "anthropic": ("api.anthropic.com", "x-api-key", "ANTHROPIC_API_KEY"),
    "voyage": ("api.voyageai.com", "bearer", "VOYAGE_API_KEY"),
}

# Headers we must not relay: hop-by-hop, or ours to set.
STRIP = {"host", "authorization", "x-api-key", "connection", "content-length",
         "transfer-encoding", "accept-encoding"}

_lock = threading.Lock()
_state = {
    "calls": 0,
    "denied": 0,
    "rejected": 0,          # requests carrying another run's token
    "by_upstream": {},
    "input_tokens": 0,
    "output_tokens": 0,
    "limit": MAX_CALLS or "unlimited",
}


def _note_usage(body: bytes) -> None:
    """Best-effort token accounting. Streamed (SSE) replies are skipped."""
    try:
        usage = json.loads(body).get("usage") or {}
    except Exception:
        return
    with _lock:
        _state["input_tokens"] += int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        )
        _state["output_tokens"] += int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
        sys.stderr.write("llm-budget: %s\n" % (fmt % args))

    # -- helpers -----------------------------------------------------------
    def _reply(self, status: int, payload: dict, headers: dict | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _handle(self) -> None:
        # Read the body up front, even on paths that will refuse the request.
        # This is HTTP/1.1: the connection is reused, so an unread body would be
        # parsed as the next request line and the client's following call would
        # fail as a syntax error instead of seeing our 403/429.
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length) if length else b""

        if self.path.rstrip("/") == "/_budget":
            with _lock:
                self._reply(200, dict(_state))
            return

        segments = self.path.lstrip("/").split("/", 1)
        name = segments[0]
        rest = "/" + (segments[1] if len(segments) > 1 else "")

        if name not in UPSTREAMS:
            self._reply(404, {"error": "unknown_upstream", "detail": name,
                              "known": sorted(UPSTREAMS)})
            return

        host, scheme, key_env = UPSTREAMS[name]
        api_key = os.environ.get(key_env, "")
        if not api_key:
            self._reply(500, {"error": "missing_upstream_key", "detail": key_env})
            return

        # This run's budget is not the neighbours' to spend.
        presented = (self.headers.get("x-api-key")
                     or (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip())
        if not hmac.compare_digest(presented or "", RUN_TOKEN):
            with _lock:
                _state["rejected"] += 1
            self.log_message("REJECTED (bad run token) %s %s", self.command, self.path)
            self._reply(403, {"error": {
                "type": "wrong_run_token",
                "message": "This proxy belongs to a different run-agent run.",
            }}, headers={
                # Stainless SDKs (openai, anthropic) obey this and fail fast
                # instead of burning their retry budget on a permanent error.
                "x-should-retry": "false",
                "x-run-agent-error": "wrong_run_token",
                "x-run-agent-detail": (
                    "Presented API key does not match this run's token. Use the "
                    "OPENAI_API_KEY from the repo .env; do not invent a key."
                ),
            })
            return

        # Budget check happens before the request leaves the box.
        with _lock:
            if MAX_CALLS and _state["calls"] >= MAX_CALLS:
                _state["denied"] += 1
                denied = _state["denied"]
            else:
                _state["calls"] += 1
                _state["by_upstream"][name] = _state["by_upstream"].get(name, 0) + 1
                denied = 0
        if denied:
            self.log_message("DENIED (over budget) %s %s", self.command, self.path)
            # A 429 normally means "slow down and retry". This one is permanent:
            # nothing replenishes, so say so in the body AND in headers, and turn
            # off SDK retries so the failure surfaces immediately rather than
            # after two backoff rounds that look like a transient rate limit.
            detail = (
                f"run-agent LLM call budget exhausted: {MAX_CALLS} of {MAX_CALLS} "
                f"calls used. This is a hard per-run cap enforced OUTSIDE the "
                f"sandbox; it does not reset and you cannot raise it from in here. "
                f"Do not retry and do not look for another route to the API. Stop, "
                f"and report in your PR body what you could not finish and why."
            )
            self._reply(429, {
                "error": {
                    "type": "budget_exhausted",
                    "code": "budget_exhausted",
                    "param": None,
                    "message": detail,
                }
            }, headers={
                "x-should-retry": "false",
                "x-run-agent-error": "budget_exhausted",
                "x-run-agent-detail": detail,
                "x-run-agent-calls-used": str(MAX_CALLS),
                "x-run-agent-calls-limit": str(MAX_CALLS),
                "x-run-agent-permanent": "true",
            })
            return

        headers = {k: v for k, v in self.headers.items() if k.lower() not in STRIP}
        if scheme == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key
        headers["Host"] = host
        if payload:
            headers["Content-Length"] = str(len(payload))

        try:
            conn = HTTPSConnection(host, 443, timeout=180,
                                   context=ssl.create_default_context())
            conn.request(self.command, rest, body=payload, headers=headers)
            upstream = conn.getresponse()
            body = upstream.read()
        except Exception as exc:  # upstream failure -> surface it, don't hang
            self.log_message("upstream error %s: %s", host, exc)
            self._reply(502, {"error": "upstream_error", "detail": str(exc)})
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

        _note_usage(body)

        self.send_response(upstream.status)
        for k, v in upstream.getheaders():
            if k.lower() in {"transfer-encoding", "connection", "content-length",
                             "content-encoding"}:
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _handle
    do_POST = _handle
    do_DELETE = _handle
    do_PATCH = _handle
    do_PUT = _handle


if __name__ == "__main__":
    if not RUN_TOKEN:
        sys.exit("llm-budget: RUN_TOKEN is required — refusing to start an "
                 "unauthenticated proxy on a shared network")
    cap = f"{MAX_CALLS} calls" if MAX_CALLS else "unlimited"
    sys.stderr.write(f"llm-budget: listening on :{PORT}, cap={cap}\n")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
