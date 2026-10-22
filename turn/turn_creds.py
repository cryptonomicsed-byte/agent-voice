#!/usr/bin/env python3
"""turn_creds.py — tiny HTTP server that issues HMAC short-term TURN credentials.

Both agent-voice (Python) and Agent-Mesh (browser, via Vantage proxy) fetch
from this endpoint.  coturn must be configured with the matching
`use-auth-secret` + `static-auth-secret` (same TURN_SECRET env var).

HMAC credential formula (RFC 5766 / coturn convention):
    username = "{expiry_unix_ts}:{label}"
    password = base64( HMAC-SHA1(TURN_SECRET, username) )

Usage:
    TURN_SECRET=<secret> TURN_HOST=omokoda.duckdns.org python3 turn_creds.py
    # listens on 127.0.0.1:8788

Env vars:
    TURN_SECRET  required — shared secret (match turnserver.conf static-auth-secret)
    TURN_HOST    optional — FQDN or IP of the TURN server (default: omokoda.duckdns.org)
    TURN_PORT    optional — TURN UDP/TCP port (default: 3478)
    CREDS_PORT   optional — port this server listens on (default: 8788)
    CREDS_BIND   optional — bind address (default: 127.0.0.1)
    TURN_TTL     optional — credential TTL in seconds (default: 86400 = 24h)
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

TURN_SECRET: str = os.environ.get("TURN_SECRET", "")
TURN_HOST: str = os.environ.get("TURN_HOST", "omokoda.duckdns.org")
TURN_PORT: str = os.environ.get("TURN_PORT", "3478")
TURN_TTL: int = int(os.environ.get("TURN_TTL", "86400"))
CREDS_PORT: int = int(os.environ.get("CREDS_PORT", "8788"))
CREDS_BIND: str = os.environ.get("CREDS_BIND", "127.0.0.1")


def generate_creds() -> dict:
    if not TURN_SECRET:
        raise RuntimeError("TURN_SECRET env var is not set")
    expiry = int(time.time()) + TURN_TTL
    username = f"{expiry}:agent"
    password = base64.b64encode(
        hmac.new(
            TURN_SECRET.encode(),
            username.encode(),
            hashlib.sha1,
        ).digest()
    ).decode()
    return {
        "urls": [
            f"turn:{TURN_HOST}:{TURN_PORT}?transport=udp",
            f"turn:{TURN_HOST}:{TURN_PORT}?transport=tcp",
            f"stun:{TURN_HOST}:{TURN_PORT}",
        ],
        "username": username,
        "credential": password,
    }


class CredHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress access log noise
        pass

    def do_GET(self):
        if self.path not in ("/turn-creds", "/turn-creds/"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            body = json.dumps(generate_creds()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except RuntimeError as e:
            err = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()


if __name__ == "__main__":
    if not TURN_SECRET:
        print("ERROR: TURN_SECRET env var required", flush=True)
        raise SystemExit(1)
    server = HTTPServer((CREDS_BIND, CREDS_PORT), CredHandler)
    print(f"turn-creds listening on {CREDS_BIND}:{CREDS_PORT}  host={TURN_HOST}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
