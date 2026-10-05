"""GAP-1 acceptance tests: Nostr subscription wiring.

Tests run against an in-process mock relay (no network required).
Each scenario exercises a different relay behaviour.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest
import websockets
from websockets.asyncio.server import serve

from agent_voice.endpoint import subscribe_calls, KIND_GIFT_WRAP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PUBKEY = "a" * 64   # dummy recipient pubkey
OTHER_PUBKEY = "b" * 64

def _gift_wrap_event(to: str, seq: int = 0) -> dict:
    """Minimal kind:1059 event addressed to `to`."""
    return {
        "id": f"ev{seq:04d}" + "0" * 60,
        "pubkey": OTHER_PUBKEY,
        "created_at": int(time.time()),
        "kind": KIND_GIFT_WRAP,
        "tags": [["p", to]],
        "content": "encrypted_payload",
        "sig": "0" * 128,
    }


def _ws_url(server) -> str:
    host, port = server.sockets[0].getsockname()
    return f"ws://{host}:{port}"


# ---------------------------------------------------------------------------
# Scenario 1: plain subscribe — relay returns 2 events then stays open
# ---------------------------------------------------------------------------

async def _relay_plain(ws):
    raw = await ws.recv()
    msg = json.loads(raw)
    assert msg[0] == "REQ"
    sub_id = msg[1]
    await ws.send(json.dumps(["EVENT", sub_id, _gift_wrap_event(PUBKEY, 0)]))
    await ws.send(json.dumps(["EVENT", sub_id, _gift_wrap_event(PUBKEY, 1)]))
    await ws.send(json.dumps(["EOSE", sub_id]))
    # stay open; test closes after collecting 2 events


@pytest.mark.asyncio
async def test_subscribe_plain_delivers_events():
    async with serve(_relay_plain, "127.0.0.1", 0) as server:
        events = []
        async for ev in subscribe_calls(_ws_url(server), PUBKEY):
            events.append(ev)
            if len(events) == 2:
                break
    assert len(events) == 2
    assert all(e["kind"] == KIND_GIFT_WRAP for e in events)
    assert all(any(t[0] == "p" and t[1] == PUBKEY for t in e["tags"]) for e in events)


# ---------------------------------------------------------------------------
# Scenario 2: relay sends AUTH before REQ is processed, then accepts
# ---------------------------------------------------------------------------

async def _relay_auth_accepted(ws):
    # Send AUTH challenge immediately on connect
    await ws.send(json.dumps(["AUTH", "challenge-abc"]))
    # Client should send AUTH response
    raw = await ws.recv()         # could be REQ (client sent before auth processed)
    msg = json.loads(raw)
    if msg[0] == "REQ":
        # swallow the early REQ; wait for AUTH
        raw = await ws.recv()
        msg = json.loads(raw)
    assert msg[0] == "AUTH", f"expected AUTH, got {msg[0]}"
    auth_ev = msg[1]
    assert auth_ev["kind"] == 22242
    assert any(t == ["challenge", "challenge-abc"] for t in auth_ev["tags"])
    # Accept
    await ws.send(json.dumps(["OK", auth_ev["id"], True, ""]))
    # Now handle the re-issued REQ
    raw = await ws.recv()
    msg = json.loads(raw)
    assert msg[0] == "REQ"
    sub_id = msg[1]
    await ws.send(json.dumps(["EVENT", sub_id, _gift_wrap_event(PUBKEY, 0)]))
    await ws.send(json.dumps(["EOSE", sub_id]))


@pytest.mark.asyncio
async def test_subscribe_nip42_auth_accepted():
    """Relay requires NIP-42; client authenticates; events flow."""
    import os
    # Minimal seckey — just needs to exist for coincurve/minipae path.
    # Use a known test vector (not a real key; bytes 1..32).
    seckey = bytes(range(1, 33))
    try:
        from coincurve import PrivateKey
        _ = PrivateKey(seckey)
    except ImportError:
        pytest.skip("coincurve not installed; skipping NIP-42 signing test")

    async with serve(_relay_auth_accepted, "127.0.0.1", 0) as server:
        events = []
        async for ev in subscribe_calls(_ws_url(server), PUBKEY, seckey=seckey):
            events.append(ev)
            if len(events) == 1:
                break
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Scenario 3: relay sends CLOSED with auth-required — raises RuntimeError
# ---------------------------------------------------------------------------

async def _relay_closed_auth_required(ws):
    raw = await ws.recv()
    msg = json.loads(raw)
    assert msg[0] == "REQ"
    sub_id = msg[1]
    await ws.send(json.dumps(["CLOSED", sub_id, "auth-required: sign in first"]))


@pytest.mark.asyncio
async def test_subscribe_closed_auth_required_raises():
    """Relay closes with auth-required: subscribe_calls must raise, not silently stop."""
    async with serve(_relay_closed_auth_required, "127.0.0.1", 0) as server:
        with pytest.raises(RuntimeError, match="auth-required"):
            async for _ in subscribe_calls(_ws_url(server), PUBKEY):
                pass  # should never yield


# ---------------------------------------------------------------------------
# Scenario 4: relay sends AUTH, no seckey provided — raises RuntimeError
# ---------------------------------------------------------------------------

async def _relay_auth_no_seckey(ws):
    await ws.send(json.dumps(["AUTH", "challenge-xyz"]))
    # client should raise before sending anything; just wait
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_subscribe_nip42_no_seckey_raises():
    async with serve(_relay_auth_no_seckey, "127.0.0.1", 0) as server:
        with pytest.raises(RuntimeError, match="no seckey"):
            async for _ in subscribe_calls(_ws_url(server), PUBKEY, seckey=None):
                pass
