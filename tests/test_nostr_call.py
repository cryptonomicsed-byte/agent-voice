"""Nostr subscription tests: NIP-42 auth flow, CLOSED handling."""
import asyncio
import json
import pytest

from websockets.asyncio.server import serve as ws_serve


async def _run_server(handler, host="127.0.0.1", port=0):
    """Start a websockets server, yield (server, port), then close."""
    server = await ws_serve(handler, host, port)
    actual_port = server.sockets[0].getsockname()[1]
    return server, actual_port


# ---------------------------------------------------------------------------

def test_placeholder():
    """Smoke test — always passes, confirms import is clean."""
    from agent_voice.endpoint import subscribe_calls, VoiceEndpoint  # noqa: F401
    assert True


@pytest.mark.asyncio
async def test_subscribe_plain():
    """Plain relay (no AUTH): yields one EVENT then stops."""
    pubkey = "aa" * 32

    async def handler(ws):
        await ws.recv()  # consume REQ
        event = {"id": "01" * 32, "pubkey": "bb" * 32, "kind": 1059,
                 "tags": [["p", pubkey]], "content": "", "sig": "cc" * 32,
                 "created_at": 0}
        await ws.send(json.dumps(["EVENT", "agent-voice-calls", event]))
        # Keep open until client disconnects
        try:
            await ws.wait_closed()
        except Exception:
            pass

    from agent_voice.endpoint import subscribe_calls
    server, port = await _run_server(handler)
    try:
        relay = f"ws://127.0.0.1:{port}"
        events = []
        async for ev in subscribe_calls(relay, pubkey):
            events.append(ev)
            break  # only need one
        assert len(events) == 1
        assert events[0]["kind"] == 1059
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_subscribe_nip42_accepted():
    """Relay sends AUTH challenge; client responds and continues."""
    pubkey = "aa" * 32
    seckey = bytes(range(32))  # deterministic test key

    async def handler(ws):
        await ws.recv()  # first REQ (may be dropped by auth-requiring relay)
        await ws.send(json.dumps(["AUTH", "test-challenge-xyz"]))
        auth_msg = json.loads(await ws.recv())
        assert auth_msg[0] == "AUTH"
        assert auth_msg[1]["kind"] == 22242
        ok_id = auth_msg[1]["id"]
        await ws.send(json.dumps(["OK", ok_id, True, ""]))
        await ws.recv()  # re-issued REQ after auth
        event = {"id": "02" * 32, "pubkey": "bb" * 32, "kind": 1059,
                 "tags": [["p", pubkey]], "content": "", "sig": "cc" * 32,
                 "created_at": 0}
        await ws.send(json.dumps(["EVENT", "agent-voice-calls", event]))
        try:
            await ws.wait_closed()
        except Exception:
            pass

    from agent_voice.endpoint import subscribe_calls
    server, port = await _run_server(handler)
    try:
        relay = f"ws://127.0.0.1:{port}"
        events = []
        async for ev in subscribe_calls(relay, pubkey, seckey=seckey):
            events.append(ev)
            break
        assert len(events) == 1
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_subscribe_closed_auth_required():
    """Relay sends CLOSED auth-required: subscribe_calls raises RuntimeError."""
    pubkey = "aa" * 32

    async def handler(ws):
        await ws.recv()  # consume REQ
        await ws.send(json.dumps(
            ["CLOSED", "agent-voice-calls", "auth-required: sign in"]))
        try:
            await ws.wait_closed()
        except Exception:
            pass

    from agent_voice.endpoint import subscribe_calls
    server, port = await _run_server(handler)
    try:
        relay = f"ws://127.0.0.1:{port}"
        with pytest.raises(RuntimeError, match="auth-required"):
            async for _ in subscribe_calls(relay, pubkey):
                pass
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_subscribe_auth_no_seckey_raises():
    """AUTH challenge with no seckey supplied raises RuntimeError."""
    pubkey = "aa" * 32

    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps(["AUTH", "challenge-no-key"]))
        try:
            await ws.wait_closed()
        except Exception:
            pass

    from agent_voice.endpoint import subscribe_calls
    server, port = await _run_server(handler)
    try:
        relay = f"ws://127.0.0.1:{port}"
        with pytest.raises(RuntimeError, match="no seckey"):
            async for _ in subscribe_calls(relay, pubkey):
                pass
    finally:
        server.close()
        await server.wait_closed()
