"""VoiceEndpoint: Nostr-signaled WebRTC voice endpoint.

Subscribes to Nostr relays for gift-wrapped call offers (NIP-17, kind:1059),
performs the WebRTC handshake via aiortc, and streams audio.

Signaling message format comes from agent-phone's signaling module.
Nostr transport uses minipae's crypto + a direct websockets subscription
(minipae.subscribe_stream is hardcoded to engram kind; we need kind:1059).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

from websockets.asyncio.client import connect as ws_connect

log = logging.getLogger(__name__)

try:
    from agent_phone import Identity, SignalType, parse_signal
    from agent_phone.signaling import CallSession
    import minipae
    HAS_SIGNALING = True
except ImportError:
    HAS_SIGNALING = False
    log.warning("agent-phone/minipae not installed; signaling disabled")

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False
    log.warning("aiortc not installed; media disabled")

KIND_GIFT_WRAP = 1059
KIND_AUTH = 22242


def _nip42_event(relay_url: str, challenge: str,
                 pubkey_hex: str, seckey: bytes) -> dict:
    """Build and sign a NIP-42 AUTH event (kind 22242).

    Signing uses secp256k1 schnorr via the `coincurve` or `cryptography`
    library if available; falls back to minipae.sign_event when present.
    """
    ev = {
        "kind": KIND_AUTH,
        "created_at": int(time.time()),
        "tags": [["relay", relay_url], ["challenge", challenge]],
        "content": "",
        "pubkey": pubkey_hex,
    }
    # Compute event id = sha256 of canonical serialization (NIP-01)
    serial = json.dumps([0, ev["pubkey"], ev["created_at"],
                         ev["kind"], ev["tags"], ev["content"]],
                        separators=(",", ":"), ensure_ascii=False)
    ev["id"] = hashlib.sha256(serial.encode()).hexdigest()

    # Sign — prefer minipae if loaded, else try coincurve
    if HAS_SIGNALING:
        import minipae as _mp
        signed = _mp.sign_event(KIND_AUTH, ev["content"], ev["tags"], seckey)
        ev["sig"] = signed["sig"]
    else:
        try:
            from coincurve import PrivateKey
            sk = PrivateKey(seckey)
            sig = sk.sign_schnorr(bytes.fromhex(ev["id"]))
            ev["sig"] = sig.hex()
        except ImportError:
            raise RuntimeError(
                "NIP-42 auth requires minipae or coincurve; "
                "install one of them or use a relay that doesn't require auth"
            )
    return ev


async def subscribe_calls(relay: str, our_pubkey_hex: str,
                           seckey: bytes | None = None):
    """Yield kind:1059 gift-wrap events addressed to us, live.

    Handles NIP-42 AUTH (sent by relay before or after REQ) and NIP-01
    CLOSED messages. If ``seckey`` is None and the relay requires auth,
    the generator raises RuntimeError rather than silently returning nothing.
    """
    filt = {
        "kinds": [KIND_GIFT_WRAP],
        "#p": [our_pubkey_hex],
        "since": int(time.time()) - 300,  # catch offers sent just before us
        "limit": 100,
    }
    sub_id = "agent-voice-calls"
    authenticated = False

    async with ws_connect(
        relay, open_timeout=15, max_size=10 * 1024 * 1024,
        ping_interval=20, ping_timeout=20,
    ) as ws:
        await ws.send(json.dumps(["REQ", sub_id, filt]))
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg[0] == "EVENT" and msg[1] == sub_id:
                yield msg[2]

            elif msg[0] == "AUTH":
                # NIP-42: relay wants us to authenticate
                challenge = msg[1]
                if seckey is None:
                    raise RuntimeError(
                        f"relay {relay} requires NIP-42 auth but no seckey provided"
                    )
                auth_ev = _nip42_event(relay, challenge, our_pubkey_hex, seckey)
                await ws.send(json.dumps(["AUTH", auth_ev]))
                log.info(f"sent NIP-42 auth to {relay}")

            elif msg[0] == "OK" and not authenticated:
                # OK response to our AUTH event — check acceptance
                accepted = msg[2] if len(msg) > 2 else False
                if not accepted:
                    reason = msg[3] if len(msg) > 3 else "unknown"
                    raise RuntimeError(f"NIP-42 auth rejected by {relay}: {reason}")
                authenticated = True
                log.info(f"NIP-42 auth accepted by {relay}")
                # Re-subscribe after auth in case relay dropped the first REQ
                await ws.send(json.dumps(["REQ", sub_id, filt]))

            elif msg[0] == "CLOSED" and len(msg) > 1 and msg[1] == sub_id:
                reason = msg[2] if len(msg) > 2 else ""
                if "auth-required" in reason or "restricted" in reason:
                    raise RuntimeError(
                        f"relay {relay} closed subscription: {reason}. "
                        "Provide seckey for NIP-42 auth."
                    )
                # Other CLOSED reasons (e.g. rate-limit) — let _relay_loop reconnect
                from websockets.exceptions import ConnectionClosed
                raise ConnectionClosed(None, None)

            # EOSE / NOTICE: ignore, stay open for live events


class VoiceEndpoint:
    """A sovereign voice endpoint: Nostr in, WebRTC out."""

    def __init__(self, identity: "Identity", relays: list[str],
                 seckey: bytes | None = None):
        if not HAS_SIGNALING:
            raise RuntimeError("agent-phone + minipae are required")
        if not HAS_AIORTC:
            raise RuntimeError("aiortc is required")
        self.identity = identity
        self.relays = relays
        self.seckey = seckey  # held for NIP-42 auth; None = unauthenticated relays only
        self.sessions: dict[str, CallSession] = {}
        self.peer_connections: dict[str, RTCPeerConnection] = {}
        self._running = False

    async def _publish(self, event: dict) -> None:
        """Publish a signed event to all relays."""
        for relay in self.relays:
            try:
                ok = await minipae.publish(relay, event)
                if not ok.get("ok"):
                    log.warning(f"relay {relay} rejected event: {ok.get('message')}")
            except Exception as e:
                log.warning(f"publish to {relay} failed: {e}")

    async def _handle_event(self, event: dict) -> None:
        """Decrypt one gift-wrap and dispatch by signal type."""
        try:
            sig = parse_signal(event, self.identity)
        except ValueError as e:
            log.warning(f"ignoring untrusted event: {e}")
            return

        caller_pubkey = bytes.fromhex(event["pubkey"])
        log.info(f"signal {sig.type.value} for call {sig.call_id[:8]}...")

        if sig.type is SignalType.OFFER:
            await self._answer_offer(sig, caller_pubkey)
        elif sig.type is SignalType.HANGUP:
            await self._close_call(sig.call_id)
        elif sig.type is SignalType.ICE:
            await self._add_ice(sig)
        # ANSWER is for callers; we only receive calls in v1

    async def _answer_offer(self, sig, caller_pubkey: bytes) -> None:
        """Set up WebRTC, create answer, publish via Nostr."""
        session = CallSession(self.identity)
        # on_signal(RINGING) -> None; we then answer explicitly
        session.on_signal(sig, caller_pubkey)
        self.sessions[sig.call_id] = session

        pc = RTCPeerConnection()
        self.peer_connections[sig.call_id] = pc

        @pc.on("track")
        def on_track(track):
            log.info(f"audio track live: {track.kind}")
            # TODO GAP-4: attach voice pipeline here
            # (VAD -> STT -> CARD-grounded LLM -> TTS -> playback track)

        @pc.on("connectionstatechange")
        async def on_state():
            log.info(f"call {sig.call_id[:8]}: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed"):
                await self._close_call(sig.call_id)

        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=sig.sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        answer_event = session.answer(pc.localDescription.sdp)
        await self._publish(answer_event)
        log.info(f"answered call {sig.call_id[:8]}...")

    async def _add_ice(self, sig) -> None:
        """Add a trickle ICE candidate (v1 may bundle; this is for completeness)."""
        # TODO: implement when caller does trickle ICE
        log.debug(f"ICE candidate for {sig.call_id[:8]}... (not yet wired)")

    async def _close_call(self, call_id: str) -> None:
        pc = self.peer_connections.pop(call_id, None)
        if pc:
            await pc.close()
        self.sessions.pop(call_id, None)
        log.info(f"call {call_id[:8]}... closed")

    async def _relay_loop(self, relay: str) -> None:
        """Subscribe to one relay; reconnect on transient failures.

        RuntimeError from auth rejection is NOT retried — the key is wrong
        or the relay doesn't accept us; surface it immediately.
        """
        while self._running:
            try:
                log.info(f"subscribing to {relay}")
                async for event in subscribe_calls(
                    relay, self.identity.pubkey_hex, self.seckey
                ):
                    await self._handle_event(event)
            except RuntimeError:
                raise  # auth rejections and missing-seckey are fatal
            except Exception as e:
                log.warning(f"relay {relay} error: {e}; reconnecting in 5s")
                await asyncio.sleep(5)

    async def run(self) -> None:
        """Main loop: subscribe to all relays concurrently."""
        self._running = True
        log.info(f"VoiceEndpoint live as {self.identity.npub[:16]}...")
        log.info(f"relays: {self.relays}")
        await asyncio.gather(*[
            self._relay_loop(r) for r in self.relays
        ])

    def stop(self) -> None:
        self._running = False
