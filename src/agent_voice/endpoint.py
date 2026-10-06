"""VoiceEndpoint: Nostr-signaled WebRTC voice endpoint.

Subscribes to Nostr relays for gift-wrapped call offers (NIP-17, kind:1059),
performs the WebRTC handshake via aiortc, and streams audio.

Signaling message format comes from agent-phone's signaling module.
Nostr transport uses minipae's crypto + a direct websockets subscription
with NIP-42 AUTH support for relays that require authentication.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

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


def _nip42_event(challenge: str, relay_url: str, seckey: bytes) -> dict:
    """Build and sign a NIP-42 AUTH event (kind:22242)."""
    content = ""
    tags = [["relay", relay_url], ["challenge", challenge]]
    created_at = int(time.time())
    serialised = json.dumps(
        [0, "", created_at, KIND_AUTH, tags, content],
        separators=(",", ":"), ensure_ascii=False)
    event_id = hashlib.sha256(serialised.encode()).hexdigest()
    # minipae.sign_event returns a full Nostr event dict
    return minipae.sign_event(KIND_AUTH, content, tags, seckey)


async def subscribe_calls(relay: str, our_pubkey_hex: str,
                          seckey: bytes | None = None):
    """Yield kind:1059 gift-wrap events addressed to us, live.

    Handles NIP-42 AUTH challenge if the relay requires it.
    Raises RuntimeError if relay demands AUTH and no seckey supplied.
    """
    from websockets.asyncio.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosed

    filt = {"kinds": [KIND_GIFT_WRAP], "#p": [our_pubkey_hex], "limit": 100}
    filt["since"] = int(time.time()) - 300
    sub_id = "agent-voice-calls"

    async with ws_connect(
        relay, open_timeout=15, max_size=10 * 1024 * 1024,
        ping_interval=20, ping_timeout=20,
    ) as ws:
        await ws.send(json.dumps(["REQ", sub_id, filt]))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=None)
            msg = json.loads(raw)
            if msg[0] == "AUTH":
                challenge = msg[1]
                if seckey is None:
                    raise RuntimeError(
                        f"relay {relay} requires NIP-42 AUTH but no seckey supplied")
                auth_event = _nip42_event(challenge, relay, seckey)
                await ws.send(json.dumps(["AUTH", auth_event]))
                # Wait for OK confirming auth acceptance
                ok_raw = await asyncio.wait_for(ws.recv(), timeout=10)
                ok = json.loads(ok_raw)
                if ok[0] == "OK" and ok[2]:
                    log.info(f"NIP-42 AUTH accepted by {relay}")
                    # Re-issue REQ — relay may have dropped the first one
                    await ws.send(json.dumps(["REQ", sub_id, filt]))
                else:
                    log.warning(f"NIP-42 AUTH rejected by {relay}: {ok}")
            elif msg[0] == "CLOSED":
                machine_msg = msg[2] if len(msg) > 2 else ""
                if "auth-required" in machine_msg.lower():
                    raise RuntimeError(
                        f"relay {relay} closed sub: auth-required")
                log.warning(f"relay {relay} closed sub: {machine_msg}")
                break
            elif msg[0] == "EVENT" and msg[1] == sub_id:
                yield msg[2]
            # EOSE / NOTICE: ignore, stay open for live events


class VoiceEndpoint:
    """A sovereign voice endpoint: Nostr in, WebRTC out."""

    def __init__(self, identity: "Identity", relays: list[str],
                 seckey: bytes | None = None,
                 card: dict | None = None, agent_backend: str = "deepseek",
                 agent_system: str | None = None):
        if not HAS_SIGNALING:
            raise RuntimeError("agent-phone + minipae are required")
        if not HAS_AIORTC:
            raise RuntimeError("aiortc is required")
        self.identity = identity
        self.relays = relays
        self.seckey = seckey
        self.card = card
        self.agent_backend = agent_backend
        self.agent_system = agent_system
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
        session.on_signal(sig, caller_pubkey)
        self.sessions[sig.call_id] = session

        from aiortc import RTCConfiguration, RTCIceServer
        ice_cfg = RTCConfiguration(iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        ])
        pc = RTCPeerConnection(configuration=ice_cfg)
        self.peer_connections[sig.call_id] = pc

        # GAP-4: attach the voice pipeline to the outbound side
        pipeline = None
        try:
            from .pipeline import (QueueAudioTrack, VADSegmenter,
                                   VAD_FRAME_BYTES, VoicePipeline,
                                   frame_pcm)
            outbound = QueueAudioTrack()
            pc.addTrack(outbound)
            pipeline = VoicePipeline(
                self.identity.npub[:16], outbound,
                agent_backend=self.agent_backend,
                card=self.card, agent_system=self.agent_system)
            log.info(f"voice pipeline attached for {sig.call_id[:8]}...")
        except ImportError as e:
            log.warning(f"pipeline unavailable: {e}")

        _greeted = {"done": False}

        async def _greet_once():
            """Speak once media is actually up.

            MUST be driven by connectionState == 'connected'. on_track fires when
            the offer SDP arrives, which can be seconds before ICE/DTLS complete;
            queueing the greeting then drains it into a dead transport by the time
            media starts, so the caller hears a fragment or nothing.
            """
            if _greeted["done"] or pipeline is None:
                return
            _greeted["done"] = True
            await pipeline.greet()

        @pc.on("track")
        def on_track(track):
            log.info(f"audio track live: {track.kind}")
            if track.kind == "audio" and pipeline is not None:
                asyncio.ensure_future(
                    self._pump_audio(track, pipeline))

        @pc.on("connectionstatechange")
        async def on_state():
            log.info(f"call {sig.call_id[:8]}: {pc.connectionState}")
            if pc.connectionState == "connected":
                asyncio.ensure_future(_greet_once())
            if pc.connectionState in ("failed", "closed"):
                await self._close_call(sig.call_id)

        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=sig.sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        answer_event = session.answer(pc.localDescription.sdp)
        await self._publish(answer_event)
        log.info(f"answered call {sig.call_id[:8]}...")

    async def _pump_audio(self, track, pipeline) -> None:
        """Feed inbound audio frames through time-based chunking into the pipeline.

        VAD (webrtcvad) consistently classifies decoded WebRTC Opus frames as
        non-speech, so we use 3-second time-based chunks instead. Whisper
        handles silence gracefully.
        """
        from .pipeline import frame_pcm, IN_RATE
        CHUNK_BYTES = IN_RATE * 2 * 3  # 3 seconds of 16kHz s16 mono
        MIN_BYTES = IN_RATE * 2 // 2   # 0.5s minimum to avoid empty STT calls
        buf = bytearray()
        last_flush = asyncio.get_event_loop().time()
        try:
            while True:
                frame = await track.recv()
                for out in pipeline.resample_inbound(frame):
                    buf += frame_pcm(out)
                now = asyncio.get_event_loop().time()
                if len(buf) >= CHUNK_BYTES or (now - last_flush >= 3.0 and len(buf) >= MIN_BYTES):
                    chunk = bytes(buf)
                    buf.clear()
                    last_flush = now
                    asyncio.ensure_future(pipeline.handle_utterance(chunk))
        except Exception as e:
            if len(buf) >= MIN_BYTES:
                asyncio.ensure_future(pipeline.handle_utterance(bytes(buf)))
            log.info(f"audio track ended ({e})")

    async def _add_ice(self, sig) -> None:
        pc = self.peer_connections.get(sig.call_id)
        if pc is None:
            return
        try:
            from aiortc.sdp import candidate_from_sdp
            cand_json = getattr(sig, "candidate", None)
            if not cand_json:
                return
            if isinstance(cand_json, str):
                cand_json = json.loads(cand_json)
            cand_str = cand_json.get("candidate", "")
            if not cand_str:
                return
            # candidate_from_sdp expects the part after "candidate:"
            cand = candidate_from_sdp(cand_str.split(":", 1)[-1])
            cand.sdpMid = cand_json.get("sdpMid")
            cand.sdpMLineIndex = cand_json.get("sdpMLineIndex")
            await pc.addIceCandidate(cand)
            log.debug(f"ICE candidate added for {sig.call_id[:8]}...")
        except Exception as e:
            log.warning(f"addIceCandidate failed for {sig.call_id[:8]}: {e}")

    async def _close_call(self, call_id: str) -> None:
        pc = self.peer_connections.pop(call_id, None)
        if pc:
            await pc.close()
        self.sessions.pop(call_id, None)
        log.info(f"call {call_id[:8]}... closed")

    async def _relay_loop(self, relay: str) -> None:
        """Subscribe to one relay; reconnect on failure."""
        while self._running:
            try:
                log.info(f"subscribing to {relay}")
                async for event in subscribe_calls(
                        relay, self.identity.pubkey_hex, self.seckey):
                    await self._handle_event(event)
            except RuntimeError as e:
                log.error(f"relay {relay} fatal: {e}; stopping")
                self._running = False
                return
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
