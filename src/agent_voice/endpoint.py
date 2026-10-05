"""VoiceEndpoint: Nostr-signaled WebRTC voice endpoint.

Subscribes to Nostr relays for gift-wrapped call offers (NIP-17, kind:1059),
performs the WebRTC handshake via aiortc, and streams audio.

Signaling message format comes from agent-phone's signaling module.
Nostr transport uses minipae's crypto + a direct websockets subscription
(minipae.subscribe_stream is hardcoded to engram kind; we need kind:1059).
"""
from __future__ import annotations

import asyncio
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


async def subscribe_calls(relay: str, our_pubkey_hex: str):
    """Yield kind:1059 gift-wrap events addressed to us, live."""
    import websockets
    filt = {"kinds": [KIND_GIFT_WRAP], "#p": [our_pubkey_hex], "limit": 100}
    # Include recent history so we don't miss offers sent before subscribe
    filt["since"] = int(time.time()) - 300
    sub_id = "agent-voice-calls"
    async with websockets.connect(
        relay, open_timeout=15, max_size=10 * 1024 * 1024,
        ping_interval=20, ping_timeout=20,
    ) as ws:
        await ws.send(json.dumps(["REQ", sub_id, filt]))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=None))
            if msg[0] == "EVENT" and msg[1] == sub_id:
                yield msg[2]
            # EOSE / NOTICE: ignore, stay open for live events


class VoiceEndpoint:
    """A sovereign voice endpoint: Nostr in, WebRTC out."""

    def __init__(self, identity: "Identity", relays: list[str],
                 card: dict | None = None, agent_backend: str = "deepseek",
                 agent_system: str | None = None):
        if not HAS_SIGNALING:
            raise RuntimeError("agent-phone + minipae are required")
        if not HAS_AIORTC:
            raise RuntimeError("aiortc is required")
        self.identity = identity
        self.relays = relays
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
        # on_signal(RINGING) -> None; we then answer explicitly
        session.on_signal(sig, caller_pubkey)
        self.sessions[sig.call_id] = session

        pc = RTCPeerConnection()
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

        @pc.on("track")
        def on_track(track):
            log.info(f"audio track live: {track.kind}")
            if track.kind == "audio" and pipeline is not None:
                asyncio.ensure_future(
                    self._pump_audio(track, pipeline))

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

    async def _pump_audio(self, track, pipeline) -> None:
        """Feed inbound audio frames through VAD into the pipeline.

        Adapted from mesh-voice's on_audio_track (same owner).
        """
        from .pipeline import VADSegmenter, VAD_FRAME_BYTES, frame_pcm
        vad = VADSegmenter()
        buf = bytearray()
        try:
            while True:
                frame = await track.recv()
                for out in pipeline.resample_inbound(frame):
                    buf += frame_pcm(out)
                    while len(buf) >= VAD_FRAME_BYTES:
                        chunk = bytes(buf[:VAD_FRAME_BYTES])
                        del buf[:VAD_FRAME_BYTES]
                        for utt in vad.feed(chunk):
                            asyncio.ensure_future(
                                pipeline.handle_utterance(utt))
        except Exception as e:
            for utt in vad.flush():
                asyncio.ensure_future(pipeline.handle_utterance(utt))
            log.info(f"audio track ended ({e})")

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
        """Subscribe to one relay; reconnect on failure."""
        while self._running:
            try:
                log.info(f"subscribing to {relay}")
                async for event in subscribe_calls(relay, self.identity.pubkey_hex):
                    await self._handle_event(event)
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
