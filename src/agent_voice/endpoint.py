"""VoiceEndpoint: Nostr-signaled WebRTC voice endpoint.

Subscribes to Nostr relays for gift-wrapped call offers (NIP-17),
performs the WebRTC handshake via aiortc, and streams audio.
Signaling message format comes from agent-phone's signaling module.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

# agent-phone provides: Identity, SignalType, parse_signal, build_answer
# Install: pip install git+https://github.com/cryptonomicsed-byte/agent-phone.git
try:
    from agent_phone import Identity, SignalType, parse_signal, build_answer
    from agent_phone.signaling import CallSession
    HAS_AGENT_PHONE = True
except ImportError:
    HAS_AGENT_PHONE = False
    log.warning("agent-phone not installed; signaling disabled")

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False
    log.warning("aiortc not installed; media disabled")


class VoiceEndpoint:
    """A sovereign voice endpoint: Nostr in, WebRTC out."""

    def __init__(self, identity: "Identity", relays: list[str]):
        if not HAS_AGENT_PHONE:
            raise RuntimeError("agent-phone is required")
        if not HAS_AIORTC:
            raise RuntimeError("aiortc is required")
        self.identity = identity
        self.relays = relays
        self.sessions: dict[str, CallSession] = {}
        self.peer_connections: dict[str, RTCPeerConnection] = {}

    async def handle_offer(self, event: dict) -> dict | None:
        """Decrypt an incoming offer, create a WebRTC answer.

        Returns the gift-wrapped answer event to publish, or None if
        the offer was invalid/ignored.
        """
        sig = parse_signal(event, self.identity)
        if sig is None or sig.type is not SignalType.OFFER:
            return None

        log.info(f"Incoming call: {sig.call_id[:8]}...")

        # New call session
        session = CallSession(call_id=sig.call_id)
        self.sessions[sig.call_id] = session

        # WebRTC: set remote offer, create answer
        pc = RTCPeerConnection()
        self.peer_connections[sig.call_id] = pc

        @pc.on("track")
        def on_track(track):
            log.info(f"Audio track received: {track.kind}")
            # TODO: wire to voice pipeline (GAP-4)
            # For now, the track flows; pipeline attaches here.

        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=sig.sdp, type="offer"
        ))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Build gift-wrapped answer via agent-phone
        # TODO: need caller pubkey from the offer event for addressing
        log.info(f"Answer created for {sig.call_id[:8]}...")
        return {
            "call_id": sig.call_id,
            "sdp": pc.localDescription.sdp,
            "type": "answer",
        }

    async def handle_hangup(self, event: dict) -> None:
        """Tear down a call."""
        sig = parse_signal(event, self.identity)
        if sig is None or sig.type is not SignalType.HANGUP:
            return
        pc = self.peer_connections.pop(sig.call_id, None)
        if pc:
            await pc.close()
            log.info(f"Call {sig.call_id[:8]}... closed")
        self.sessions.pop(sig.call_id, None)

    async def run(self) -> None:
        """Main loop: subscribe to Nostr, dispatch to handlers."""
        # TODO: implement Nostr subscription via minipae
        # For now, this is the structure; subscription wiring is next.
        log.info(f"VoiceEndpoint listening as {self.identity.npub}")
        log.info(f"Relays: {self.relays}")
        # Placeholder: keep alive until subscription is implemented
        while True:
            await asyncio.sleep(60)


async def main() -> None:
    """Entry point: load identity, start endpoint."""
    # TODO: load identity from vault/key file, relays from config
    raise NotImplementedError("wire up identity loading")


if __name__ == "__main__":
    asyncio.run(main())
