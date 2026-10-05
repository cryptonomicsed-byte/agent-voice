# agent-voice

Sovereign agent voice stack. Companion to [agent-phone](https://github.com/cryptonomicsed-byte/agent-phone).

**agent-phone** owns identity + signaling (the number and the ringing).
**agent-voice** owns media + endpoint + caller (the call itself).

## Architecture

```
Caller (browser)          Nostr Relays              Agent (agent_voice.py)
     |                         |                            |
     |--- gift-wrapped OFFER -->|                            |
     |    (SDP, encrypted)     |--- notify ---------------->|
     |                         |                       (decrypt, answer)
     |<-- gift-wrapped ANSWER --|                            |
     |                         |                            |
     |============ WebRTC P2P audio (no relays) =============|
     |<------------------- voice -------------------------->|
```

- **Signaling:** Nostr gift-wrapped events (NIP-17), via `agent-phone`.
- **Media:** WebRTC P2P via `aiortc`. Audio never touches a relay.
- **Brain:** CARD-grounded voice pipeline (VAD → STT → LLM → TTS).
- **Caller:** Static web page. No app install. QR → browser → call.

## Layout

```
src/agent_voice/
    __init__.py
    endpoint.py        # Nostr subscriber + WebRTC media (GAP-1)
    card_binding.py    # CARD.md -> Nostr npub binding (GAP-2)
    pipeline.py        # Voice pipeline wiring (GAP-4)
caller/
    index.html         # Browser call page (GAP-3)
    call.js            # nostr-tools + WebRTC
turn/
    turnserver.conf    # coturn config (GAP-5)
tests/
    test_nostr_call.py # End-to-end: Nostr offer -> WebRTC audio
```

## Status

Phase 1 in progress. See the [sovereign voice blueprint](https://github.com/cryptonomicsed-byte/agent-voice/blob/main/docs/blueprint.md) (forthcoming).

## License

Private — HelixMain IP.
