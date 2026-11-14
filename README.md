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

## Call center calls (SIP, via ai-native-callcenter)

`python -m agent_voice.realtime_gateway` is the AI leg for
[ai-native-callcenter](https://github.com/rasonyang/ai-native-callcenter): it
speaks the Realtime protocol that project's `gateway` provider dials, and runs
the same speech pipeline behind it.

```bash
HELIX_REALTIME_API_KEY=<key> LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... \
  python -m agent_voice.realtime_gateway --port 8080 --companies /etc/helix/companies
```

Call center settings: `AICC_PROVIDER=gateway`,
`AICC_PROVIDER_ENDPOINT=ws://127.0.0.1:8080/v1/realtime`,
`REALTIME_API_KEY=<the same key>`, and name a transcription provider
(`AICC_TRANSCRIBE_PROVIDER`), as its docs require for a gateway.

**Which business answers.** The call center sends no number or tenant to the
provider, only the flow's settings. Set the flow's `voice` to
`helix:<card id>` or `helix:<company directory name>`; the gateway grounds the
call on that card. A reference it cannot resolve is refused, not answered
generically. The flow's own instructions are added as directions for the call,
and its tools (transfer to a person, HTTP tools) are offered to the model and
executed by the call center.

Verified: `tests/test_realtime_gateway.py`, and the call center's own client
against this gateway (`tools/callcenter-interop/`). Not yet verified: a real
SIP call through FreeSWITCH.
