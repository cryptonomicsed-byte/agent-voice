"""Entry point for the agent-voice endpoint — `python3 -m agent_voice`.

`endpoint.py` exposes `VoiceEndpoint` + `subscribe_calls` but has no `__main__`
and pyproject declares no console script, so there was no way to *start* the
endpoint. This adds one.

Config (env):
    AGENT_VOICE_SECKEY   hex 32-byte nostr seckey. If unset it is read from the
                         state file, or generated once and persisted there.
    AGENT_VOICE_STATE    state dir (default: ~/.agent-voice)
    AGENT_VOICE_RELAYS   comma-separated relay URLs
                         (default: wss://omokoda.duckdns.org:3443)
    AGENT_VOICE_CARD     path to a CARD.md to ground the agent's system prompt
    AGENT_VOICE_BACKEND  "deepseek" (provider resolved from LLM_* env) or "echo"

The LLM provider comes from LLM_BASE_URL / LLM_API_KEY / LLM_MODEL — see
pipeline.py. Media requires aiortc, so run this on a real Linux host, not a
phone.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("agent_voice")

DEFAULT_RELAY = "wss://omokoda.duckdns.org:3443"


def _load_or_create_seckey(state_dir: Path) -> bytes:
    env = os.environ.get("AGENT_VOICE_SECKEY", "").strip()
    if env:
        return bytes.fromhex(env)
    keyfile = state_dir / "seckey"
    if keyfile.exists():
        return bytes.fromhex(keyfile.read_text().strip())
    import secrets
    sk = secrets.token_bytes(32)
    state_dir.mkdir(parents=True, exist_ok=True)
    keyfile.write_text(sk.hex())
    os.chmod(keyfile, 0o600)
    log.info("generated a new identity; saved to %s (0600)", keyfile)
    return sk


async def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from agent_phone import Identity

    from .endpoint import HAS_AIORTC, HAS_SIGNALING, VoiceEndpoint

    if not HAS_SIGNALING:
        print("agent-phone + minipae are required — see requirements.txt",
              file=sys.stderr)
        return 2
    if not HAS_AIORTC:
        print("aiortc is required for media — run this on a Linux host, not the phone",
              file=sys.stderr)
        return 2

    state_dir = Path(os.environ.get("AGENT_VOICE_STATE", str(Path.home() / ".agent-voice")))
    identity = Identity(_load_or_create_seckey(state_dir))

    relays = [r.strip() for r in
              os.environ.get("AGENT_VOICE_RELAYS", DEFAULT_RELAY).split(",") if r.strip()]

    card = None
    card_path = os.environ.get("AGENT_VOICE_CARD")
    if card_path:
        from .card_binding import load_card_frontmatter
        card = load_card_frontmatter(card_path)

    backend = os.environ.get("AGENT_VOICE_BACKEND", "deepseek")
    ep = VoiceEndpoint(identity, relays, seckey=identity.seckey, card=card,
                       agent_backend=backend)

    log.info("agent-voice endpoint live — npub %s…", identity.npub[:20])
    log.info("relays: %s | backend: %s | card: %s",
             relays, backend, card_path or "(none)")
    try:
        await ep.run()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        ep.stop()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
