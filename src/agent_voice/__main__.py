"""Entry point for the agent-voice endpoint — `python3 -m agent_voice`.

Config (env):
    AGENT_VOICE_SECKEY   hex 32-byte nostr seckey. If unset it is read from the
                         state file, or generated once and persisted there.
    AGENT_VOICE_STATE    state dir (default: ~/.agent-voice)
    AGENT_VOICE_RELAYS   comma-separated relay URLs
                         (default: wss://omokoda.duckdns.org:3443)
    AGENT_VOICE_CARD     path to a card file. Accepts:
                           - .json  — raw card JSON (id/version/offerings/voice/…)
                           - .md    — CARD.md (only frontmatter metadata; no offerings
                                      — use the source .json for a grounded agent)
    AGENT_VOICE_BACKEND  "deepseek" (provider resolved from LLM_* env) or "echo"

LLM provider: LLM_BASE_URL / LLM_API_KEY / LLM_MODEL — see pipeline.py.
"""
from __future__ import annotations

import asyncio
import json
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


def _load_card(card_path: str) -> dict:
    """Load a card from either a .json file or a CARD.md file.

    .json  — parsed directly; must have offerings/identity/voice at top level.
    .md    — YAML frontmatter only (id/hash/status); no offerings/voice, so
             the resulting system prompt will be generic. Use .json for a
             fully-grounded agent.
    """
    if card_path.endswith(".json"):
        with open(card_path) as f:
            return json.load(f)
    # CARD.md path
    try:
        import yaml
    except ImportError:
        raise RuntimeError("pyyaml required to load CARD.md — use a .json card instead")
    with open(card_path) as f:
        text = f.read()
    if not text.startswith("---"):
        raise ValueError(f"CARD.md at {card_path} has no YAML frontmatter")
    _, fm, _ = text.split("---", 2)
    return yaml.safe_load(fm)


async def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from agent_phone import Identity
    from .endpoint import HAS_AIORTC, HAS_SIGNALING, VoiceEndpoint

    if not HAS_SIGNALING:
        print("agent-phone + minipae required — see requirements.txt", file=sys.stderr)
        return 2
    if not HAS_AIORTC:
        print("aiortc required for media — run on Linux, not a phone", file=sys.stderr)
        return 2

    state_dir = Path(os.environ.get("AGENT_VOICE_STATE",
                                    str(Path.home() / ".agent-voice")))
    identity = Identity(_load_or_create_seckey(state_dir))

    relays = [r.strip() for r in
              os.environ.get("AGENT_VOICE_RELAYS", DEFAULT_RELAY).split(",")
              if r.strip()]

    card: dict | None = None
    card_path = os.environ.get("AGENT_VOICE_CARD")
    if card_path:
        card = _load_card(card_path)
        card_type = "json" if card_path.endswith(".json") else "md"
        has_offerings = bool(card.get("offerings"))
        log.info("card loaded from %s (%s) — offerings: %s",
                 card_path, card_type,
                 len(card.get("offerings", [])) if has_offerings else "none (md frontmatter only)")

    backend = os.environ.get("AGENT_VOICE_BACKEND", "deepseek")
    ep = VoiceEndpoint(identity, relays, seckey=identity.seckey,
                       card=card, agent_backend=backend)

    log.info("relays: %s | backend: %s | card: %s", relays, backend, card_path or "(none)")
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
