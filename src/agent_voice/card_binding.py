"""CARD.md -> Nostr npub binding (GAP-2).

Proves that npub X represents the business in CARD.md Y.

The agent publishes a Nostr kind:0 metadata event whose content includes
the CARD.md's id + acting_hash (read from frontmatter — the compiler
computes it; we don't re-parse markdown). Verifiers fetch kind:0 and
check: this npub claims this card hash.

CARD.md also gets a networks.voice entry (nostr-webrtc shape):
    networks:
      voice:
        protocol: nostr-webrtc
        public_key: <hex ed25519>
        relays: [wss://...]
        visibility: public
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import minipae
    HAS_MINIPAE = True
except ImportError:
    HAS_MINIPAE = False


def load_card_frontmatter(card_path: str) -> dict:
    """Parse the YAML frontmatter of a CARD.md file."""
    if not HAS_YAML:
        raise RuntimeError("pyyaml is required")
    with open(card_path) as f:
        text = f.read()
    if not text.startswith("---"):
        raise ValueError("CARD.md has no frontmatter")
    _, fm, _ = text.split("---", 2)
    return yaml.safe_load(fm)


def build_binding(card_path: str, npub: str, pubkey_hex: str,
                   relays: list[str]) -> dict:
    """Create the binding record (not yet signed).

    Returns the kind:0 content dict.
    """
    fm = load_card_frontmatter(card_path)
    card_id = fm.get("id")
    acting_hash = fm.get("acting_hash")
    if not card_id or not acting_hash:
        raise ValueError("CARD.md frontmatter missing id or acting_hash")
    return {
        "name": fm.get("title", card_id),
        "about": f"Sovereign voice agent for {card_id}. "
                 f"Verify: CARD.md acting_hash {acting_hash[:16]}...",
        "agent_voice_binding": {
            "card_id": card_id,
            "card_version": fm.get("version"),
            "acting_hash": acting_hash,
            "protocol": "nostr-webrtc",
            "npub": npub,
            "pubkey": pubkey_hex,
            "relays": relays,
        },
    }


def sign_binding(content: dict, seckey: bytes) -> dict:
    """Sign the binding as a Nostr kind:0 metadata event."""
    if not HAS_MINIPAE:
        raise RuntimeError("minipae is required")
    return minipae.sign_event(0, json.dumps(content), [], seckey)


async def publish_binding(event: dict, relays: list[str]) -> None:
    """Publish the signed kind:0 event to relays."""
    if not HAS_MINIPAE:
        raise RuntimeError("minipae is required")
    for relay in relays:
        try:
            ok = await minipae.publish(relay, event)
            log.info(f"binding published to {relay}: {ok.get('ok')}")
        except Exception as e:
            log.warning(f"publish to {relay} failed: {e}")


async def verify_binding(npub_hex: str, expected_hash: str,
                         relays: list[str]) -> bool:
    """Fetch npub's kind:0 from relays; check acting_hash matches.

    Returns True if a valid binding is found on any relay.
    """
    if not HAS_MINIPAE:
        raise RuntimeError("minipae is required")
    for relay in relays:
        try:
            events = await minipae.query_authenticated(
                relay, [npub_hex], None, kinds=[0])
            for ev in events or []:
                try:
                    content = json.loads(ev.get("content", "{}"))
                    binding = content.get("agent_voice_binding", {})
                    if binding.get("acting_hash") == expected_hash:
                        log.info(f"binding verified via {relay}")
                        return True
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception as e:
            log.warning(f"verify via {relay} failed: {e}")
    return False


def voice_network_entry(pubkey_hex: str, relays: list[str]) -> dict:
    """networks.voice entry for CARD.md (nostr-webrtc shape)."""
    return {
        "protocol": "nostr-webrtc",
        "public_key": pubkey_hex,
        "relays": relays,
        "visibility": "public",
    }
