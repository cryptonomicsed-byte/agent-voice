"""CARD.md -> Nostr npub binding (GAP-2).

Proves that npub X represents the business in CARD.md Y.
The agent publishes a Nostr kind:0 metadata event referencing the
CARD.md acting_hash. Verifiers check: this npub claims this card.
"""
from __future__ import annotations


def bind_card_to_npub(card_path: str, npub: str) -> dict:
    """Create the binding record: npub -> {card_hash, ...}.

    Returns a dict suitable for embedding in a Nostr kind:0 event
    or a CARD.md networks.voice entry.
    """
    # TODO: compute acting_hash from CARD.md via card-protocol
    raise NotImplementedError("wire up card-protocol hash")


def verify_binding(npub: str, card_hash: str, relays: list[str]) -> bool:
    """Check that npub's kind:0 metadata claims card_hash."""
    # TODO: fetch kind:0 from relays, compare
    raise NotImplementedError("wire up Nostr fetch")
