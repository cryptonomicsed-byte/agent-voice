"""card_memory — per-card agent memory over minipae (``mem/card/<id>/*``).

A voice agent should get better at a particular business the more it talks to
that business's callers. minipae (NIP-AE engrams) is the substrate; this module
is the thin adapter that binds ONE card to its OWN namespace, so cards never
collide on slugs.

Slug contract (registered in minipae's NAMESPACES.md):

    mem/card/<card-id>/<key>     one string value

Hard rule: **memory must never take down a phone line.** Every operation here
is best-effort — if minipae is absent, its relay config is missing, or any
call raises, we log and degrade to "no memory". A working call with no memory
beats a failed call with perfect memory.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: Namespace root, registered in minipae NAMESPACES.md.
NAMESPACE_ROOT = "mem/card"
#: Key holding the appended per-card learnings ring.
LEARNINGS_KEY = "learnings"
#: Cap on how many remembered keys we load into a prompt.
MAX_LOADED_KEYS = 50
#: How many learning lines the ring keeps.
LEARNINGS_RING = 20


def card_id_for(card: dict | None) -> str:
    """Pick the memory namespace for a loaded card.

    Prefers the card's own `id`; falls back to the organization id. Returns ""
    when neither exists, which disables memory rather than guessing a shared
    namespace (writing one business's notes under another's id would be worse
    than writing none).
    """
    if not isinstance(card, dict):
        return ""
    for key in ("id", "organization_id"):
        val = card.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


class CardMemory:
    """Per-card memory. Construct with a card id; all ops are best-effort."""

    def __init__(self, card_id: str, client=None):
        self.card_id = str(card_id or "").strip()
        self._client = client
        self._prefix = f"{NAMESPACE_ROOT}/{self.card_id}/"

    # -- internals -------------------------------------------------------
    def _get_client(self):
        """Lazily build a NipaeClient. Returns None (and logs once) if unavailable."""
        if self._client is not None:
            return self._client
        try:
            from minipae import NipaeClient
        except ImportError:
            log.info("card memory off: minipae is not installed")
            return None
        try:
            self._client = NipaeClient()
        except Exception as e:  # no NIPAE_NSEC / no relays, etc.
            log.info("card memory off: %s", e)
            return None
        return self._client

    # -- API -------------------------------------------------------------
    def available(self) -> bool:
        return bool(self.card_id) and self._get_client() is not None

    def load(self) -> dict[str, str]:
        """Return this card's remembered {key: value}. Never raises.

        Keys are the slug segments after ``mem/card/<id>/``.
        """
        if not self.card_id:
            return {}
        client = self._get_client()
        if client is None:
            return {}
        try:
            entries = client.list(self._prefix)
        except Exception as e:
            log.warning("card memory read failed for %s: %s", self.card_id, e)
            return {}
        out: dict[str, str] = {}
        for slug, val in (entries or {}).items():
            slug = str(slug)
            key = slug[len(self._prefix):] if slug.startswith(self._prefix) else slug
            if key and val:
                out[key] = str(val)
            if len(out) >= MAX_LOADED_KEYS:
                break
        return out

    def remember(self, key: str, value: str) -> bool:
        """Write one memory entry. Never raises; returns whether it landed."""
        if not self.card_id or not key:
            return False
        client = self._get_client()
        if client is None:
            return False
        slug = f"{self._prefix}{key}"
        try:
            ok = bool(client.set(slug, str(value)))
        except Exception as e:
            log.warning("card memory write failed for %s: %s", slug, e)
            return False
        log.info("card memory %s: %s", "wrote" if ok else "FAILED", slug)
        return ok

    def append_learning(self, line: str, ring: int = LEARNINGS_RING) -> bool:
        """Append one bounded line to ``mem/card/<id>/learnings``. Never raises."""
        line = (line or "").strip().replace("\n", " ")
        if not line:
            return False
        try:
            existing = self.load().get(LEARNINGS_KEY, "")
            lines = [l for l in existing.split("\n") if l.strip()]
            lines.append(line)
            return self.remember(LEARNINGS_KEY, "\n".join(lines[-ring:]))
        except Exception as e:
            log.warning("card memory append failed for %s: %s", self.card_id, e)
            return False
