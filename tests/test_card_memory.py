"""Per-card memory over minipae (mem/card/<id>/*) — HELIXMAIN #6.

The hard requirement under test: memory must never break a call. Every
operation degrades to a no-op on any substrate failure.
"""
import pytest

from agent_voice.card_memory import (
    LEARNINGS_KEY,
    MAX_LOADED_KEYS,
    NAMESPACE_ROOT,
    CardMemory,
    card_id_for,
)


class FakeClient:
    """Stands in for minipae.NipaeClient — no relays, no keys, no network."""

    def __init__(self, entries=None, fail=False):
        self.entries = dict(entries or {})
        self.fail = fail
        self.writes = []

    def list(self, prefix=None):
        if self.fail:
            raise RuntimeError("relay down")
        return {k: v for k, v in self.entries.items()
                if not prefix or k.startswith(prefix)}

    def set(self, slug, value):
        if self.fail:
            raise RuntimeError("relay down")
        self.entries[slug] = value
        self.writes.append((slug, value))
        return True


# ---------------------------------------------------------------------------
# card_id_for

def test_card_id_prefers_id_then_organization_id():
    assert card_id_for({"id": "kings-cut"}) == "kings-cut"
    assert card_id_for({"organization_id": "org_1"}) == "org_1"
    assert card_id_for({"id": "a", "organization_id": "b"}) == "a"


def test_card_id_blank_or_missing_disables_memory():
    # "" disables memory rather than writing under a shared namespace
    assert card_id_for({}) == ""
    assert card_id_for({"id": "   "}) == ""
    assert card_id_for(None) == ""


# ---------------------------------------------------------------------------
# load

def test_load_strips_the_namespace_prefix():
    c = CardMemory("kings-cut", client=FakeClient({
        "mem/card/kings-cut/allergies": "no latex",
        "mem/card/kings-cut/learnings": "2026-10-10 call abc (61s)",
    }))
    got = c.load()
    assert got == {"allergies": "no latex", "learnings": "2026-10-10 call abc (61s)"}


def test_load_ignores_other_cards():
    c = CardMemory("kings-cut", client=FakeClient({
        "mem/card/kings-cut/note": "mine",
        "mem/card/other-shop/note": "not mine",
    }))
    assert c.load() == {"note": "mine"}


def test_load_is_capped():
    entries = {f"mem/card/c/k{i}": "v" for i in range(MAX_LOADED_KEYS + 25)}
    assert len(CardMemory("c", client=FakeClient(entries)).load()) == MAX_LOADED_KEYS


def test_load_without_card_id_is_empty():
    assert CardMemory("", client=FakeClient({"mem/card/x/y": "v"})).load() == {}


# ---------------------------------------------------------------------------
# write

def test_remember_writes_under_the_namespace():
    fc = FakeClient()
    assert CardMemory("cafe", client=fc).remember("seating", "12 tables") is True
    assert fc.writes == [(f"{NAMESPACE_ROOT}/cafe/seating", "12 tables")]


def test_append_learning_appends_and_bounds_the_ring():
    fc = FakeClient()
    cm = CardMemory("cafe", client=fc)
    for i in range(25):
        cm.append_learning(f"call {i}")
    lines = cm.load()[LEARNINGS_KEY].split("\n")
    assert len(lines) == 20          # bounded
    assert lines[-1] == "call 24"    # newest kept
    assert lines[0] == "call 5"      # oldest evicted
    assert "call 24" not in "".join(lines[:-1])


# ---------------------------------------------------------------------------
# the hard rule: never break a call

def test_load_never_raises_on_substrate_failure():
    assert CardMemory("cafe", client=FakeClient(fail=True)).load() == {}


def test_remember_returns_false_on_substrate_failure():
    assert CardMemory("cafe", client=FakeClient(fail=True)).remember("k", "v") is False


def test_append_learning_never_raises_on_substrate_failure():
    assert CardMemory("cafe", client=FakeClient(fail=True)).append_learning("x") is False


def test_available_false_without_a_card_id():
    assert CardMemory("", client=FakeClient()).available() is False


def test_available_false_when_no_client_can_be_built(monkeypatch):
    # no minipae, no NIPAE_NSEC -> Client construction must not be attempted blindly
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "minipae":
            raise ImportError("no minipae")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    c = CardMemory("cafe")
    assert c.available() is False
    assert c.load() == {}
    assert c.remember("k", "v") is False
