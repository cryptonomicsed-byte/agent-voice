"""Card format version validation (HELIXMAIN hardening #2).

The voice agent must not ground a call on a card whose `format` it cannot
interpret. Policy:
  * format PRESENT and unsupported -> ValueError (fail closed)
  * format ABSENT (the compiled card.json case) -> loud warning, still loads
"""
import json
import logging

import pytest

from agent_voice.__main__ import (
    CARD_FORMAT_CURRENT,
    CARD_FORMAT_KNOWN_UNSUPPORTED,
    CARD_FORMAT_SUPPORTED,
    _load_card,
    validate_card_format,
)


# ---------------------------------------------------------------------------
# the compatibility table itself

def test_current_format_is_supported():
    assert CARD_FORMAT_CURRENT in CARD_FORMAT_SUPPORTED
    assert CARD_FORMAT_CURRENT == "business-card/0.1"


def test_upstream_0_2_is_known_but_not_supported():
    # card-protocol upstream is at 0.2; helixmain has deliberately NOT adopted it.
    assert "business-card/0.2" in CARD_FORMAT_KNOWN_UNSUPPORTED
    assert "business-card/0.2" not in CARD_FORMAT_SUPPORTED


# ---------------------------------------------------------------------------
# validate_card_format

def test_supported_format_returns_it():
    assert validate_card_format({"format": "business-card/0.1"}) == "business-card/0.1"


def test_unknown_format_fails_closed():
    """The acceptance case: format: business-card/9.9 is refused."""
    with pytest.raises(ValueError) as ei:
        validate_card_format({"format": "business-card/9.9"}, "bad.md")
    msg = str(ei.value)
    assert "9.9" in msg
    assert "bad.md" in msg
    assert "business-card/0.1" in msg  # tells the operator what IS supported


def test_known_upstream_format_fails_closed_with_specific_message():
    with pytest.raises(ValueError) as ei:
        validate_card_format({"format": "business-card/0.2"}, "upstream.md")
    msg = str(ei.value)
    assert "0.2" in msg
    assert "does not support yet" in msg


def test_absent_format_warns_but_does_not_raise(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_voice"):
        out = validate_card_format({"id": "dna_x"}, "card.json")
    assert out is None
    assert any("no `format` field" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _load_card end to end

def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_load_card_md_rejects_9_9(tmp_path):
    """Acceptance: agent-voice rejects a CARD.md with format business-card/9.9."""
    path = _write(tmp_path, "CARD.md",
                  "---\nformat: business-card/9.9\nid: dna_bad\nversion: 1\n---\n\n# Bad\n")
    with pytest.raises(ValueError) as ei:
        _load_card(path)
    assert "9.9" in str(ei.value)


def test_load_card_md_accepts_0_1(tmp_path):
    path = _write(tmp_path, "CARD.md",
                  "---\nformat: business-card/0.1\nid: dna_ok\nversion: 1\n---\n\n# OK\n")
    card = _load_card(path)
    assert card["id"] == "dna_ok"
    assert card["format"] == "business-card/0.1"


def test_load_card_json_without_format_warns_and_loads(tmp_path, caplog):
    """card.json has no `format` key (compiler never writes one) — must not break."""
    path = _write(tmp_path, "card.json",
                  json.dumps({"id": "dna_json", "version": 2, "offerings": []}))
    with caplog.at_level(logging.WARNING, logger="agent_voice"):
        card = _load_card(path)
    assert card["id"] == "dna_json"
    assert any("no `format` field" in r.message for r in caplog.records)


def test_load_card_json_with_unknown_format_rejected(tmp_path):
    path = _write(tmp_path, "card.json",
                  json.dumps({"id": "dna_json", "format": "business-card/9.9"}))
    with pytest.raises(ValueError):
        _load_card(path)
