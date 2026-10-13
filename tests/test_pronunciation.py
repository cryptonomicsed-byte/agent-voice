"""Pure-string tests for the Piper text-normalization front-end (no model).

Runnable anywhere — no piper install needed. They assert normalize() rewrites
the words/say-as forms espeak gets wrong. The phoneme-level check (that the
rewritten text actually phonemizes to the intended IPA) lives in
tools/check_pronunciation.py and needs the voice model.

    python tests/test_pronunciation.py      # standalone
    pytest tests/test_pronunciation.py      # if pytest is installed
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_voice.pronunciation import normalize  # noqa: E402


def test_heteronym_live_verb():
    assert normalize("Save money. Live better.") == "Save money. [[lˈɪv]] better."


def test_heteronym_live_after_pronoun():
    assert normalize("we live downtown") == "we [[lˈɪv]] downtown"


def test_heteronym_read_past():
    assert "[[ɹˈɛd]]" in normalize("I read it yesterday")


def test_heteronym_close_verb():
    assert normalize("close the door") == "[[klˈoʊz]] the door"


def test_phoneme_overrides_are_not_real_words():
    # The verb reading of "live" must survive as an IPA override, not the
    # word "live" (espeak would re-guess it).
    assert "live" not in normalize("we live better")


def test_phone_number_spelled_as_digits():
    assert normalize("Call us at 555-0134") == \
        "Call us at five five five, oh one three four"


def test_phone_number_with_area_code():
    assert normalize("Call 555-555-0134") == \
        "Call five five five, five five five, oh one three four"


def test_money_number_then_unit():
    assert normalize("A haircut is $25") == "A haircut is 25 dollars"


def test_money_with_cents():
    assert normalize("It's $25.50") == "It's 25 dollars and 50 cents"


def test_ratio_247():
    assert normalize("We're open 24/7") == "We're open twenty four seven"


def test_ordinary_text_untouched():
    assert normalize("We close at six and take walk-ins.") == \
        "We close at six and take walk-ins."


def test_empty_string():
    assert normalize("") == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
