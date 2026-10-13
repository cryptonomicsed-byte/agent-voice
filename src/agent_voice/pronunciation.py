"""Text normalization + pronunciation lexicon for the Piper (espeak-ng) front-end.

Why this exists
---------------
Piper phonemizes with espeak-ng. espeak's grapheme-to-phoneme stage is rule
based and, for many English heterographs, guesses the noun/adjective reading;
its say-as handling is thin. Two classes of error reach the caller:

  1. Heteronyms   "Save money. Live better."  -> /lˈaɪv/   (want /lˈɪv/)
                  "I read it yesterday"       -> /ɹˈiːd/   (want /ɹˈɛd/)

  2. Say-as       "555-0134"  -> "five hundred fifty five dash zero one ..."
                  "$25"       -> "dollar twenty five"
                  "24/7"      -> "twenty four slash seven"

normalize() rewrites a string into a form espeak reads correctly:

  SAY-AS    digit groups, currency and the 24/7 idiom are spelled into words
            espeak already reads well ("five five five, oh one three four";
            "25 dollars"; "twenty four seven").
  PHRASES   exact multi-word overrides, for brand-critical strings.
  HOMONYMS  context regexes that pin a heteronym using Piper's inline phoneme
            override [[IPA]], which espeak passes through verbatim. Every IPA
            symbol used here is present in the voice's phoneme_id_map
            (checked against en_US-lessac-medium).

`[[...]]` is Piper/espeak syntax, so normalize() is applied ONLY on the Piper
path; the edge-tts fallback would read the brackets literally.

To extend: add a PHRASES entry for a brand-critical string, or a RULES entry
for another heteronym. A per-company file may override/augment the phrases:
  AGENT_VOICE_PRONUNCIATION=/etc/helix/companies/<slug>/pronunciation.json
  {"phrases": {"acai bowl": "[[ɐsˈaɪi]] bowl"}}
"""
from __future__ import annotations

import json
import logging
import os
import re

log = logging.getLogger(__name__)


# --- say-as ---------------------------------------------------------------

_DIGITS = {"0": "oh", "1": "one", "2": "two", "3": "three", "4": "four",
           "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}

# (555) 555-0134 | 555-555-0134 | 555-0134 | 555.0134  ->  spoken digit groups.
_PHONE_10 = re.compile(r"\(?\b(\d{3})\)?[ .\-](\d{3})[.\-](\d{4})\b")
_PHONE_7 = re.compile(r"\b(\d{3})[.\-](\d{4})\b")
_MONEY = re.compile(r"\$(\d[\d,]*)(?:\.(\d{2}))?")
_RATIO_247 = re.compile(r"\b24/7\b")


def _spell(digits: str) -> str:
    return " ".join(_DIGITS[c] for c in digits if c.isdigit())


def _say_as_phone(text: str) -> str:
    text = _PHONE_10.sub(
        lambda m: f"{_spell(m.group(1))}, {_spell(m.group(2))}, {_spell(m.group(3))}",
        text)
    return _PHONE_7.sub(lambda m: f"{_spell(m.group(1))}, {_spell(m.group(2))}", text)


def _say_as_money(text: str) -> str:
    def repl(m):
        whole = int(m.group(1).replace(",", ""))
        cents = m.group(2)
        if cents is None:
            return f"{whole} dollar" + ("" if whole == 1 else "s")
        c = int(cents)
        if whole == 0:
            return f"{c} cent" + ("" if c == 1 else "s")
        return (f"{whole} dollar" + ("" if whole == 1 else "s")
                + f" and {c} cent" + ("" if c == 1 else "s"))
    return _MONEY.sub(repl, text)


def _say_as_ratio(text: str) -> str:
    return _RATIO_247.sub("twenty four seven", text)


# --- pronunciation --------------------------------------------------------

def _ipa(phonemes: str) -> str:
    """Wrap espeak IPA in Piper's inline phoneme override."""
    return f"[[{phonemes}]]"


# Exact, case-insensitive, word-bounded phrases -> replacement. Highest
# priority. Put brand slogans and product names here.
PHRASES: dict[str, str] = {
    "live better": f"{_ipa('lˈɪv')} better",   # verb "to live", not /laɪv/
    "live well":   f"{_ipa('lˈɪv')} well",
}

_SUBJ = r"(?:I|we|you|they|he|she|it|to|can|could|will|would|may|might|must|should|shall)"

# (compiled regex with a group named "w", IPA for that word). Applied in
# order; a match replaces ONLY the word, leaving espeak to voice the rest.
RULES: list[tuple[re.Pattern, str]] = [
    # live -> verb /lˈɪv/ after a subject pronoun/modal, or before a verb-y word
    (re.compile(rf"\b{_SUBJ}\s+(?P<w>liv(?:e|es|ed|ing))\b", re.I), "lˈɪv"),
    (re.compile(r"\b(?P<w>liv(?:e|es|ed|ing))\b(?=\s+(?:better|well|here|there|in|at|with|near|abroad|together|downtown|longer|forever))", re.I), "lˈɪv"),
    # read -> past /ɹˈɛd/ before a sequencing/demonstrative word
    (re.compile(r"\b(?P<w>read)\b(?=\s+(?:it|that|this|them|yesterday|earlier|already|before|again|your|my|his|her|the\s+(?:book|message|letter|report|note)))", re.I), "ɹˈɛd"),
    # wind -> verb /wˈaɪnd/ before up/down/the clock|watch
    (re.compile(r"\b(?P<w>wind)\b(?=\s+(?:up|down|the\s+(?:clock|watch|window)))", re.I), "wˈaɪnd"),
    # close -> verb /klˈoʊz/ before a determiner/pronoun object
    (re.compile(r"\b(?P<w>close)\b(?=\s+(?:the|a|an|your|my|our|it|them|us|before|after|now|today|early|late)\b)", re.I), "klˈoʊz"),
    # minute -> tiny /maɪnˈjuːt/ before detail(s)/adjustment(s)/steak
    (re.compile(r"\b(?P<w>minute)\b(?=\s+(?:detail|details|adjustment|adjustments|amount|amounts|changes|difference|steak))", re.I), "maɪnˈjuːt"),
    # present -> verb /pɹɪzˈɛnt/ before a determiner/pronoun object
    (re.compile(r"\b(?P<w>present)\b(?=\s+(?:the|a|an|your|my|our|it|them|this|ed\b))", re.I), "pɹɪzˈɛnt"),
    # record -> verb /ɹɪkˈɔːɹd/ before a determiner/pronoun object
    (re.compile(r"\b(?P<w>record)\b(?=\s+(?:the|a|an|your|my|our|it|this|that)\b)", re.I), "ɹɪkˈɔːɹd"),
    # lead -> metal /lˈɛd/ before pipe(s)/paint/acid/battery/poisoning
    (re.compile(r"\b(?P<w>lead)\b(?=\s+(?:pipe|pipes|paint|acid|battery|poisoning|free))", re.I), "lˈɛd"),
]


_overrides_loaded = False


def _load_overrides() -> None:
    global _overrides_loaded
    if _overrides_loaded:
        return
    _overrides_loaded = True
    path = os.environ.get("AGENT_VOICE_PRONUNCIATION", "").strip()
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning("pronunciation overrides unreadable (%s): %s", path, e)
        return
    phrases = data.get("phrases") if isinstance(data, dict) else None
    if isinstance(phrases, dict):
        PHRASES.update({str(k): str(v) for k, v in phrases.items()})
        log.info("pronunciation: merged %d override phrase(s) from %s",
                 len(phrases), path)


def normalize(text: str) -> str:
    """Rewrite `text` into a form Piper/espeak reads correctly. Safe to call
    on every utterance; ordinary text is returned untouched."""
    if not text:
        return text
    _load_overrides()
    out = _say_as_ratio(text)
    out = _say_as_money(out)
    out = _say_as_phone(out)
    for phrase, repl in PHRASES.items():
        out = re.sub(rf"\b{re.escape(phrase)}\b", repl, out, flags=re.I)
    for rx, ipa in RULES:
        out = rx.sub(lambda m, _i=ipa: m.group(0).replace(m.group("w"), _ipa(_i)), out)
    return out
