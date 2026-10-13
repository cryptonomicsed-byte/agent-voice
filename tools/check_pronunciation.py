#!/usr/bin/env python3
"""Assert the Piper front-end phonemizes the lexicon's words correctly.

Needs the piper package + the voice model, so it runs where the agent runs:

    /opt/agent-voice-venv/bin/python tools/check_pronunciation.py [-v]

For each case it phonemizes the NORMALIZED text and asserts the intended IPA
appears (and the wrong reading does not). Comparison is on a compact form —
stress marks, punctuation and inter-symbol spaces stripped, because espeak
scatters ˈ/ˌ through its output. Exits non-zero on any failure, so it can gate
a deploy.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piper.voice import PiperVoice                      # noqa: E402
from agent_voice.pronunciation import normalize         # noqa: E402

VOICE = os.environ.get("AGENT_VOICE_PIPER_VOICE",
                       "/opt/agent-voice/voices/en_US-lessac-medium.onnx")
v = PiperVoice.load(VOICE)


def ph(text: str) -> str:
    return " ".join(" ".join(s) for s in v.phonemize(text))


def compact(text: str) -> str:
    s = ph(text)
    for ch in " ˈˌ.,!?;:'\"-":
        s = s.replace(ch, "")
    return s


# (label, input, must_contain, must_not_contain) — compact phoneme strings.
CASES = [
    ("live (verb, brand slogan)", "Save money. Live better.", "lɪv", "laɪv"),
    ("live (verb, after pronoun)", "we live downtown", "lɪv", "laɪv"),
    ("read (past)", "I read it yesterday", "ɹɛd", None),
    ("close (verb)", "close the door", "kloʊz", "kloʊs"),
    ("lead (metal)", "lead pipe", "lɛd", "liːd"),
    ("phone number", "Call us at 555-0134", "oʊwʌnθɹiːfɔːɹ", "hʌndɹɪd"),
    ("phone: not 'dash'", "Call us at 555-0134", "faɪvfaɪvfaɪv", "dæʃ"),
    ("money (number then unit)", "A haircut is $25", "twɛntifaɪvdɑːlɚz", "dɑːlɚtwɛnti"),
    ("ratio 24/7", "We're open 24/7", "fɔːɹsɛvən", "slæʃ"),
]

fails = 0
for label, text, want, avoid in CASES:
    n = normalize(text)
    got = compact(n)
    ok = (want in got) and (avoid not in got if avoid else True)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if not ok or "-v" in sys.argv:
        print(f"        in : {text}")
        print(f"        out: {n}")
        print(f"        ipa: {ph(n)}")
        if want not in got:
            print(f"        MISSING: {want!r}")
        if avoid and avoid in got:
            print(f"        PRESENT (should not be): {avoid!r}")
    if not ok:
        fails += 1

print()
print(f"{len(CASES) - fails}/{len(CASES)} passed")
sys.exit(1 if fails else 0)
