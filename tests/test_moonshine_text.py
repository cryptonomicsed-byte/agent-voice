"""Transcript extraction for the moonshine STT backend.

Regression cover for a real leak: moonshine's `transcribe_without_streaming` returns a
`Transcript` whose `.lines` are `TranscriptLine`s, and `.text` is the clean string. But
some lines come back with EMPTY `.text` while their repr still carries the recognised
words *and* a `metadata: [...]` block. The old fallback (`getattr(l, "text", "") or
str(l)`) therefore handed the whole metadata dump to the LLM. Captured verbatim off the
live path on 2026-10-13 (agent-voice-demo-barber journal):

    heard: [0.00s]: '', metadata: [duration=0.35s, line_id=9051562705621547848,
           is_complete=True, is_updated=True, is_new=True, has_text_changed=True,
           have_speakers_changed=False, speaker_spans=[], audio_data_len=5632,
           last_transcription_latency_ms=104, words=0] Hi, why are you opening hours?

Note `speaker_spans=[]` — the metadata block nests a list, so the closing bracket can
only be found by depth. A non-greedy regex stops at that inner `]` and leaves the rest
of the dump behind, which is exactly the trap these tests pin down.

Run: /opt/agent-voice-venv/bin/python tests/test_moonshine_text.py
"""

from agent_voice.pipeline import _moonshine_text, _strip_line_repr

# Verbatim from the journal. Two things to notice: `.text` is '' (hence the old code
# fell through to str(line)), and the words sit *after* the metadata block.
LIVE_REPR = (
    "[0.00s]: '', metadata: [duration=0.35s, line_id=9051562705621547848, "
    "is_complete=True, is_updated=True, is_new=True, has_text_changed=True, "
    "have_speakers_changed=False, speaker_spans=[], audio_data_len=5632, "
    "last_transcription_latency_ms=104, words=0] Hi, why are you opening hours?"
)
WANT = "Hi, why are you opening hours?"


class Line:
    """Stands in for moonshine's TranscriptLine: clean `.text`, noisy `str()`."""

    def __init__(self, text, repr_=None):
        self.text = text
        self._repr = repr_

    def __str__(self):
        return self._repr if self._repr is not None else f"[0.00s]: {self.text!r}, metadata: []"


class Transcript:
    def __init__(self, lines):
        self.lines = lines


def test_live_metadata_dump_is_stripped():
    assert _moonshine_text(Transcript([Line("", LIVE_REPR)])) == WANT


def test_clean_text_untouched():
    assert _moonshine_text(Transcript([Line("On Saturday.")])) == "On Saturday."


def test_nested_bracket_does_not_truncate_the_strip():
    # If the bracket scan stopped at speaker_spans=[], "audio_data_len=…, words=0]"
    # would survive; the guard would then blank the whole line and we would lose WANT.
    got = _strip_line_repr(LIVE_REPR)
    assert got == WANT
    assert "words=0" not in got and "audio_data_len" not in got


def test_the_dump_never_reaches_the_model():
    got = _moonshine_text(Transcript([Line("", LIVE_REPR), Line("On Saturday.")]))
    for smell in ("metadata:", "line_id=", "speaker_spans", "audio_data_len", "words="):
        assert smell not in got
    assert got == WANT + " On Saturday."


def test_apostrophes_survive():
    r = "[1.20s]: \"don't stop, it's fine\", metadata: [words=3]"
    assert _strip_line_repr(r) == "don't stop, it's fine"


def test_unparseable_degrades_to_silence_not_a_dump():
    # A shape we did not anticipate must not be passed through raw.
    assert _strip_line_repr("metadata: [line_id=1, words=0]") == ""
    assert _moonshine_text(Transcript([])) == ""


def test_bare_sequence_of_transcripts():
    res = [Transcript([Line("Hi,")]), Transcript([Line("", LIVE_REPR)])]
    assert _moonshine_text(res) == "Hi, " + WANT


def test_transcript_level_text_still_honoured():
    class T(Transcript):
        text = "legacy string shape"

    assert _moonshine_text(T([])) == "legacy string shape"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} pass")
