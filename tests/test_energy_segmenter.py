"""Regression tests for EnergySegmenter.

The live inbound path bypasses webrtcvad and used blind 2-second time chunks, so
utterances reached STT cut mid-word (measured ~2.5x the WER of whole utterances).
These tests pin the shapes that matter: a complete utterance is emitted once, on
trailing silence; a continuous talker is capped rather than buffered forever; and
barge-in is signalled early and only once per run of speech.
"""
import math
import struct
import sys

sys.path.insert(0, "/opt/agent-voice/src")

from agent_voice.pipeline import IN_RATE, EnergySegmenter  # noqa: E402

FRAME_MS = 20
FRAME_SAMPLES = IN_RATE * FRAME_MS // 1000


def pcm(seconds, amplitude):
    """16 kHz s16 mono: a tone at `amplitude` (0 = silence)."""
    samples = int(IN_RATE * seconds)
    if amplitude == 0:
        return b"\x00\x00" * samples
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 220 * i / IN_RATE)))
        for i in range(samples)
    )


def frames(pcm_bytes):
    step = FRAME_SAMPLES * 2
    return [pcm_bytes[i:i + step] for i in range(0, len(pcm_bytes), step)]


SPEECH = 3000   # rms ~2120, well over the 300 gate
QUIET = 40      # rms ~28, under the gate


def run(seg, pcm_bytes):
    utts, barbs = [], 0
    for f in frames(pcm_bytes):
        got, barge = seg.feed(f)
        utts += got
        barbs += 1 if barge else 0
    return utts, barbs


def test_silence_alone_emits_nothing():
    utts, barbs = run(EnergySegmenter(), pcm(3.0, 0))
    assert utts == [], f"silence produced {len(utts)} utterances"
    assert barbs == 0, "silence must not signal barge-in"


def test_speech_then_silence_emits_one_complete_utterance():
    seg = EnergySegmenter()
    utts, _ = run(seg, pcm(1.2, SPEECH) + pcm(1.0, QUIET))
    assert len(utts) == 1, f"expected 1 utterance, got {len(utts)}"
    # The whole 1.2 s of speech must be inside it; emission happens once 0.6 s of
    # trailing silence has accumulated, so the tail may stop short of the full 1.0 s.
    speech_bytes = IN_RATE * 2 * 1.2
    assert len(utts[0]) >= speech_bytes, "utterance lost speech"
    assert len(utts[0]) <= speech_bytes + IN_RATE * 2 * 0.7, "utterance over-long"


def test_short_blip_below_minimum_is_dropped():
    seg = EnergySegmenter()
    utts, _ = run(seg, pcm(0.2, SPEECH) + pcm(1.0, QUIET))
    assert utts == [], "a 0.2 s blip (under the 0.5 s floor) should not reach STT"


def test_continuous_talker_is_capped_not_unbounded():
    seg = EnergySegmenter(max_seconds=2.0)
    utts, _ = run(seg, pcm(6.0, SPEECH))
    assert len(utts) >= 2, "a 6 s monologue must be chunked, not buffered forever"
    assert all(len(u) <= IN_RATE * 2 * 2.1 for u in utts), "cap exceeded"


def test_barge_in_signals_early_and_once_per_run():
    seg = EnergySegmenter()
    _, barbs = run(seg, pcm(1.0, SPEECH))
    assert barbs == 1, f"expected exactly 1 barge-in per run of speech, got {barbs}"

    # a fresh run of speech signals again; silence in between re-arms it
    seg2 = EnergySegmenter()
    _, b1 = run(seg2, pcm(0.5, SPEECH))
    run(seg2, pcm(1.0, QUIET))
    _, b2 = run(seg2, pcm(0.5, SPEECH))
    assert b1 == 1 and b2 == 1, "barge-in must re-arm after silence"


def test_barge_in_fires_within_a_fraction_of_a_second():
    """It must not wait for a 2 s chunk boundary — that was the old behaviour."""
    seg = EnergySegmenter(barge_after=0.15)
    barbs_at = None
    t = 0.0
    for f in frames(pcm(1.0, SPEECH)):
        _, barge = seg.feed(f)
        t += FRAME_MS / 1000
        if barge and barbs_at is None:
            barbs_at = t
    assert barbs_at is not None and barbs_at <= 0.3, f"barge-in took {barbs_at}s"


def test_flush_emits_pending_audio_on_call_end():
    seg = EnergySegmenter()
    for f in frames(pcm(0.8, SPEECH)):
        seg.feed(f)
    out = seg.flush()
    assert len(out) == 1, "flush should emit the pending partial utterance"


def test_flush_drops_a_buffer_with_no_real_speech():
    seg = EnergySegmenter()
    for f in frames(pcm(0.05, SPEECH)):      # well under the speech floor
        seg.feed(f)
    assert seg.flush() == [], "a blip must not be emitted by flush"
