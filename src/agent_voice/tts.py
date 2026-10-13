"""Local sovereign TTS for the voice agent.

edge-tts (Microsoft) began failing at the WSS handshake with 403 from this
host: the datacenter IP is refused, and 7.2.8 is already the newest release,
so no version bump fixes it. A voice agent that cannot speak is not a voice
agent — so speech runs on Piper (ONNX; onnxruntime is already a dependency):
local, offline, and not revocable by a third party.

make_synthesizer() returns Piper when a voice model is present and chains
edge-tts behind it, so this can never regress into "no speech at all".
"""
from __future__ import annotations

import asyncio
import fractions
import io
import logging
import os
import time

from .pronunciation import normalize

log = logging.getLogger(__name__)

try:
    import av
    HAS_AV = True
except ImportError:  # pragma: no cover
    HAS_AV = False

OUT_RATE = 48000
DEFAULT_PIPER_VOICE = "/opt/agent-voice/voices/en_US-lessac-medium.onnx"

# PiperVoice.load() reads a 63 MB ONNX graph. Do it once per process, not once
# per call — _answer_offer builds a fresh VoicePipeline for every call.
_PIPER_CACHE: dict = {}


def resample_s16(pcm: bytes, in_rate: int, out_rate: int = OUT_RATE) -> bytes:
    """Resample mono s16 PCM. Every plane is sliced: PyAV resampler output is
    aligned/padded and concatenating it raw injects junk into the audio."""
    if not pcm or in_rate == out_rate:
        return pcm
    frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
    frame.sample_rate = in_rate
    frame.pts = 0
    frame.time_base = fractions.Fraction(1, in_rate)
    frame.planes[0].update(pcm)
    res = av.AudioResampler(format="s16", layout="mono", rate=out_rate)
    out = bytearray()
    for f in res.resample(frame):
        out += bytes(f.planes[0])[: f.samples * 2]
    for f in res.resample(None):  # flush the resampler tail
        out += bytes(f.planes[0])[: f.samples * 2]
    return bytes(out)


class PiperSynthesizer:
    """Piper (ONNX) TTS -> 48 kHz mono s16 PCM, matching the outbound track."""

    name = "piper"

    def __init__(self, model_path: str):
        from piper.voice import PiperVoice

        t0 = time.time()
        self._voice = PiperVoice.load(model_path)
        self.rate = int(self._voice.config.sample_rate)
        self.model_path = model_path
        log.info("piper voice loaded: %s (%d Hz, %.1fs)",
                 os.path.basename(model_path), self.rate, time.time() - t0)

    def _synth_blocking(self, text: str) -> bytes:
        pcm = bytearray()
        # Heteronyms / phone numbers / currency -> a form espeak reads right.
        text = normalize(text)
        for chunk in self._voice.synthesize(text):
            # 1.8.x yields AudioChunk; older builds yield raw bytes.
            b = getattr(chunk, "audio_int16_bytes", None)
            pcm += b if b is not None else bytes(chunk)
        return resample_s16(bytes(pcm), self.rate, OUT_RATE)

    async def synth(self, text: str) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synth_blocking, text)


class EdgeSynthesizer:
    """Microsoft edge-tts. Kept as a fallback; 403s from blocked IPs."""

    name = "edge"

    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice

    def _synth_blocking(self, text: str) -> bytes:
        import edge_tts

        buf = io.BytesIO()
        comm = edge_tts.Communicate(text, self.voice)

        async def _collect():
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])

        # edge_tts is async-only; this runs on an executor thread with no loop.
        asyncio.run(_collect())
        buf.seek(0)
        container = av.open(buf, format="mp3")
        res = av.AudioResampler(format="s16", layout="mono", rate=OUT_RATE)
        out = bytearray()
        for f in container.decode(audio=0):
            for g in res.resample(f):
                out += bytes(g.planes[0])[: g.samples * 2]
        for g in res.resample(None):
            out += bytes(g.planes[0])[: g.samples * 2]
        return bytes(out)

    async def synth(self, text: str) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synth_blocking, text)


class ChainSynthesizer:
    """Try each synthesizer in order; the first non-empty PCM wins."""

    def __init__(self, parts):
        self.parts = parts
        self.name = "+".join(p.name for p in parts)

    async def synth(self, text: str) -> bytes:
        last = None
        for p in self.parts:
            try:
                pcm = await p.synth(text)
                if pcm:
                    return pcm
                log.warning("tts %s returned no audio", p.name)
            except Exception as e:
                last = e
                log.warning("tts %s failed: %s", p.name, e)
        if last:
            raise last
        return b""


def make_synthesizer(tts_voice: str | None = None, prefer: str | None = None):
    """Build the synthesizer chain.

    AGENT_VOICE_TTS        auto (default) | piper | edge
    AGENT_VOICE_PIPER_VOICE  path to a .onnx voice (default below)
    AGENT_VOICE_TTS_VOICE    edge-tts voice name (fallback)
    """
    prefer = (prefer if prefer is not None
              else os.environ.get("AGENT_VOICE_TTS", "auto")).lower()
    model = os.environ.get("AGENT_VOICE_PIPER_VOICE", DEFAULT_PIPER_VOICE)
    edge_voice = (tts_voice
                  or os.environ.get("AGENT_VOICE_TTS_VOICE", "en-US-AriaNeural"))

    parts = []
    if prefer in ("auto", "piper"):
        if os.path.exists(model):
            try:
                synth = _PIPER_CACHE.get(model)
                if synth is None:
                    synth = PiperSynthesizer(model)
                    _PIPER_CACHE[model] = synth
                parts.append(synth)
            except Exception as e:
                log.warning("piper unavailable (%s); falling back to edge-tts", e)
        elif prefer == "piper":
            raise RuntimeError(f"AGENT_VOICE_TTS=piper but no model at {model}")
        else:
            log.info("no piper model at %s; using edge-tts", model)

    if prefer in ("auto", "edge"):
        try:
            import edge_tts  # noqa: F401
            parts.append(EdgeSynthesizer(edge_voice))
        except ImportError:
            pass

    if not parts:
        raise RuntimeError("no TTS available (no piper model and no edge-tts)")
    if len(parts) == 1:
        return parts[0]
    return ChainSynthesizer(parts)
