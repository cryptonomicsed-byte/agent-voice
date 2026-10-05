"""GAP-4: Voice pipeline — VAD -> STT -> CARD-grounded LLM -> TTS.

Adapted from mesh-voice's voice_pipeline.py (same owner, HelixMain IP).
The key difference: the agent's system prompt is compiled from CARD.md,
with explicit null-price refusal (the WS5 "live refusal" path).

Pipeline:
    inbound WebRTC audio (48kHz) --resample--> 16kHz mono PCM
        --VADSegmenter--> utterances --Transcriber--> text
        --CardGroundedAgent--> reply text --Synthesizer--> 48kHz mono PCM
        --QueueAudioTrack--> outbound WebRTC audio
"""
from __future__ import annotations

import asyncio
import fractions
import io
import logging
import os
import wave

log = logging.getLogger(__name__)

try:
    import av
    from aiortc import AudioStreamTrack
    from aiortc.mediastreams import MediaStreamError
    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False

    # Stubs so the module still imports when aiortc/av are absent — e.g. Termux,
    # where `av` cannot build (needs ffmpeg dev headers). Without these the
    # module-level `class QueueAudioTrack(AudioStreamTrack)` raises NameError at
    # import and takes the whole pipeline (LLM/STT/TTS included) down with it.
    # The media path is still gated on HAS_AIORTC, so nothing silently runs.
    class AudioStreamTrack:  # type: ignore[no-redef]
        pass

    class MediaStreamError(Exception):  # type: ignore[no-redef]
        pass

IN_RATE = 16000
OUT_RATE = 48000
FRAME_MS = 20
OUT_SAMPLES = OUT_RATE * FRAME_MS // 1000
OUT_BYTES = OUT_SAMPLES * 2
VAD_FRAME_BYTES = IN_RATE * 2 * 30 // 1000


def build_voice_system_prompt(card: dict) -> str:
    """Compile a concise spoken-style system prompt from a CARD.md dict.

    Includes the explicit null-price refusal rule: if an offering has no
    price in the card, the agent must NOT invent one — it says so and
    offers escalation (the WS5 live-refusal acceptance criterion).
    """
    ident = card.get("identity", {})
    voice = card.get("voice", {})
    offerings = card.get("offerings", [])
    policies = card.get("policies", [])
    handoff = card.get("handoff", {})

    L = []
    name = ident.get("name", "this business")
    L.append(f"You are the AI phone agent for **{name}**.")
    L.append("You are on a live voice call. Keep every reply to one or two "
             "short spoken sentences. No lists, no markdown, no emojis.")
    if voice.get("tone"):
        L.append(f"Tone: {', '.join(voice['tone'])}.")
    if voice.get("personality"):
        L.append(f"Personality: {voice['personality']}")
    L.append("")
    if offerings:
        L.append("What the business offers:")
        for o in offerings:
            line = f"- {o.get('name')}"
            if o.get("description"):
                line += f" — {o['description']}"
            price = o.get("price_hint") or o.get("price")
            if price:
                line += f" ({price})"
            else:
                line += " (price: NOT LISTED)"
            L.append(line)
        L.append("")
    L.append("PRICE RULE (binding): If a caller asks the price of anything "
             "marked NOT LISTED above, or anything not listed at all, do NOT "
             "invent, estimate, or guess a price. Say you don't have that "
             "price and offer to take their number for a callback.")
    if policies:
        L.append("")
        L.append("Policies:")
        for p in policies:
            topic = p.get("topic", "")
            rule = p.get("rule", "")
            L.append(f"- {topic}: {rule}")
    if handoff:
        L.append("")
        L.append(f"If you cannot help, {handoff.get('instruction', 'offer to escalate to a human.')}")
    L.append("")
    L.append("If a caller asks about anything not grounded in the above, "
             "say so honestly and offer to escalate. Never invent facts.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Audio helpers (from mesh-voice; PyAV resampling, no numpy)


def make_resampler(rate=IN_RATE):
    r = av.AudioResampler(format="s16", layout="mono", rate=rate)
    return r


def frame_pcm(frame) -> bytes:
    return bytes(frame.planes[0])


def decode_to_pcm48(data: bytes) -> bytes:
    buf = io.BytesIO(data)
    container = av.open(buf, format="mp3")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=OUT_RATE)
    out = bytearray()
    for av_frame in container.decode(audio=0):
        for frame in resampler.resample(av_frame):
            out += bytes(frame.planes[0])
    return bytes(out)


class QueueAudioTrack(AudioStreamTrack):
    """Outbound WebRTC audio: PCM bytes queued, packetized at 20ms."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue()
        self._buf = bytearray()
        self._timestamp = 0

    def put_pcm(self, pcm: bytes):
        self._buf += pcm

    def queued_seconds(self) -> float:
        return len(self._buf) / 2 / OUT_RATE

    async def recv(self):
        while len(self._buf) < OUT_BYTES:
            try:
                chunk = self._queue.get_nowait()
                self._buf += chunk
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)
        data = bytes(self._buf[:OUT_BYTES])
        del self._buf[:OUT_BYTES]
        frame = av.AudioFrame(format="s16", layout="mono", samples=OUT_SAMPLES)
        frame.planes[0].update(data)
        frame.sample_rate = OUT_RATE
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, OUT_RATE)
        self._timestamp += OUT_SAMPLES
        return frame


# ---------------------------------------------------------------------------
# VAD / STT / Agent / TTS (from mesh-voice)


def _make_vad(aggressiveness=2):
    import webrtcvad
    return webrtcvad.Vad(aggressiveness)


class VADSegmenter:
    """Collects speech into utterances; emits on trailing silence."""

    def __init__(self, aggressiveness=2, silence_frames=20,
                 pre_roll_frames=10, max_seconds=30, _vad=None):
        self._vad = _vad or _make_vad(aggressiveness)
        self.silence_frames = silence_frames
        self.max_frames = int(max_seconds * 1000 / 30)
        self._pre = []
        self._pre_n = pre_roll_frames
        self._utterance = None
        self._silence = 0

    def feed(self, frame30ms: bytes):
        if len(frame30ms) != VAD_FRAME_BYTES:
            raise ValueError(f"VAD needs {VAD_FRAME_BYTES}-byte frames")
        is_speech = self._vad.is_speech(frame30ms, IN_RATE)
        done = []
        if self._utterance is None:
            self._pre.append(frame30ms)
            if len(self._pre) > self._pre_n:
                self._pre.pop(0)
            if is_speech:
                self._utterance = bytearray(b"".join(self._pre))
                self._pre.clear()
                self._silence = 0
        else:
            self._utterance += frame30ms
            if is_speech:
                self._silence = 0
            else:
                self._silence += 1
                if (self._silence >= self.silence_frames or
                        len(self._utterance) >= self.max_frames * VAD_FRAME_BYTES):
                    done.append(bytes(self._utterance))
                    self._utterance = None
        return done

    def flush(self):
        if self._utterance:
            out = [bytes(self._utterance)]
            self._utterance = None
            return out
        return []


class Transcriber:
    def __init__(self, model="base"):
        self._model = None
        self._name = model

    def _load(self):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(self._name, device="cpu", compute_type="int8")

    def transcribe(self, pcm16k: bytes) -> str:
        if self._model is None:
            self._load()
        import numpy as np
        audio = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="en")
        return " ".join(s.text for s in segments).strip()


class CardGroundedAgent:
    """LLM agent with a CARD-compiled system prompt.

    backend="deepseek" (default) or "echo" for dry runs.
    """

    def __init__(self, bot_name, backend="deepseek", card=None,
                 system=None):
        self.bot_name = bot_name
        self.backend = backend
        if system:
            self.system = system
        elif card:
            self.system = build_voice_system_prompt(card)
        else:
            self.system = (
                f"You are {bot_name}, a friendly voice assistant. "
                "Keep replies to one or two short spoken sentences.")
        self._client = None
        self._model: str = "deepseek-chat"
        if backend in ("deepseek", "openai"):
            # Provider resolution: an explicit LLM_* env override wins over the
            # DeepSeek default, so ANY OpenAI-compatible endpoint (NVIDIA NIM,
            # OpenRouter, a local Ollama, self-hosted vLLM) can back the voice
            # agent without a code change. DeepSeek stays the default for
            # backward compatibility when nothing is overridden.
            key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError(
                    "no LLM key: set LLM_API_KEY (or DEEPSEEK_API_KEY)")
            base = (os.environ.get("LLM_BASE_URL")
                    or os.environ.get("DEEPSEEK_BASE_URL")
                    or "https://api.deepseek.com")
            self._model = (os.environ.get("LLM_MODEL")
                           or ("deepseek-chat" if "deepseek" in base else "gpt-4o-mini"))
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=base)
        elif backend != "echo":
            raise ValueError(f"unknown backend: {backend}")

    def reply(self, text: str) -> str:
        if self.backend == "echo":
            return f"You said: {text}"
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": self.system},
                      {"role": "user", "content": text}],
            max_tokens=150,
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()


class Synthesizer:
    def __init__(self, voice="en-US-AriaNeural"):
        self.voice = voice

    async def synth(self, text: str) -> bytes:
        import edge_tts
        buf = io.BytesIO()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, decode_to_pcm48, buf.read())


class VoicePipeline:
    """Full turn: transcribe -> CARD-grounded think -> speak."""

    def __init__(self, bot_name, outbound: QueueAudioTrack,
                 agent_backend="deepseek", stt_model="base",
                 tts_voice="en-US-AriaNeural", card=None, agent_system=None):
        self.bot_name = bot_name
        self.outbound = outbound
        self.stt = Transcriber(stt_model)
        self.agent = CardGroundedAgent(
            bot_name, agent_backend, card=card, system=agent_system)
        self.tts = Synthesizer(tts_voice)
        self._lock = asyncio.Lock()
        self._resampler = make_resampler()

    def resample_inbound(self, frame) -> list:
        return self._resampler.resample(frame)

    async def handle_utterance(self, pcm16k: bytes):
        async with self._lock:
            if len(pcm16k) < IN_RATE * 2 * 0.5:
                return
            if self.outbound.queued_seconds() > 2.0:
                log.info(f"[{self.bot_name}] dropping utterance (speaking)")
                return
            loop = asyncio.get_running_loop()
            try:
                text = await loop.run_in_executor(
                    None, self.stt.transcribe, pcm16k)
            except Exception as e:
                log.warning(f"[{self.bot_name}] STT failed: {e}")
                return
            if not text:
                return
            log.info(f"[{self.bot_name}] heard: {text}")
            try:
                reply = await loop.run_in_executor(None, self.agent.reply, text)
            except Exception as e:
                log.warning(f"[{self.bot_name}] agent failed: {e}")
                return
            if not reply:
                return
            log.info(f"[{self.bot_name}] says: {reply}")
            try:
                pcm48 = await self.tts.synth(reply)
            except Exception as e:
                log.warning(f"[{self.bot_name}] TTS failed: {e}")
                return
            self.outbound.put_pcm(pcm48)
