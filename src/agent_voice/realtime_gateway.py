"""HELIX Realtime Gateway — the AI leg of the call center.

ai-native-callcenter (FreeSWITCH + its Go service) hands each AI call to a
"gateway" provider: a WebSocket speaking the OpenAI Realtime protocol, with
linear PCM16 at 24 kHz in both directions (its GatewayProfile). This module is
that service. Behind the protocol it runs the same pipeline that answers Nostr
and LAN calls: speech-to-text -> card-grounded agent -> text-to-speech.

Per-call agent selection. The call center connects with one credential and
sends no call id, number or tenant. What it does send per call is the flow's
`voice`, which its own docs leave to "whatever engine the gateway drives". A
HELIX flow sets it to ``helix:<card id or company slug>``; the gateway loads
that business's card from the company directory and grounds the call on it.
The flow is operator configuration, not caller input, and the connection is
authenticated, so a caller cannot choose another business's agent. An unknown
reference is refused rather than answered generically.

Tools stay with the call center: the flow's tools are offered to the model,
tool calls are returned as Realtime function calls, and the call center runs
them (its own HTTP flow tools, transfer to a human, hang up) and sends back
the results.

Run:
    python -m agent_voice.realtime_gateway --port 8080 \
        --companies /etc/helix/companies
    env: HELIX_REALTIME_API_KEY (required), LLM_BASE_URL / LLM_API_KEY /
         LLM_MODEL, and the usual agent-voice STT/TTS settings.

Call center side:
    AICC_PROVIDER=gateway
    AICC_PROVIDER_ENDPOINT=ws://127.0.0.1:8080/v1/realtime
    REALTIME_API_KEY=<the same key>
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import fractions
import hmac
import json
import logging
import os
import uuid
from pathlib import Path

from .pipeline import (IN_RATE, EnergySegmenter, build_voice_system_prompt)

log = logging.getLogger(__name__)

WIRE_RATE = 24000                     # the call center's gateway profile: PCM16 mono 24 kHz
TTS_RATE = 48000
DELTA_BYTES = WIRE_RATE * 2 // 10     # 100 ms of audio per delta
AGENT_PREFIX = "helix:"
MAX_HISTORY = 40


# ── agent directory ───────────────────────────────────────────────────────────
class AgentDirectory:
    """Resolves ``helix:<ref>`` to a business card. `ref` is a card id or a
    company directory name under the companies root."""

    def __init__(self, companies_root: str | None, default_card: dict | None = None):
        self.root = Path(companies_root) if companies_root else None
        self.default_card = default_card

    def _load(self, path: Path) -> dict | None:
        from .__main__ import _load_card          # validates the card format
        try:
            return _load_card(str(path))
        except Exception as e:
            log.warning("card at %s not usable: %s", path, e)
            return None

    def resolve(self, voice: str) -> dict | None:
        """Return the card for a flow's voice value.

        ""              -> the default card (may be None: instructions-only agent)
        "helix:<ref>"   -> that business's card; LookupError if there is none
        anything else   -> LookupError (this gateway has no vendor voices)
        """
        voice = (voice or "").strip()
        if not voice:
            return self.default_card
        if not voice.startswith(AGENT_PREFIX):
            raise LookupError(f"voice '{voice}' is not a HELIX agent reference "
                              f"(expected {AGENT_PREFIX}<card id or company>)")
        ref = voice[len(AGENT_PREFIX):].strip()
        if not ref or "/" in ref or ref.startswith("."):
            raise LookupError("empty or invalid agent reference")
        if self.root is not None and self.root.is_dir():
            direct = self.root / ref / "card.json"
            if direct.is_file():
                card = self._load(direct)
                if card is not None:
                    return card
            for path in sorted(self.root.glob("*/card.json")):
                card = self._load(path)
                if card and ref in (card.get("id"), card.get("organization_id")):
                    return card
        raise LookupError(f"no business card for agent reference '{ref}'")


# ── model backend ─────────────────────────────────────────────────────────────
class ChatBackend:
    """OpenAI-compatible chat completions with tool calling.

    complete(messages, tools) -> {"content": str, "tool_calls": [{"id","name","arguments"}]}
    """

    def __init__(self):
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY (or DEEPSEEK_API_KEY) is not set")
        from openai import OpenAI
        self._client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
        self._model = os.environ.get("LLM_MODEL", "deepseek-chat")

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        kwargs = {"model": self._model, "messages": messages,
                  "max_tokens": 200, "temperature": 0.7}
        if tools:
            kwargs["tools"] = tools
        msg = self._client.chat.completions.create(**kwargs).choices[0].message
        return {
            "content": (msg.content or "").strip(),
            "tool_calls": [{"id": c.id, "name": c.function.name,
                            "arguments": c.function.arguments or "{}"}
                           for c in (msg.tool_calls or [])],
        }


def chat_tools(realtime_tools: list) -> list[dict]:
    """Realtime tool specs -> chat-completions tool specs."""
    out = []
    for t in realtime_tools or []:
        if isinstance(t, dict) and t.get("type") == "function" and t.get("name"):
            out.append({"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})
    return out


# ── audio ─────────────────────────────────────────────────────────────────────
class StreamResampler:
    """Stateful mono s16 resampler for a continuous stream."""

    def __init__(self, in_rate: int, out_rate: int):
        import av
        self._av, self.in_rate = av, in_rate
        self._res = av.AudioResampler(format="s16", layout="mono", rate=out_rate)
        self._pts = 0

    def feed(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        frame = self._av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
        frame.sample_rate = self.in_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.in_rate)
        frame.planes[0].update(pcm[: (len(pcm) // 2) * 2])
        self._pts += len(pcm) // 2
        out = bytearray()
        for f in self._res.resample(frame):
            out += bytes(f.planes[0])[: f.samples * 2]
        return bytes(out)


# ── one call ──────────────────────────────────────────────────────────────────
class Session:
    def __init__(self, ws, gateway: "Gateway"):
        self.ws, self.gw = ws, gateway
        self.id = "helix_" + uuid.uuid4().hex[:16]
        self.instructions = ""
        self.tools: list[dict] = []
        self.card: dict | None = gateway.directory.default_card
        self.agent_ok = True
        self.creates_response = True
        self.history: list[dict] = []
        self.pending_calls: dict[str, dict] = {}     # tool calls awaiting their output
        self.seg = EnergySegmenter()
        self.inbound = StreamResampler(WIRE_RATE, IN_RATE)
        self.response_task: asyncio.Task | None = None
        self.caller_speaking = False
        self._send_lock = asyncio.Lock()
        self._seq = 0

    def _next(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq}"

    async def send(self, event: dict) -> None:
        async with self._send_lock:
            await self.ws.send(json.dumps(event))

    async def error(self, code: str, message: str) -> None:
        await self.send({"type": "error",
                         "error": {"type": "invalid_request_error", "code": code,
                                   "message": message}})

    def system_prompt(self) -> str:
        parts = []
        if self.card:
            parts.append(build_voice_system_prompt(self.card))
        if self.instructions:
            parts.append(("## Directions for this call\n" if self.card else "") + self.instructions)
        return "\n\n".join(parts) or (
            "You are a helpful voice assistant. Keep replies to one or two short spoken sentences.")

    # -- inbound events --------------------------------------------------------
    async def run(self) -> None:
        await self.send({"type": "session.created", "session": {"id": self.id}})
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    event = json.loads(raw)
                    kind = event.get("type")
                except (ValueError, AttributeError):
                    continue
                handler = getattr(self, "_on_" + str(kind).replace(".", "_"), None)
                if handler is None:
                    log.debug("ignored client event %s", kind)
                    continue
                await handler(event)
                if not self.agent_ok:
                    return
        finally:
            await self._cancel_response(announce=False)

    async def _on_session_update(self, event: dict) -> None:
        s = event.get("session") or {}
        if "instructions" in s:
            self.instructions = str(s.get("instructions") or "")
        if "tools" in s:
            self.tools = chat_tools(s.get("tools"))
        audio = s.get("audio") or {}
        voice = (audio.get("output") or {}).get("voice", s.get("voice"))
        if voice is not None:
            try:
                self.card = self.gw.directory.resolve(str(voice))
            except LookupError as e:
                # Fail closed: never answer one business's line as a generic agent.
                log.warning("session %s refused: %s", self.id, e)
                self.agent_ok = False
                await self.error("agent_not_found", str(e))
                await self.ws.close(code=1008, reason="agent not found")
                return
        detection = (audio.get("input") or {}).get("turn_detection", s.get("turn_detection"))
        if isinstance(detection, dict) and detection.get("create_response") is False:
            self.creates_response = False
        await self.send({"type": "session.updated", "session": {"id": self.id}})

    async def _on_input_audio_buffer_append(self, event: dict) -> None:
        try:
            pcm24 = base64.b64decode(event.get("audio") or "")
        except Exception:
            return
        utterances, barge = self.seg.feed(self.inbound.feed(pcm24))
        if barge and not self.caller_speaking:
            self.caller_speaking = True
            await self.send({"type": "input_audio_buffer.speech_started"})
            # The caller is talking over the agent: stop the agent.
            await self._cancel_response(announce=True)
        for utter in utterances:
            await self._on_utterance(utter)

    async def _on_utterance(self, pcm16k: bytes) -> None:
        if not self.caller_speaking:
            await self.send({"type": "input_audio_buffer.speech_started"})
        self.caller_speaking = False
        item_id = self._next("item")
        await self.send({"type": "input_audio_buffer.speech_stopped", "item_id": item_id})
        await self.send({"type": "input_audio_buffer.committed", "item_id": item_id})
        loop = asyncio.get_running_loop()
        try:
            text = (await loop.run_in_executor(None, self.gw.stt.transcribe, pcm16k)).strip()
        except Exception as e:
            log.warning("speech-to-text failed: %s", e)
            text = ""
        await self.send({"type": "conversation.item.input_audio_transcription.completed",
                         "item_id": item_id, "transcript": text})
        if not text:
            return
        self.history.append({"role": "user", "content": text})
        if self.creates_response:
            await self._start_response(None)

    async def _on_conversation_item_create(self, event: dict) -> None:
        item = event.get("item") or {}
        if item.get("type") == "function_call_output":
            call_id = str(item.get("call_id") or "")
            if call_id in self.pending_calls:
                self.pending_calls.pop(call_id)
                self.history.append({"role": "tool", "tool_call_id": call_id,
                                     "content": str(item.get("output") or "")})
            else:
                await self.error("unknown_call_id", f"no tool call '{call_id}' is waiting for output")
                return
        elif item.get("type") == "message":
            text = " ".join(str(c.get("text") or "") for c in item.get("content") or []
                            if isinstance(c, dict)).strip()
            role = item.get("role") if item.get("role") in ("user", "system", "assistant") else "user"
            if text:
                self.history.append({"role": role, "content": text})
        item_id = item.get("id") or self._next("item")
        await self.send({"type": "conversation.item.created",
                         "item": {**item, "id": item_id}})

    async def _on_conversation_item_delete(self, event: dict) -> None:
        await self.send({"type": "conversation.item.deleted", "item_id": event.get("item_id")})

    async def _on_conversation_item_truncate(self, event: dict) -> None:
        return        # playback position is the call center's; nothing to rewrite here

    async def _on_response_create(self, event: dict) -> None:
        response = event.get("response") or {}
        if "tool_choice" in response or "tools" in response:
            # Matches the reference gateway the call center's profile was measured against.
            await self.error("unsupported_parameter",
                             "Unsupported parameter: response.tool_choice / response.tools")
            return
        await self._start_response(response.get("instructions"))

    async def _on_response_cancel(self, event: dict) -> None:
        if self.response_task is None or self.response_task.done():
            await self.error("response_cancel_not_active", "no active response to cancel")
            return
        await self._cancel_response(announce=True)

    # -- responses -------------------------------------------------------------
    async def _cancel_response(self, announce: bool) -> None:
        task, self.response_task = self.response_task, None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        if announce:
            await self.send({"type": "response.done",
                             "response": {"id": task.get_name(), "status": "cancelled"}})

    async def _start_response(self, direction: str | None) -> None:
        if self.response_task is not None and not self.response_task.done():
            return                                   # one response at a time
        rid = self._next("resp")
        self.response_task = asyncio.create_task(self._respond(rid, direction), name=rid)

    def _messages(self, direction: str | None) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt()}]
        # Tool outputs that never arrived would make the history invalid for the model.
        open_ids = set(self.pending_calls)
        for m in self.history[-MAX_HISTORY:]:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                if any(c["id"] in open_ids for c in m["tool_calls"]):
                    continue
            messages.append(m)
        if direction:
            messages.append({"role": "system", "content": direction})
        if not any(m["role"] in ("user", "tool") for m in messages):
            # Opening turn: most chat APIs require a non-system message.
            messages.append({"role": "user", "content": "(The call has just connected.)"})
        return messages

    async def _respond(self, rid: str, direction: str | None) -> None:
        loop = asyncio.get_running_loop()
        await self.send({"type": "response.created", "response": {"id": rid}})
        try:
            result = await loop.run_in_executor(
                None, self.gw.chat.complete, self._messages(direction), self.tools)
        except Exception as e:
            log.warning("model call failed: %s", e)
            await self.send({"type": "response.done", "response": {
                "id": rid, "status": "failed",
                "status_details": {"error": {"code": "model_error", "message": type(e).__name__}}}})
            return

        if result.get("tool_calls"):
            calls = result["tool_calls"]
            self.history.append({"role": "assistant", "content": result.get("content") or None,
                                 "tool_calls": [{"id": c["id"], "type": "function",
                                                 "function": {"name": c["name"],
                                                              "arguments": c["arguments"]}}
                                                for c in calls]})
            output = []
            for c in calls:
                item_id = self._next("item")
                self.pending_calls[c["id"]] = c
                await self.send({"type": "response.output_item.added", "response_id": rid,
                                 "item": {"id": item_id, "type": "function_call",
                                          "call_id": c["id"], "name": c["name"]}})
                await self.send({"type": "response.function_call_arguments.done",
                                 "response_id": rid, "item_id": item_id, "call_id": c["id"],
                                 "name": c["name"], "arguments": c["arguments"]})
                output.append({"id": item_id, "type": "function_call", "call_id": c["id"]})
            await self.send({"type": "response.done",
                             "response": {"id": rid, "status": "completed", "output": output}})
            return

        text = (result.get("content") or "").strip()
        item_id = self._next("item")
        await self.send({"type": "response.output_item.added", "response_id": rid,
                         "item": {"id": item_id, "type": "message", "role": "assistant"}})
        if text:
            try:
                pcm48 = await self.gw.tts.synth(text)
            except Exception as e:
                log.warning("text-to-speech failed: %s", e)
                await self.send({"type": "response.done", "response": {
                    "id": rid, "status": "failed",
                    "status_details": {"error": {"code": "tts_error", "message": type(e).__name__}}}})
                return
            from .tts import resample_s16
            pcm24 = await loop.run_in_executor(None, resample_s16, pcm48, TTS_RATE, WIRE_RATE)
            self.history.append({"role": "assistant", "content": text})
            for i in range(0, len(pcm24), DELTA_BYTES):
                await self.send({"type": "response.output_audio.delta", "response_id": rid,
                                 "item_id": item_id,
                                 "delta": base64.b64encode(pcm24[i:i + DELTA_BYTES]).decode()})
                await asyncio.sleep(0.02)      # ~5x real time; leaves room to be interrupted
            await self.send({"type": "response.output_audio_transcript.done",
                             "response_id": rid, "item_id": item_id, "transcript": text})
        await self.send({"type": "response.done", "response": {
            "id": rid, "status": "completed",
            "output": [{"id": item_id, "type": "message"}]}})


# ── server ────────────────────────────────────────────────────────────────────
class Gateway:
    def __init__(self, api_key: str, directory: AgentDirectory, stt, tts, chat):
        if not api_key:
            raise ValueError("an API key is required; the gateway never runs open")
        self.api_key, self.directory = api_key, directory
        self.stt, self.tts, self.chat = stt, tts, chat

    def _authorized(self, headers) -> bool:
        value = headers.get("Authorization", "")
        scheme, _, token = value.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(
            token.strip().encode(), self.api_key.encode())

    async def _process_request(self, connection, request):
        path = request.path.split("?", 1)[0]
        if path == "/health":
            return connection.respond(200, "ok\n")
        if path != "/v1/realtime":
            return connection.respond(404, "not found\n")
        if not self._authorized(request.headers):
            return connection.respond(401, "unauthorized\n")
        return None

    async def _handle(self, ws) -> None:
        session = Session(ws, self)
        log.info("call session %s opened", session.id)
        try:
            await session.run()
        except Exception as e:
            log.info("call session %s ended (%s)", session.id, type(e).__name__)
        else:
            log.info("call session %s closed", session.id)

    async def serve(self, host: str, port: int):
        from websockets.asyncio.server import serve
        return await serve(self._handle, host, port, process_request=self._process_request,
                           max_size=4 * 1024 * 1024)


def main() -> int:
    ap = argparse.ArgumentParser(description="HELIX Realtime Gateway for ai-native-callcenter")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--companies", default=os.environ.get("HELIX_COMPANIES_ROOT", "/etc/helix/companies"),
                    help="directory of <company>/card.json, selected per call by voice 'helix:<ref>'")
    ap.add_argument("--card", default=None,
                    help="card used when a flow names no voice (omit to require helix:<ref>)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    api_key = os.environ.get("HELIX_REALTIME_API_KEY", "")
    if not api_key:
        print("HELIX_REALTIME_API_KEY is required", flush=True)
        return 2
    from .__main__ import _load_card
    from .pipeline import Transcriber
    from .tts import make_synthesizer
    default_card = _load_card(args.card) if args.card else None
    gateway = Gateway(api_key, AgentDirectory(args.companies, default_card),
                      Transcriber(), make_synthesizer(), ChatBackend())

    async def run():
        server = await gateway.serve(args.host, args.port)
        log.info("HELIX Realtime Gateway on ws://%s:%d/v1/realtime (companies: %s)",
                 args.host, args.port, args.companies)
        await server.serve_forever()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
