"""HELIX Realtime Gateway: the protocol the call center's `gateway` provider speaks.

A WebSocket client plays the call center: it authenticates, configures the
session the way internal/provider/realtime.go does, sends caller audio as
24 kHz PCM16 and reads the events back. Speech models are faked; the protocol
handling, agent selection, tool-call round trip and audio path are real.
"""
import asyncio
import base64
import json
import math
import struct

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from agent_voice.realtime_gateway import AgentDirectory, Gateway, chat_tools

KEY = "test-key"
CARD = {"format": "business-card/0.1", "id": "dna_pizza01", "organization_id": "org_1",
        "identity": {"name": "Santino's Pizza"}, "offerings": [{"name": "Cheese Pizza"}]}
OTHER = {"format": "business-card/0.1", "id": "dna_barber01", "identity": {"name": "King's Cut"}}


def tone(seconds, rate, amp=12000):
    n = int(seconds * rate)
    return struct.pack(f"<{n}h", *(int(amp * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)))


class FakeSTT:
    def transcribe(self, pcm16k):
        return "do you have cheese pizza"


class FakeTTS:
    async def synth(self, text):
        return tone(0.5, 48000)


class FakeChat:
    """Records what the model was asked; scripted replies."""

    def __init__(self, script=None):
        self.calls, self.script = [], list(script or [])

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        if self.script:
            return self.script.pop(0)
        return {"content": "Yes, we do.", "tool_calls": []}


@pytest.fixture
def companies(tmp_path):
    for slug, card in (("santinos", CARD), ("kings-cut", OTHER)):
        (tmp_path / slug).mkdir()
        (tmp_path / slug / "card.json").write_text(json.dumps(card))
    return tmp_path


async def start(companies, chat=None, default_card=None):
    chat = chat or FakeChat()
    gw = Gateway(KEY, AgentDirectory(str(companies), default_card), FakeSTT(), FakeTTS(), chat)
    server = await gw.serve("127.0.0.1", 0)
    return server, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime?model=cascade", chat


def session_update(voice, instructions="Greet the caller.", tools=None):
    """The GA-style session.update the call center sends for the gateway profile."""
    session = {"type": "realtime", "instructions": instructions,
               "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                   "turn_detection": {"type": "server_vad"}},
                         "output": {"format": {"type": "audio/pcm", "rate": 24000}}}}
    if voice:
        session["audio"]["output"]["voice"] = voice
    if tools:
        session["tools"] = tools
    return {"type": "session.update", "session": session}


async def until(ws, kind, collect=None):
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if collect is not None:
            collect.append(ev)
        if ev["type"] == kind:
            return ev


async def say(ws, seconds=1.0):
    """Caller speech then silence, in 20 ms frames like the call center sends."""
    pcm = tone(seconds, 24000) + b"\x00" * int(24000 * 2 * 1.0)
    for i in range(0, len(pcm), 960):
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(pcm[i:i + 960]).decode()}))


def auth():
    return {"Authorization": f"Bearer {KEY}"}


async def test_rejects_missing_or_wrong_key(companies):
    server, url, _ = await start(companies)
    try:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            with pytest.raises(InvalidStatus) as e:
                async with connect(url, additional_headers=headers):
                    pass
            assert e.value.response.status_code == 401
    finally:
        server.close()


async def test_flow_voice_selects_the_business_and_grounds_the_call(companies):
    server, url, chat = await start(companies)
    try:
        async with connect(url, additional_headers=auth()) as ws:
            assert (await until(ws, "session.created"))["session"]["id"]
            await ws.send(json.dumps(session_update("helix:dna_pizza01")))
            await until(ws, "session.updated")
            await ws.send(json.dumps({"type": "response.create"}))       # opening turn
            events = []
            done = await until(ws, "response.done", events)
            assert done["response"]["status"] == "completed"
            kinds = [e["type"] for e in events]
            assert kinds[0] == "response.created"
            assert "response.output_item.added" in kinds
            audio = b"".join(base64.b64decode(e["delta"]) for e in events
                             if e["type"] == "response.output_audio.delta")
            assert abs(len(audio) - 24000 * 2 * 0.5) < 2000          # 0.5 s at 24 kHz
            system = chat.calls[0][0][0]["content"]
            assert "Santino's Pizza" in system and "King's Cut" not in system
            assert "Greet the caller." in system
    finally:
        server.close()


async def test_company_directory_name_also_selects(companies):
    server, url, chat = await start(companies)
    try:
        async with connect(url, additional_headers=auth()) as ws:
            await until(ws, "session.created")
            await ws.send(json.dumps(session_update("helix:kings-cut")))
            await until(ws, "session.updated")
            await ws.send(json.dumps({"type": "response.create"}))
            await until(ws, "response.done")
            assert "King's Cut" in chat.calls[0][0][0]["content"]
    finally:
        server.close()


@pytest.mark.parametrize("voice", ["helix:dna_unknown", "alloy", "helix:../etc"])
async def test_unknown_agent_is_refused_not_answered_generically(companies, voice):
    server, url, chat = await start(companies)
    try:
        async with connect(url, additional_headers=auth()) as ws:
            await until(ws, "session.created")
            await ws.send(json.dumps(session_update(voice)))
            err = await until(ws, "error")
            assert err["error"]["code"] == "agent_not_found"
            with pytest.raises(ConnectionClosed):
                while True:
                    await asyncio.wait_for(ws.recv(), 5)
            assert chat.calls == []
    finally:
        server.close()


async def test_caller_speech_is_transcribed_and_answered(companies):
    server, url, chat = await start(companies)
    try:
        async with connect(url, additional_headers=auth()) as ws:
            await until(ws, "session.created")
            await ws.send(json.dumps(session_update("helix:dna_pizza01")))
            await until(ws, "session.updated")
            await say(ws)
            events = []
            await until(ws, "response.done", events)
            kinds = [e["type"] for e in events]
            for expected in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                             "input_audio_buffer.committed",
                             "conversation.item.input_audio_transcription.completed",
                             "response.created", "response.output_audio.delta",
                             "response.output_audio_transcript.done"):
                assert expected in kinds, expected
            heard = next(e for e in events
                         if e["type"] == "conversation.item.input_audio_transcription.completed")
            assert heard["transcript"] == "do you have cheese pizza"
            assert chat.calls[-1][0][-1] == {"role": "user", "content": "do you have cheese pizza"}
    finally:
        server.close()


async def test_tool_call_goes_to_the_call_center_and_its_result_comes_back(companies):
    script = [{"content": "", "tool_calls": [
                  {"id": "call_1", "name": "transfer_to_agent", "arguments": '{"queue":"support"}'}]},
              {"content": "Connecting you now.", "tool_calls": []}]
    server, url, chat = await start(companies, FakeChat(script))
    tools = [{"type": "function", "name": "transfer_to_agent", "description": "Hand over to a person",
              "parameters": {"type": "object", "properties": {"queue": {"type": "string"}}}}]
    try:
        async with connect(url, additional_headers=auth()) as ws:
            await until(ws, "session.created")
            await ws.send(json.dumps(session_update("helix:dna_pizza01", tools=tools)))
            await until(ws, "session.updated")
            await say(ws)
            call = await until(ws, "response.function_call_arguments.done")
            assert (call["call_id"], call["name"]) == ("call_1", "transfer_to_agent")
            assert json.loads(call["arguments"]) == {"queue": "support"}
            done = await until(ws, "response.done")
            assert done["response"]["output"][0]["type"] == "function_call"
            assert chat.calls[0][1] == chat_tools(tools)

            # the call center runs the tool and answers, then asks for the next turn
            await ws.send(json.dumps({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": "call_1", "output": '{"ok":true}'}}))
            await ws.send(json.dumps({"type": "response.create"}))
            said = await until(ws, "response.output_audio_transcript.done")
            assert said["transcript"] == "Connecting you now."
            roles = [m["role"] for m in chat.calls[1][0]]
            assert roles[-2:] == ["assistant", "tool"]
            assert chat.calls[1][0][-1]["tool_call_id"] == "call_1"
    finally:
        server.close()


async def test_spoken_line_direction_and_refused_parameters(companies):
    server, url, chat = await start(companies)
    try:
        async with connect(url, additional_headers=auth()) as ws:
            await until(ws, "session.created")
            await ws.send(json.dumps(session_update("helix:dna_pizza01")))
            await until(ws, "session.updated")
            await ws.send(json.dumps({"type": "response.create",
                                      "response": {"instructions": "Say exactly: please hold."}}))
            await until(ws, "response.done")
            assert chat.calls[0][0][-2] == {"role": "system", "content": "Say exactly: please hold."}
            await ws.send(json.dumps({"type": "response.create",
                                      "response": {"instructions": "x", "tool_choice": "none"}}))
            assert "Unsupported parameter" in (await until(ws, "error"))["error"]["message"]
            await ws.send(json.dumps({"type": "response.cancel"}))
            assert (await until(ws, "error"))["error"]["code"] == "response_cancel_not_active"
    finally:
        server.close()


async def test_caller_talking_over_the_agent_cancels_the_response(companies):
    class SlowTTS:
        async def synth(self, text):
            return tone(6.0, 48000)

    chat = FakeChat()
    gw = Gateway(KEY, AgentDirectory(str(companies)), FakeSTT(), SlowTTS(), chat)
    server = await gw.serve("127.0.0.1", 0)
    url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
    try:
        async with connect(url, additional_headers=auth()) as ws:
            await until(ws, "session.created")
            await ws.send(json.dumps(session_update("helix:dna_pizza01")))
            await until(ws, "session.updated")
            await ws.send(json.dumps({"type": "response.create"}))
            await until(ws, "response.output_audio.delta")
            await say(ws, 0.5)
            events = []
            done = await until(ws, "response.done", events)
            assert done["response"]["status"] == "cancelled"
            assert "input_audio_buffer.speech_started" in [e["type"] for e in events]
    finally:
        server.close()


def test_gateway_never_runs_without_a_key(companies):
    with pytest.raises(ValueError):
        Gateway("", AgentDirectory(str(companies)), FakeSTT(), FakeTTS(), FakeChat())
