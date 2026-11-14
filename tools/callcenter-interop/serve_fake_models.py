#!/usr/bin/env python3
"""Run the HELIX Realtime Gateway with scripted speech models, for the interop
check against ai-native-callcenter's own Realtime client (see README.md).

Everything on the wire is the production gateway; only speech-to-text,
the model and text-to-speech are replaced, so the check needs no API keys.
"""
import asyncio
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

from agent_voice.realtime_gateway import AgentDirectory, Gateway


def tone(seconds, rate=48000):
    n = int(seconds * rate)
    return struct.pack(f"<{n}h", *(int(9000 * math.sin(2 * math.pi * 330 * i / rate)) for i in range(n)))


class STT:
    def transcribe(self, pcm16k):
        return "I want to talk to a person"


class TTS:
    async def synth(self, text):
        return tone(0.6)


class Chat:
    def __init__(self):
        self.turn = 0

    def complete(self, messages, tools):
        self.turn += 1
        if messages[-1]["role"] == "user" and "person" in messages[-1]["content"] and tools:
            return {"content": "", "tool_calls": [{"id": "call_helix_1", "name": tools[0]["function"]["name"],
                                                   "arguments": '{"queue":"support"}'}]}
        if messages[-1]["role"] == "tool":
            return {"content": "Connecting you now.", "tool_calls": []}
        return {"content": "Thanks for calling Santino's Pizza.", "tool_calls": []}


async def main(port: int):
    root = Path(tempfile.mkdtemp(prefix="helix-companies-"))
    (root / "santinos").mkdir()
    (root / "santinos" / "card.json").write_text(json.dumps({
        "format": "business-card/0.1", "id": "dna_pizza01",
        "identity": {"name": "Santino's Pizza"}}))
    gw = Gateway("interop-key", AgentDirectory(str(root)), STT(), TTS(), Chat())
    server = await gw.serve("127.0.0.1", port)
    print(f"gateway on ws://127.0.0.1:{port}/v1/realtime", flush=True)
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 18080))
