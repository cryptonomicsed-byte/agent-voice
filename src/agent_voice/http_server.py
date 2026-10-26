"""HTTP chat bridge for agent-voice.

Exposes a simple REST interface so helixmain (Fastify) can reach the
card-grounded agent without WebRTC.  Activated by AGENT_VOICE_HTTP_PORT.

Routes:
    GET  /health
    POST /chat   — {message, session_id?, history?} → {reply, session_id, history}
    GET  /prompt — returns the compiled system prompt + opening line for this card
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

log = logging.getLogger(__name__)

# In-memory session store: session_id → list of {role, content} messages
_sessions: dict[str, list[dict[str, str]]] = {}

# Lazily initialised — set by start_http_server before uvicorn starts
_card: dict[str, Any] = {}
_agent_backend: str = "deepseek"


def _get_app():
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError as e:
        raise RuntimeError("fastapi + pydantic required for the HTTP bridge") from e

    from .pipeline import build_voice_system_prompt, opening_line, CardGroundedAgent

    app = FastAPI(title="agent-voice HTTP bridge", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Pre-build the agent once at startup — CardGroundedAgent.__init__ is cheap
    _agent_cache: dict[str, CardGroundedAgent] = {}

    def _agent() -> CardGroundedAgent:
        key = _agent_backend
        if key not in _agent_cache:
            _agent_cache[key] = CardGroundedAgent(
                bot_name="Helix",
                backend=_agent_backend,
                card=_card or None,
            )
        return _agent_cache[key]

    class ChatRequest(BaseModel):
        message: str
        session_id: str | None = None
        history: list[dict[str, str]] | None = None

    class ChatResponse(BaseModel):
        reply: str
        session_id: str
        history: list[dict[str, str]]

    @app.get("/health")
    def health():
        card_id = (_card or {}).get("id") or (_card or {}).get("organizationId")
        return {"status": "ok", "card_id": card_id, "backend": _agent_backend}

    @app.get("/prompt")
    def prompt():
        system = build_voice_system_prompt(_card) if _card else ""
        greeting = opening_line(_card) if _card else None
        return {"system_prompt": system, "opening_line": greeting}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        sid = req.session_id or str(uuid.uuid4())
        history: list[dict[str, str]] = req.history or _sessions.get(sid, [])

        # Build messages list for a multi-turn reply — CardGroundedAgent.reply()
        # only takes the latest user message and uses its own system prompt, so
        # we prepend prior turns as a single context block when history is long.
        user_msg = req.message
        if history:
            prior = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
            )
            user_msg = f"[Prior conversation]\n{prior}\n\nCALLER: {req.message}"

        agent = _agent()
        loop = asyncio.get_running_loop()
        reply_text = await loop.run_in_executor(None, agent.reply, user_msg)

        history = list(history) + [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": reply_text},
        ]
        _sessions[sid] = history[-20:]  # keep last 10 turns

        return ChatResponse(reply=reply_text, session_id=sid, history=history)

    return app


async def start_http_server(port: int, card: dict, backend: str = "deepseek") -> None:
    """Start the FastAPI server on *port*.  Returns when the server stops."""
    global _card, _agent_backend
    _card = card
    _agent_backend = backend

    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError("uvicorn required — pip install uvicorn") from e

    app = _get_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    log.info("HTTP bridge listening on http://127.0.0.1:%d", port)
    await server.serve()
