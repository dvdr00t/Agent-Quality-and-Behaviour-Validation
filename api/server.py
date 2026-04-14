"""
FastAPI HTTP interface for the Retail Support multi-agent system.

Wraps the LangChain multi-agent orchestrator as a REST API so external
evaluation tools (promptfoo, Inspect AI) can target it during pre-deployment
red-teaming.

Endpoints:
  POST   /chat                  — Send a message to the agent
  DELETE /session/{session_id}  — Wipe a session (test isolation)
  GET    /health                — Liveness check

Start:
  uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when the module is started directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retail_support import RetailSupportOrchestrator, SupportSettings, SupportSession

app = FastAPI(
    title="Retail Support Agent API",
    description=(
        "Thin HTTP wrapper around the LangChain multi-agent retail-support system. "
        "Designed for pre-deployment red-team and safety evaluation."
    ),
    version="1.0.0",
)

_orchestrator: RetailSupportOrchestrator | None = None

# Simple in-memory session store — supports multi-turn crescendo attacks
_sessions: dict[str, SupportSession] = {}


def _get_orchestrator() -> RetailSupportOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        settings = SupportSettings.from_env()
        _orchestrator = RetailSupportOrchestrator(settings)
    return _orchestrator


# ── Request / response models ─────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    user_id: str = "user_001"
    agent: str = "supervisor"
    # Pass the same session_id across turns to simulate multi-turn attacks
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    handled_by: str
    tool_calls: list[str]
    route: list[str]
    session_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Forward a message to the retail support agent.

    Provide a ``session_id`` to persist conversation history across multiple
    turns, which is required for crescendo-style multi-turn attack testing in
    promptfoo. Omit it (or use a fresh ID) for stateless single-turn probes.
    """
    orchestrator = _get_orchestrator()

    # Reuse an existing session or start a fresh one
    if request.session_id and request.session_id in _sessions:
        session = _sessions[request.session_id]
    else:
        session = orchestrator.start_session()
        if request.session_id:
            _sessions[request.session_id] = session

    try:
        reply = orchestrator.reply(
            session=session,
            user_message=request.message,
            target=request.agent,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        response=reply.text,
        handled_by=reply.handled_by,
        tool_calls=reply.tool_calls,
        route=reply.route,
        session_id=request.session_id,
    )


@app.delete("/session/{session_id}", status_code=204)
def clear_session(session_id: str) -> None:
    """
    Remove a stored session to reset conversation history between independent
    test cases. The evaluation runner calls this between non-sequential probes.
    """
    _sessions.pop(session_id, None)


@app.get("/health")
def health() -> dict:
    """Liveness check consumed by the runner script before starting evaluations."""
    return {"status": "ok", "active_sessions": len(_sessions)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
