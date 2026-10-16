"""HTTP API: the web chat, the WhatsApp webhook and the admin view all go through here.

    GET  /health                          provider/model/agent, ready or not
    POST /chat                            {conversation_id?, message} -> reply, action, lead
    GET  /conversations/{id}              transcript, lead, bookings, handoffs, agent state
    GET  /webhook/whatsapp                Meta's verification handshake
    POST /webhook/whatsapp                Cloud API webhook -> agent -> reply via the channel
    GET  /admin/leads | /admin/appointments | /admin/handoffs | /admin/conversations
    POST /admin/handoffs/{id}/reply       a team member answers in the conversation
    POST /admin/handoffs/{id}/resolve     give the conversation back to the bot
    GET  /admin/whatsapp/outbox           what the mock WhatsApp sender would have sent

If APP_API_KEY is set, every route except /health and the webhook needs header X-API-Key.

Run:  uv run uvicorn app.api:app --port 8000
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.agents.baseline_agent import BaselineAgent
from app.agents.graph_agent import GraphAgent
from app.channels import MockWhatsApp, parse_whatsapp_webhook
from app.config import settings
from app.deps import build_deps
from app.llm import ProviderNotConfigured
from app.service import ChatService

app = FastAPI(title="LeadPilot", version="1.0",
              description="Lead qualification and booking agent with human handoff")


@lru_cache(maxsize=1)
def get_service() -> ChatService:
    deps = build_deps()
    agent = BaselineAgent(deps) if settings.agent == "baseline" else GraphAgent(deps)
    return ChatService(deps, agent, channels={"whatsapp": MockWhatsApp()})


def service() -> ChatService:
    try:
        return get_service()
    except ProviderNotConfigured as exc:
        raise HTTPException(503, str(exc)) from exc


def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.api_key_required and x_api_key != settings.app_api_key:
        raise HTTPException(401, "Missing or wrong X-API-Key")


Svc = Annotated[ChatService, Depends(service)]
Auth = [Depends(require_key)]


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(None, max_length=100)


class ChatOut(BaseModel):
    conversation_id: str
    reply: str | None
    mode: str
    action: str
    events: list[str]
    lead: dict | None
    llm_calls: int


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict:
    try:
        svc = get_service()
    except ProviderNotConfigured as exc:
        return {"status": "not_configured", "detail": str(exc)}
    return {"status": "ok", "agent": svc.agent.name, "provider": svc.deps.llm.name, "model": svc.deps.llm.model,
            "clock": svc.deps.now().isoformat(timespec="minutes")}


# Plain `def`: these call the LLM (blocking), so FastAPI runs them in its thread pool.
@app.post("/chat", dependencies=Auth)
def chat(body: ChatIn, svc: Svc) -> ChatOut:
    cid = body.conversation_id or f"web:{uuid.uuid4().hex[:12]}"
    if cid.startswith("wa:"):
        raise HTTPException(400, "wa: conversations belong to the WhatsApp channel")
    result = svc.handle(cid, body.message, "web")
    return ChatOut(**result.__dict__)


@app.get("/conversations/{conversation_id}", dependencies=Auth)
def conversation(conversation_id: str, svc: Svc) -> dict:
    deps = svc.deps
    messages = deps.transcripts.history(conversation_id)
    if not messages:
        raise HTTPException(404, "No such conversation")
    state = svc.agent.state(conversation_id) if isinstance(svc.agent, GraphAgent) else {}
    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "lead": deps.crm.get_lead(conversation_id),
        "appointments": deps.calendar.for_conversation(conversation_id),
        "handoffs": deps.handoffs.for_conversation(conversation_id),
        "mode": "human" if deps.handoffs.active(conversation_id) else "bot",
        "agent_state": {k: state.get(k) for k in ("stage", "lead", "asked", "offered", "offer_rounds",
                                                  "action", "analysis") if k in state},
    }


@app.get("/webhook/whatsapp", response_class=PlainTextResponse)
def whatsapp_verify(
    mode: Annotated[str, Query(alias="hub.mode")],
    token: Annotated[str, Query(alias="hub.verify_token")],
    challenge: Annotated[str, Query(alias="hub.challenge")],
) -> str:
    if mode != "subscribe" or token != settings.whatsapp_verify_token:
        raise HTTPException(403, "Verification failed")
    return challenge


@app.post("/webhook/whatsapp")
def whatsapp_webhook(body: dict, tasks: BackgroundTasks, svc: Svc) -> dict:
    # Meta wants a fast 200; the agent answers afterwards and replies through the channel.
    messages = parse_whatsapp_webhook(body)
    for msg in messages:
        tasks.add_task(svc.handle, f"wa:{msg.sender}", msg.text, "whatsapp")
    return {"received": len(messages)}


@app.get("/admin/leads", dependencies=Auth)
def leads(svc: Svc) -> list[dict]:
    return svc.deps.crm.leads()


@app.get("/admin/appointments", dependencies=Auth)
def appointments(svc: Svc) -> list[dict]:
    return svc.deps.calendar.appointments()


@app.get("/admin/handoffs", dependencies=Auth)
def handoffs(svc: Svc) -> list[dict]:
    return svc.deps.handoffs.all()


@app.get("/admin/conversations", dependencies=Auth)
def conversations(svc: Svc) -> list[dict]:
    return svc.deps.transcripts.conversations()


@app.post("/admin/handoffs/{handoff_id}/reply", dependencies=Auth)
def handoff_reply(handoff_id: int, body: TextIn, svc: Svc) -> dict:
    try:
        svc.human_reply(handoff_id, body.text)
    except KeyError as exc:
        raise HTTPException(404, "No such handoff") from exc
    return {"ok": True}


@app.post("/admin/handoffs/{handoff_id}/resolve", dependencies=Auth)
def handoff_resolve(handoff_id: int, svc: Svc) -> dict:
    try:
        svc.resolve(handoff_id)
    except KeyError as exc:
        raise HTTPException(404, "No such handoff") from exc
    return {"ok": True}


@app.get("/admin/whatsapp/outbox", dependencies=Auth)
def whatsapp_outbox(svc: Svc) -> list[dict]:
    sender = svc.channels.get("whatsapp")
    return list(getattr(sender, "outbox", []))
