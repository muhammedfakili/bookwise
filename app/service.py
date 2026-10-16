"""One entry point for every channel: save the message, let the bot or a human answer.

    customer message -> transcript -> open handoff?  yes -> wait for the team (bot stays silent)
                                                     no  -> agent.respond() -> transcript -> channel

If the model fails after its retries, the conversation goes to a human instead of
leaving the customer without an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.common import Agent, AgentReply
from app.deps import Deps
from app.llm import LLM_ERRORS
from app.ports import ChannelPort

ERROR_REPLY = ("Sorry, I'm having a technical problem right now. I've passed your message to our team "
               "and someone will reply here shortly.")


@dataclass
class TurnResult:
    conversation_id: str
    reply: str | None  # None while a human owns the conversation
    mode: str  # "bot" or "human"
    action: str
    events: list[str] = field(default_factory=list)
    lead: dict | None = None
    llm_calls: int = 0


class ChatService:
    def __init__(self, deps: Deps, agent: Agent, channels: dict[str, ChannelPort] | None = None,
                 handoff_on_error: bool = True) -> None:
        self.deps = deps
        self.agent = agent
        self.channels = channels or {}
        # The evaluation turns this off: a provider outage is not the agent's decision to hand off.
        self.handoff_on_error = handoff_on_error

    def handle(self, conversation_id: str, text: str, channel: str = "web") -> TurnResult:
        deps = self.deps
        text = text.strip()
        if not text:
            raise ValueError("empty message")
        deps.transcripts.ensure_conversation(conversation_id, channel)
        deps.transcripts.add(conversation_id, "customer", text)

        if deps.handoffs.active(conversation_id):
            return TurnResult(conversation_id, None, "human", "waiting_for_human",
                              lead=deps.crm.get_lead(conversation_id))
        try:
            reply = self.agent.respond(conversation_id, channel)
        except LLM_ERRORS as exc:
            if not self.handoff_on_error:
                raise
            reason = f"assistant error: {type(exc).__name__}"
            deps.handoffs.open(conversation_id, reason, f"Reason: {reason}. Model call failed: {exc}"[:500])
            deps.crm.upsert_lead(conversation_id, channel, {"status": "handoff", "status_reason": reason})
            reply = AgentReply(ERROR_REPLY, "handoff", [f"error: {type(exc).__name__}"])
        if not reply.text:
            reply.text = "Sorry, could you say that again?"
        deps.transcripts.add(conversation_id, "assistant", reply.text)
        self._deliver(conversation_id, channel, reply.text)
        mode = "human" if deps.handoffs.active(conversation_id) else "bot"
        return TurnResult(conversation_id, reply.text, mode, reply.action, reply.events,
                          deps.crm.get_lead(conversation_id), reply.llm_calls)

    def human_reply(self, handoff_id: int, text: str) -> None:
        """A team member answers inside the conversation (from the admin screen)."""
        handoff = self.deps.handoffs.get(handoff_id)
        if handoff is None:
            raise KeyError(handoff_id)
        cid = handoff["conversation_id"]
        self.deps.transcripts.add(cid, "human_agent", text.strip())
        channel = next((c["channel"] for c in self.deps.transcripts.conversations() if c["id"] == cid), "web")
        self._deliver(cid, channel, text.strip())

    def resolve(self, handoff_id: int) -> None:
        """Close the handoff; the bot answers again from the next message."""
        if self.deps.handoffs.get(handoff_id) is None:
            raise KeyError(handoff_id)
        self.deps.handoffs.resolve(handoff_id)

    def _deliver(self, conversation_id: str, channel: str, text: str) -> None:
        sender = self.channels.get(channel)
        if sender is not None:
            # WhatsApp conversations are keyed "wa:<phone>"; the channel wants the bare number.
            sender.send(conversation_id.removeprefix("wa:"), text)
