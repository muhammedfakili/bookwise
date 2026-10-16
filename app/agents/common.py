"""What both agents share: the reply type and how a transcript is shown to the LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class AgentReply:
    text: str
    action: str  # what the agent did this turn: ask, offer_slots, book, handoff, ...
    events: list[str] = field(default_factory=list)  # tool calls, for the UI and the eval log
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Agent(Protocol):
    name: str

    def respond(self, conversation_id: str, channel: str) -> AgentReply:
        """Answer the customer's latest message (already saved in the transcript)."""
        ...


ROLE_LABELS = {"customer": "Customer", "assistant": "Assistant", "human_agent": "Team member", "system": "System"}


def render_transcript(messages: list[dict]) -> str:
    return "\n".join(f"{ROLE_LABELS[m['role']]}: {m['text']}" for m in messages) or "(no messages yet)"
