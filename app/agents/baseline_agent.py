"""The comparison agent: one prompt with the rules, free native tool calling, no graph.

This is how most "AI booking agent" demos are built: describe the policy in the
system prompt, give the model tools, let it decide. It shares the calendar, CRM and
handoff queue with the graph agent (so the database still refuses double bookings),
but whether to book, disqualify or hand off is entirely the model's call. The
evaluation runs both agents against the same simulated customers.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

from app import business
from app.agents.common import AgentReply
from app.deps import Deps
from app.llm import Msg, ToolCall, ToolSpec
from app.ports import Slot, SlotUnavailable

MAX_TOOL_ROUNDS = 6

SYSTEM = f"""\
You are {business.ASSISTANT_NAME}, the chat assistant of {business.NAME}.

{business.FACTS}

Your job: qualify the lead, then book a free in-home design consultation if they qualify.
A lead qualifies only if ALL of these hold:
- the project is a kitchen remodel, a bathroom remodel, or both;
- the property ZIP code starts with 786 or 787;
- they want to start within {business.MAX_START_WEEKS} weeks;
- the budget is at least ${business.MIN_BUDGET_USD['kitchen']:,} for a kitchen, \
${business.MIN_BUDGET_USD['bathroom']:,} for a bathroom, ${business.MIN_BUDGET_USD['kitchen_and_bathroom']:,} for both;
- they own the home and make the decision.
Ask for missing facts one question at a time. If a rule fails, politely decline and save the lead \
as disqualified. If they qualify, get their full name and phone number, call check_availability, \
offer up to 3 times, and book the one they pick with book_consultation.
Save what you learn with save_lead. Hand off to a human (hand_off_to_human) for complaints, angry \
customers, requests for a person, exact quotes, discounts, contract or warranty questions, or \
reschedule/cancel requests, or when the customer still won't give a required fact after you asked \
twice. Never make up times; only offer times returned by check_availability.
Be warm and brief (at most 3 sentences), plain text, one question at a time.
"""

TOOLS = [
    ToolSpec("check_availability", "Free consultation times (90 min, weekdays).", {
        "type": "object", "properties": {
            "part_of_day": {"type": "string", "enum": ["morning", "afternoon"]},
            "weekdays": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['tuesday']"},
        }}),
    ToolSpec("book_consultation", "Book a consultation time returned by check_availability.", {
        "type": "object", "properties": {
            "start": {"type": "string", "description": "ISO start time exactly as returned, e.g. 2026-10-20T09:00:00"},
            "name": {"type": "string"}, "phone": {"type": "string"},
        }, "required": ["start", "name", "phone"]}),
    ToolSpec("save_lead", "Create or update the lead in the CRM.", {
        "type": "object", "properties": {
            "name": {"type": "string"}, "phone": {"type": "string"}, "zip_code": {"type": "string"},
            "service": {"type": "string", "enum": ["kitchen", "bathroom", "kitchen_and_bathroom", "other"]},
            "budget_usd": {"type": "integer"}, "timeline_weeks": {"type": "integer"},
            "decision_maker": {"type": "boolean"},
            "status": {"type": "string", "enum": ["in_progress", "qualified", "disqualified"]},
            "status_reason": {"type": "string"},
        }, "required": ["status"]}),
    ToolSpec("hand_off_to_human", "Pass the conversation to a team member.", {
        "type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}),
]


def _to_dict(m: Msg) -> dict:
    d = asdict(m)
    d["tool_calls"] = [asdict(c) for c in m.tool_calls]
    return d


def _from_dict(d: dict) -> Msg:
    return Msg(**{**d, "tool_calls": [ToolCall(**c) for c in d["tool_calls"]]})


class BaselineAgent:
    name = "baseline"

    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    def respond(self, conversation_id: str, channel: str) -> AgentReply:
        deps = self.deps
        thread = [_from_dict(d) for d in deps.threads.load(conversation_id)]
        latest = deps.transcripts.history(conversation_id, 1)
        thread.append(Msg("user", latest[-1]["text"] if latest else ""))
        system = SYSTEM + f"\nToday is {deps.now():%A %Y-%m-%d}."
        reply = AgentReply("", "reply")
        for _ in range(MAX_TOOL_ROUNDS):
            turn = deps.llm.chat_tools(system, thread, TOOLS)
            reply.llm_calls += 1
            reply.input_tokens += turn.input_tokens
            reply.output_tokens += turn.output_tokens
            thread.append(turn.as_message())
            if not turn.tool_calls:
                reply.text = turn.text.strip()
                break
            for call in turn.tool_calls:
                result = self._run_tool(conversation_id, channel, call, reply)
                thread.append(Msg("tool", json.dumps(result), tool_call_id=call.id, tool_name=call.name))
        else:
            reply.text = "Sorry, something went wrong on my side. A team member will follow up shortly."
        deps.threads.save(conversation_id, [_to_dict(m) for m in thread])
        return reply

    def _run_tool(self, conversation_id: str, channel: str, call: ToolCall, reply: AgentReply) -> dict:
        deps, args = self.deps, call.args
        reply.events.append(f"{call.name}({json.dumps(args)})")
        if call.name == "check_availability":
            slots = deps.calendar.find_free_slots(3, args.get("part_of_day"), args.get("weekdays"))
            return {"slots": [{"start": s.start.isoformat(), "label": Slot.label(s)} for s in slots]}
        if call.name == "book_consultation":
            try:
                start = datetime.fromisoformat(str(args.get("start")))
                appt = deps.calendar.book(start, conversation_id, str(args.get("name", "")),
                                          str(args.get("phone", "")))
            except (ValueError, SlotUnavailable) as exc:
                return {"ok": False, "error": str(exc)}
            deps.crm.upsert_lead(conversation_id, channel, {"status": "booked",
                                                            "status_reason": f"consultation #{appt.id}"})
            reply.action = "book"
            return {"ok": True, "appointment_id": appt.id, "start": appt.start.isoformat()}
        if call.name == "save_lead":
            fields = {k: v for k, v in args.items() if k in (
                "name", "phone", "zip_code", "service", "budget_usd", "timeline_weeks", "decision_maker",
                "status", "status_reason")}
            lead = deps.crm.get_lead(conversation_id)
            if lead and lead["status"] in ("booked", "handoff") and "status" in fields:
                fields.pop("status")  # never downgrade a booked lead
            deps.crm.upsert_lead(conversation_id, channel, fields)
            if fields.get("status") == "disqualified":
                reply.action = "disqualify"
            return {"ok": True}
        if call.name == "hand_off_to_human":
            reason = str(args.get("reason", "requested by assistant"))
            hid = deps.handoffs.open(conversation_id, reason, f"Reason: {reason}.")
            deps.crm.upsert_lead(conversation_id, channel, {"status": "handoff", "status_reason": reason})
            reply.action = "handoff"
            return {"ok": True, "handoff_id": hid}
        return {"ok": False, "error": f"unknown tool {call.name}"}
