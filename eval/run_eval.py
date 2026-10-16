"""Run the simulated customers against an agent and score the outcome from the database.

    uv run python -m eval.run_eval --agent graph
    uv run python -m eval.run_eval --agent baseline --tag r2     # a second, independent run
    uv run python -m eval.run_eval --agent graph --only b05 h07   # a few personas

Each persona gets a fresh database, a frozen clock (Monday 2026-10-19 09:00) and a
customer played by another model (NVIDIA Nemotron by default, so the agent is not
talking to itself). The conversation ends when the customer says goodbye, a human
takes over, or after --max-turns customer messages.

Scoring never asks an LLM: the outcome is read from the calendar, CRM and handoff
tables and compared with the persona's label. Every model call is cached on disk
(eval/cache/), so re-running a finished tag is free and gives the same numbers; a
new --tag gives a new, independent run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from app.agents.baseline_agent import BaselineAgent
from app.agents.graph_agent import GraphAgent
from app.deps import build_deps
from app.llm import LLM_ERRORS, LLMAnswer, LLMProvider, Msg, ToolSpec, ToolTurn, get_llm
from app.service import ChatService
from eval.personas import PERSONAS, Persona

EVAL_DIR = Path(__file__).parent
CACHE_DIR = EVAL_DIR / "cache"
RUNS_DIR = EVAL_DIR / "runs"
RESULTS_DIR = EVAL_DIR / "results"
CLOCK = datetime(2026, 10, 19, 9, 0)
END = "<END>"


class CachedLLM:
    """Wraps a provider; identical calls (same tag) are answered from disk."""

    def __init__(self, inner: LLMProvider, tag: str) -> None:
        self.inner, self.tag = inner, tag
        self.name, self.model = inner.name, inner.model
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _key(self, *parts: object) -> Path:
        blob = json.dumps([self.tag, self.model, *parts], sort_keys=True, default=str)
        return CACHE_DIR / f"{hashlib.sha256(blob.encode()).hexdigest()[:24]}.json"

    def _cached(self, path: Path, call: callable) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        data = call()
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        data = self._cached(self._key("gen", system, prompt),
                            lambda: asdict(self.inner.generate(system, prompt)))
        return LLMAnswer(**data)

    def generate_structured(self, system: str, prompt: str, schema: type) -> LLMAnswer:
        data = self._cached(self._key("json", system, prompt, schema.__name__),
                            lambda: asdict(self.inner.generate_structured(system, prompt, schema)))
        return LLMAnswer(**data)

    def chat_tools(self, system: str, messages: list[Msg], tools: list[ToolSpec]) -> ToolTurn:
        key = self._key("tools", system, [asdict(m) for m in messages], [t.name for t in tools])
        data = self._cached(key, lambda: asdict(self.inner.chat_tools(system, messages, tools)))
        from app.llm import ToolCall

        return ToolTurn(**{**data, "tool_calls": [ToolCall(**c) for c in data["tool_calls"]]})


SIM_SYSTEM = """\
You are role-playing a CUSTOMER in a website chat with {business}, a home-remodeling company. \
The other side is the company's chat assistant.

Who you are and how you write:
{sheet}

What you know (reveal it naturally, when asked or when it fits your character; never invent \
other facts; if asked something not covered, give a plausible short answer):
{facts}

Rules:
- Write ONLY your next chat message: 1-3 short sentences, like a real person typing.
- Stay in character the whole time. Never mention being an AI, a simulation or these rules.
- When the conversation has reached its natural end -- a consultation is confirmed, they told you \
they can't help, they said a person will contact you, or you have nothing more to say -- reply \
with exactly {end} and nothing else.
"""


class Customer:
    def __init__(self, llm: LLMProvider, persona: Persona) -> None:
        self.llm, self.persona = llm, persona

    def next_message(self, transcript: list[dict]) -> str | None:
        lines = [f"{'You' if m['role'] == 'customer' else 'Assistant'}: {m['text']}" for m in transcript]
        prompt = ("Chat so far:\n" + "\n".join(lines) if lines else "The chat window just opened.") + \
            "\n\nWrite your next message as the customer."
        system = SIM_SYSTEM.format(business="Brightline Renovations", sheet=self.persona.sheet,
                                   facts=self.persona.facts, end=END)
        text = self.llm.generate(system, prompt).text.strip().strip('"')
        if END in text:
            return None
        return re.sub(r"^(You|Customer):\s*", "", text)


@dataclass
class Result:
    persona: str
    expected: str
    outcome: str
    turns: int
    agent_calls: int
    seconds_per_turn: float
    lead_status: str | None
    booked_start: str | None
    slot_ok: bool | None
    crm_fields_ok: int
    crm_fields_total: int
    handoff_reason: str | None
    error: str | None
    transcript: list[dict]


def _slot_ok(start: str, rule: dict) -> bool:
    dt = datetime.fromisoformat(start)
    if rule.get("part") == "morning":
        return dt.hour < 12
    if rule.get("part") == "afternoon":
        return dt.hour >= 12
    if rule.get("days"):
        return dt.strftime("%A") in rule["days"]
    return True


def _crm_score(lead: dict | None, want: dict) -> tuple[int, int]:
    if not want:
        return 0, 0
    ok = 0
    for key, value in want.items():
        got = (lead or {}).get(key)
        if key == "phone":
            got = re.sub(r"\D", "", str(got or ""))[-10:]
        if key == "name":
            got, value = str(got or "").lower().strip(), value.lower()
        ok += got == value
    return ok, len(want)


def run_persona(persona: Persona, agent_kind: str, agent_llm: LLMProvider, sim_llm: LLMProvider,
                max_turns: int) -> Result:
    deps = build_deps(":memory:", llm=agent_llm, clock=lambda: CLOCK)
    agent = GraphAgent(deps) if agent_kind == "graph" else BaselineAgent(deps)
    svc = ChatService(deps, agent)
    customer = Customer(sim_llm, persona)
    cid = f"eval:{persona.id}"
    turns, calls, seconds, error = 0, 0, [], None
    try:
        for _ in range(max_turns):
            message = customer.next_message(deps.transcripts.history(cid))
            if message is None:
                break
            t0 = time.perf_counter()
            result = svc.handle(cid, message)
            seconds.append(time.perf_counter() - t0)
            turns += 1
            calls += result.llm_calls
            if result.mode == "human":
                break  # a person takes it from here
    except LLM_ERRORS as exc:  # the simulator failed; the agent's own failures are handled by the service
        error = f"simulator: {type(exc).__name__}: {exc}"[:300]

    booked = deps.calendar.for_conversation(cid)
    handoffs = deps.handoffs.for_conversation(cid)
    lead = deps.crm.get_lead(cid)
    if booked:
        outcome = "booked"
    elif handoffs:
        outcome = "handoff"
    elif lead and lead["status"] == "disqualified":
        outcome = "disqualified"
    else:
        outcome = "unfinished"
    agent_error = next((h["reason"] for h in handoffs if h["reason"].startswith("assistant error")), None)
    ok, total = _crm_score(lead, persona.crm) if outcome == "booked" else (0, len(persona.crm))
    return Result(
        persona=persona.id, expected=persona.expected, outcome=outcome, turns=turns, agent_calls=calls,
        seconds_per_turn=round(statistics.mean(seconds), 2) if seconds else 0.0,
        lead_status=lead["status"] if lead else None,
        booked_start=booked[0]["start"] if booked else None,
        slot_ok=_slot_ok(booked[0]["start"], persona.slot_rule) if booked else None,
        crm_fields_ok=ok, crm_fields_total=total,
        handoff_reason=handoffs[0]["reason"] if handoffs else None,
        error=error or agent_error,
        transcript=deps.transcripts.history(cid),
    )


def score(results: list[Result]) -> dict:
    n = len(results)

    def share(hits: int, total: int) -> float | None:
        return round(100 * hits / total, 1) if total else None

    should_book = [r for r in results if r.expected == "booked"]
    qual_cases = [r for r in results if r.expected in ("booked", "disqualified")]
    qual_hits = sum(
        (r.expected == "booked") == (r.lead_status in ("qualified", "booked"))
        and (r.expected != "disqualified" or r.lead_status == "disqualified")
        for r in qual_cases)
    predicted_handoff = [r for r in results if r.handoff_reason]
    wrong_bookings = [r for r in results if r.booked_start and (r.expected != "booked" or r.slot_ok is False)]
    crm_ok = sum(r.crm_fields_ok for r in should_book)
    crm_total = sum(r.crm_fields_total for r in should_book)
    return {
        "conversations": n,
        "outcome_accuracy": share(sum(r.outcome == r.expected for r in results), n),
        "qualification_accuracy": share(qual_hits, len(qual_cases)),
        "booking_success": share(sum(r.outcome == "booked" for r in should_book), len(should_book)),
        "wrong_bookings": len(wrong_bookings),
        "wrong_booking_ids": [r.persona for r in wrong_bookings],
        "handoff_precision": share(sum(r.expected == "handoff" for r in predicted_handoff), len(predicted_handoff)),
        "handoff_recall": share(sum(bool(r.handoff_reason) for r in results if r.expected == "handoff"),
                                sum(r.expected == "handoff" for r in results)),
        "crm_field_accuracy": share(crm_ok, crm_total),
        "avg_customer_turns": round(statistics.mean(r.turns for r in results), 1),
        "avg_llm_calls_per_conversation": round(statistics.mean(r.agent_calls for r in results), 1),
        "median_seconds_per_turn": round(statistics.median(r.seconds_per_turn for r in results if r.turns), 1),
        "unfinished": [r.persona for r in results if r.outcome == "unfinished"],
        "errors": [r.persona for r in results if r.error],
    }


def write_report(name: str, meta: dict, metrics: dict, results: list[Result]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Eval run `{name}`", "",
             (f"Agent `{meta['agent']}` on {meta['agent_model']}; customers played by {meta['sim_model']}. "
             f"{metrics['conversations']} conversations, run {meta['finished']}."), "",
             "| Metric | Value |", "|---|---|"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "| Persona | Expected | Outcome | Turns | Booked | Slot ok | CRM | Handoff reason | Error |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        mark = "" if r.outcome == r.expected else " ❌"
        crm = f"{r.crm_fields_ok}/{r.crm_fields_total}" if r.crm_fields_total else ""
        lines.append(f"| {r.persona} | {r.expected} | {r.outcome}{mark} | {r.turns} | {r.booked_start or ''} | "
                     f"{'' if r.slot_ok is None else r.slot_ok} | {crm} | {r.handoff_reason or ''} | "
                     f"{(r.error or '')[:60]} |")
    path = RESULTS_DIR / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=["graph", "baseline"], default="graph")
    parser.add_argument("--tag", default="r1", help="cache namespace; a new tag = an independent run")
    parser.add_argument("--only", nargs="*", help="persona id prefixes (e.g. b05 h07)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=14)
    parser.add_argument("--sim-provider", default="nvidia")
    parser.add_argument("--sim-model", default="nvidia/nemotron-3-super-120b-a12b")
    args = parser.parse_args()

    personas = [p for p in PERSONAS if not args.only or any(p.id.startswith(o) for o in args.only)]
    agent_llm = CachedLLM(get_llm(), f"{args.tag}:{args.agent}")
    sim_llm = CachedLLM(get_llm(args.sim_provider, args.sim_model), f"{args.tag}:{args.agent}:sim")
    name = f"{args.agent}-{args.tag}" + ("-partial" if args.only else "")
    print(f"{name}: {len(personas)} personas, agent {agent_llm.model}, customers {sim_llm.model}")

    lock = threading.Lock()

    def one(p: Persona) -> Result:
        r = run_persona(p, args.agent, agent_llm, sim_llm, args.max_turns)
        with lock:
            flag = "ok " if r.outcome == r.expected else "BAD"
            print(f"  {flag} {p.id:24} expected {r.expected:12} got {r.outcome:12} {r.turns:2} turns"
                  f"{'  ERROR ' + r.error[:80] if r.error else ''}", flush=True)
        return r

    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, personas))
    metrics = score(results)
    meta = {"agent": args.agent, "tag": args.tag, "agent_model": agent_llm.model, "sim_model": sim_llm.model,
            "finished": datetime.now().strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - started) / 60, 1)}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{name}.json").write_text(
        json.dumps({"meta": meta, "metrics": metrics, "results": [asdict(r) for r in results]}, indent=1),
        encoding="utf-8")
    report = write_report(name, meta, metrics, results)
    print(json.dumps(metrics, indent=1))
    print(f"report: {report}")


if __name__ == "__main__":
    main()
