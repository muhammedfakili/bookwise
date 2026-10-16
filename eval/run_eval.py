"""Run the simulated customers against an agent and score the outcome from the database.

    uv run python -m eval.run_eval --agent graph
    uv run python -m eval.run_eval --agent baseline --tag r2     # a second, independent run
    uv run python -m eval.run_eval --agent graph --only b05 h07   # a few personas

Each persona gets a fresh database, a frozen clock (Monday 2026-10-19 09:00) and a
customer played by an LLM with its own role prompt (Nemotron 3 Super on NVIDIA NIM, free;
the same model as the agent, because it was the only free NIM model that stayed up for
hours of runs -- outcomes are read from the database, never judged by the model). The
conversation ends when the customer says goodbye, a human takes over, or after
--max-turns customer messages.

Scoring never asks an LLM: the outcome is "booked" if a consultation was booked, else the
agent's first final action (decline or hand off); bookings, CRM fields and handoffs are read
from the database and compared with the persona's label. Every model call is cached on disk
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
TERMINAL = {"book": "booked", "disqualify": "disqualified", "handoff": "handoff"}
PERSONA_ATTEMPTS = 5


class CachedLLM:
    """Wraps a provider; identical calls (same tag) are answered from disk.

    Each entry also keeps how long the real call took, so a re-scored run (all answers
    from the cache) still reports the latency the agent really had.
    """

    def __init__(self, inner: LLMProvider, tag: str) -> None:
        self.inner, self.tag = inner, tag
        self.name, self.model = inner.name, inner.model
        self._local = threading.local()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def take_seconds(self) -> float | None:
        """Model seconds spent by this thread since the last call (None if a cache entry had no timing)."""
        spent, self._local.seconds = getattr(self._local, "seconds", 0.0), 0.0
        return spent

    def _key(self, *parts: object) -> Path:
        blob = json.dumps([self.tag, self.model, *parts], sort_keys=True, default=str)
        return CACHE_DIR / f"{hashlib.sha256(blob.encode()).hexdigest()[:24]}.json"

    def _cached(self, path: Path, call: callable) -> dict:
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            if "answer" not in entry:  # older entries: the answer only, no timing
                self._add(None)
                return entry
        else:
            t0 = time.perf_counter()
            entry = {"answer": call(), "seconds": round(time.perf_counter() - t0, 2)}
            path.write_text(json.dumps(entry), encoding="utf-8")
        self._add(entry["seconds"])
        return entry["answer"]

    def _add(self, seconds: float | None) -> None:
        current = getattr(self._local, "seconds", 0.0)
        self._local.seconds = None if current is None or seconds is None else current + seconds

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
- Never invent events that are not in your facts (payments, deposits, signed contracts, earlier \
calls, promises the company made). If they decline you, accept it politely.
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
        # Some models write a last message and <END> together: keep the message (the next turn ends).
        text = re.sub(r"^(You|Customer):\s*", "", text.replace(END, "").strip())
        return text or None


@dataclass
class Result:
    persona: str
    expected: str
    outcome: str
    turns: int
    agent_calls: int
    seconds_per_turn: float | None  # model time per agent turn; None if the cache had no timing
    lead_status: str | None
    booked_start: str | None
    slot_ok: bool | None
    crm_fields_ok: int
    crm_fields_total: int
    handoff_reason: str | None
    error: str | None
    decisions: list[str]
    transcript: list[dict]

    @property
    def verdict(self) -> str | None:
        """The agent's qualification call: booked or got to the slot stage = qualified."""
        if self.booked_start or self.lead_status in ("qualified", "booked"):
            return "qualified"
        if "disqualified" in self.decisions or self.lead_status == "disqualified":
            return "disqualified"
        return None


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


def run_persona(persona: Persona, agent_kind: str, agent_llm: CachedLLM, sim_llm: LLMProvider,
                max_turns: int) -> Result:
    deps = build_deps(":memory:", llm=agent_llm, clock=lambda: CLOCK)
    agent = GraphAgent(deps) if agent_kind == "graph" else BaselineAgent(deps)
    svc = ChatService(deps, agent, handoff_on_error=False)
    customer = Customer(sim_llm, persona)
    cid = f"eval:{persona.id}"
    turns, calls, seconds, error = 0, 0, [], None
    decisions: list[str] = []  # the agent's final actions in order (book, disqualify, handoff)
    try:
        for _ in range(max_turns):
            message = customer.next_message(deps.transcripts.history(cid))
            if message is None:
                break
            agent_llm.take_seconds()
            result = svc.handle(cid, message)
            spent = agent_llm.take_seconds()
            if spent is not None:
                seconds.append(spent)
            turns += 1
            calls += result.llm_calls
            if result.action in TERMINAL:
                decisions.append(TERMINAL[result.action])
            if result.mode == "human":
                break  # a person takes it from here
    except LLM_ERRORS as exc:
        # A provider outage (after all retries) on either side. Not scored; re-run the tag to retry.
        error = f"{type(exc).__name__}: {exc}"[:300]

    booked = deps.calendar.for_conversation(cid)
    handoffs = deps.handoffs.for_conversation(cid)
    lead = deps.crm.get_lead(cid)
    if error:
        outcome = "error"
    elif booked:
        outcome = "booked"
    elif decisions:
        outcome = decisions[0]  # e.g. declined, then the customer argued and got a person: "disqualified"
    elif handoffs:
        outcome = "handoff"
    elif lead and lead["status"] == "disqualified":
        outcome = "disqualified"
    else:
        outcome = "unfinished"
    ok, total = _crm_score(lead, persona.crm) if outcome == "booked" else (0, len(persona.crm))
    return Result(
        persona=persona.id, expected=persona.expected, outcome=outcome, turns=turns, agent_calls=calls,
        seconds_per_turn=round(statistics.mean(seconds), 2) if seconds else None,
        lead_status=lead["status"] if lead else None,
        booked_start=booked[0]["start"] if booked else None,
        slot_ok=_slot_ok(booked[0]["start"], persona.slot_rule) if booked else None,
        crm_fields_ok=ok, crm_fields_total=total,
        handoff_reason=handoffs[0]["reason"] if handoffs else None,
        error=error,
        decisions=decisions,
        transcript=deps.transcripts.history(cid),
    )


def score(all_results: list[Result]) -> dict:
    results = [r for r in all_results if not r.error]  # outages are reported, never scored
    n = len(results)

    def share(hits: int, total: int) -> float | None:
        return round(100 * hits / total, 1) if total else None

    should_book = [r for r in results if r.expected == "booked"]
    qual_cases = [r for r in results if r.expected in ("booked", "disqualified")]
    qual_hits = sum(r.verdict == ("qualified" if r.expected == "booked" else "disqualified") for r in qual_cases)
    predicted_handoff = [r for r in results if r.handoff_reason]
    wrong_bookings = [r for r in results if r.booked_start and (r.expected != "booked" or r.slot_ok is False)]
    crm_ok = sum(r.crm_fields_ok for r in should_book)
    crm_total = sum(r.crm_fields_total for r in should_book)
    return {
        "conversations": n,
        "errors_not_scored": [r.persona for r in all_results if r.error],
        "outcome_accuracy": share(sum(r.outcome == r.expected for r in results), n),
        "qualification_accuracy": share(qual_hits, len(qual_cases)),
        "booking_success": share(sum(r.outcome == "booked" for r in should_book), len(should_book)),
        "wrong_bookings": len(wrong_bookings),
        "wrong_booking_ids": [r.persona for r in wrong_bookings],
        "handoff_precision": share(sum(r.expected == "handoff" for r in predicted_handoff), len(predicted_handoff)),
        "unneeded_handoffs": [r.persona for r in predicted_handoff if r.expected != "handoff"],
        "handoff_recall": share(sum(bool(r.handoff_reason) for r in results if r.expected == "handoff"),
                                sum(r.expected == "handoff" for r in results)),
        "crm_field_accuracy": share(crm_ok, crm_total),
        "avg_customer_turns": round(statistics.mean(r.turns for r in results), 1),
        "avg_llm_calls_per_conversation": round(statistics.mean(r.agent_calls for r in results), 1),
        "median_seconds_per_turn": (round(statistics.median(timed), 1)
                                    if (timed := [r.seconds_per_turn for r in results if r.seconds_per_turn])
                                    else None),
        "unfinished": [r.persona for r in results if r.outcome == "unfinished"],
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
    parser.add_argument("--provider", default="nvidia", help="the agent's model provider")
    parser.add_argument("--model", default="nvidia/nemotron-3-super-120b-a12b")
    parser.add_argument("--sim-provider", default="nvidia")
    parser.add_argument("--sim-model", default="nvidia/nemotron-3-super-120b-a12b")
    args = parser.parse_args()

    personas = [p for p in PERSONAS if not args.only or any(p.id.startswith(o) for o in args.only)]
    agent_llm = CachedLLM(get_llm(args.provider, args.model), f"{args.tag}:{args.agent}")
    sim_llm = CachedLLM(get_llm(args.sim_provider, args.sim_model), f"{args.tag}:{args.agent}:sim")
    name = f"{args.agent}-{args.tag}" + ("-partial" if args.only else "")
    print(f"{name}: {len(personas)} personas, agent {agent_llm.model}, customers {sim_llm.model}")

    lock = threading.Lock()

    def one(p: Persona) -> Result:
        # A provider outage mid-conversation: wait, then replay the conversation. Finished turns
        # come back from the cache instantly, so it continues where it stopped.
        for attempt in range(PERSONA_ATTEMPTS):
            r = run_persona(p, args.agent, agent_llm, sim_llm, args.max_turns)
            if not r.error or attempt == PERSONA_ATTEMPTS - 1:
                break
            time.sleep(60)
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
