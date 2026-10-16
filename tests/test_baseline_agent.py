from app.agents.baseline_agent import BaselineAgent
from app.llm import ToolCall, ToolTurn
from app.service import ChatService
from tests.conftest import FakeLLM


def turn(text: str = "", *calls: tuple[str, dict]) -> ToolTurn:
    return ToolTurn(text, [ToolCall(f"id{i}", n, a) for i, (n, a) in enumerate(calls)], "fake", 10, 5)


def test_tool_loop_books_through_the_shared_calendar(make_deps):
    deps = make_deps()
    slot = deps.calendar.find_free_slots(1)[0].start.isoformat()
    deps.llm = FakeLLM(turns=[
        turn("", ("check_availability", {})),
        turn("Here are some times: ..."),
        turn("", ("book_consultation", {"start": slot, "name": "Ann", "phone": "5125550101"}),
             ("save_lead", {"status": "qualified", "service": "kitchen"})),
        turn("You're booked!"),
    ])
    svc = ChatService(deps, BaselineAgent(deps))
    first = svc.handle("c1", "kitchen, 78704, 40k, own it, Ann 5125550101")
    assert first.reply == "Here are some times: ..."
    second = svc.handle("c1", "the first one")
    assert (second.action, second.llm_calls) == ("book", 2)
    assert deps.calendar.for_conversation("c1")[0]["start"] == slot
    assert deps.crm.get_lead("c1")["status"] == "booked", "save_lead must not downgrade a booking"
    # The tool history is persisted, so the next turn sees the earlier tool calls.
    assert [m["role"] for m in deps.threads.load("c1")] == [
        "user", "assistant", "tool", "assistant", "user", "assistant", "tool", "tool", "assistant"]


def test_bad_tool_arguments_are_reported_back_not_raised(make_deps):
    deps = make_deps()
    deps.llm = FakeLLM(turns=[
        turn("", ("book_consultation", {"start": "next tuesday", "name": "A", "phone": "1"})),
        turn("Sorry, let me check the calendar."),
    ])
    reply = BaselineAgent(deps).respond("c1", "web")
    thread = deps.threads.load("c1")
    assert '"ok": false' in thread[2]["text"]
    assert reply.text == "Sorry, let me check the calendar."
    assert deps.calendar.for_conversation("c1") == []


def test_handoff_tool_opens_the_queue(make_deps):
    deps = make_deps()
    deps.llm = FakeLLM(turns=[turn("", ("hand_off_to_human", {"reason": "complaint"})), turn("Passing you on.")])
    result = ChatService(deps, BaselineAgent(deps)).handle("c1", "your crew broke my sink")
    assert (result.action, result.mode) == ("handoff", "human")
    assert deps.crm.get_lead("c1")["status"] == "handoff"
