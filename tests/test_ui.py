"""Screen tests with Streamlit's AppTest; the API client is replaced by a stub."""

import pytest
from streamlit.testing.v1 import AppTest

from ui import client

LEAD = {"status": "booked", "status_reason": "consultation #3", "service": "kitchen", "zip_code": "78704",
        "timeline_weeks": 4, "budget_usd": 40_000, "decision_maker": True, "name": "Mark", "phone": "5125550101"}
CONVERSATION = {
    "messages": [{"role": "customer", "text": "kitchen please"}, {"role": "assistant", "text": "Great!"},
                 {"role": "human_agent", "text": "Hi, Dana here"}],
    "lead": LEAD, "mode": "bot", "handoffs": [],
    "appointments": [{"start": "2026-10-21T09:00:00"}],
}
HANDOFF = {"id": 7, "conversation_id": "web:abc", "reason": "complaint", "summary": "Reason: complaint.",
           "status": "open", "created": "2026-10-19T09:00:00"}


@pytest.fixture
def api(monkeypatch):
    calls = []
    stub = {
        "chat": lambda self, message, cid: calls.append(("chat", message)) or {
            "conversation_id": "web:abc", "reply": "Hello", "mode": "bot", "action": "ask",
            "events": ["crm.upsert_lead(status=in_progress)"], "lead": LEAD, "llm_calls": 2},
        "conversation": lambda self, cid: CONVERSATION,
        "admin": lambda self, what: {"leads": [{**LEAD, "updated": "x", "conversation_id": "web:abc",
                                                "channel": "web"}],
                                     "appointments": [], "handoffs": [HANDOFF], "whatsapp/outbox": []}[what],
        "reply": lambda self, hid, text: calls.append(("reply", hid, text)),
        "resolve": lambda self, hid: calls.append(("resolve", hid)),
    }
    for name, fn in stub.items():
        monkeypatch.setattr(client.ApiClient, name, fn)
    return calls


def test_chat_sends_and_shows_the_lead_card(api):
    at = AppTest.from_file("../ui/app_pages/chat.py", default_timeout=10).run()
    assert not at.exception
    at.chat_input[0].set_value("We want a new kitchen").run()
    assert ("chat", "We want a new kitchen") in api
    assert at.session_state.conversation_id == "web:abc"
    texts = [m.markdown[0].value for m in at.chat_message]
    assert "kitchen please" in texts
    assert any("Dana here" in t for t in texts)
    assert "$40,000" in at.dataframe[0].value.to_string()
    assert "Consultation booked" in at.success[0].value


def test_team_inbox_replies_and_resolves(api):
    at = AppTest.from_file("../ui/app_pages/admin.py", default_timeout=10).run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["1", "1", "0", "1"]
    at.text_input(key="text-7").set_value("Calling you now")
    at.button[0].click().run()
    assert ("reply", 7, "Calling you now") in api
    at.button[1].click().run()
    assert ("resolve", 7) in api
