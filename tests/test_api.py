import pytest
from fastapi.testclient import TestClient

from app import api
from app.agents.graph_agent import GraphAgent
from app.channels import MockWhatsApp
from app.service import ChatService
from tests.conftest import FakeLLM, analysis


@pytest.fixture
def client(make_deps):
    deps = make_deps(FakeLLM([analysis(service="kitchen")] + [analysis(intent="wants_human")] * 5))
    svc = ChatService(deps, GraphAgent(deps), channels={"whatsapp": MockWhatsApp()})
    api.app.dependency_overrides[api.service] = lambda: svc
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_chat_creates_a_conversation_and_returns_the_lead(client):
    r = client.post("/chat", json={"message": "We want a new kitchen"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"].startswith("web:")
    assert (body["mode"], body["action"], body["lead"]["service"]) == ("bot", "ask", "kitchen")

    conv = client.get(f"/conversations/{body['conversation_id']}").json()
    assert [m["role"] for m in conv["messages"]] == ["customer", "assistant"]
    assert conv["agent_state"]["asked"] == {"zip_code": 1}


def test_validation_and_unknown_ids(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "conversation_id": "wa:1"}).status_code == 400
    assert client.get("/conversations/nope").status_code == 404
    assert client.post("/admin/handoffs/999/reply", json={"text": "hi"}).status_code == 404
    assert client.post("/admin/handoffs/999/resolve").status_code == 404


def test_handoff_queue_reply_and_resolve(client):
    cid = client.post("/chat", json={"message": "kitchen"}).json()["conversation_id"]
    r = client.post("/chat", json={"message": "can I talk to a person", "conversation_id": cid}).json()
    assert r["mode"] == "human"
    [handoff] = client.get("/admin/handoffs").json()
    assert (handoff["status"], handoff["reason"]) == ("open", "customer asked for a person")
    assert client.post(f"/admin/handoffs/{handoff['id']}/reply", json={"text": "Hi, Dana here"}).status_code == 200
    assert client.post(f"/admin/handoffs/{handoff['id']}/resolve").status_code == 200
    assert client.get("/admin/handoffs").json()[0]["status"] == "resolved"
    assert client.get("/admin/leads").json()[0]["status"] == "handoff"


WEBHOOK = {"object": "whatsapp_business_account", "entry": [{"id": "WABA", "changes": [{"field": "messages", "value": {
    "messaging_product": "whatsapp", "metadata": {"display_phone_number": "15125550142", "phone_number_id": "PNID"},
    "contacts": [{"profile": {"name": "Rosa"}, "wa_id": "15125550177"}],
    "messages": [
        {"from": "15125550177", "id": "wamid.A", "timestamp": "1760860800", "type": "text",
         "text": {"body": "Hi, I need a kitchen remodel"}},
        {"from": "15125550177", "id": "wamid.B", "timestamp": "1760860801", "type": "image",
         "image": {"id": "img1"}},
    ]}}]}]}


def test_whatsapp_webhook_answers_through_the_channel(client):
    r = client.post("/webhook/whatsapp", json=WEBHOOK)
    assert r.json() == {"received": 1}  # the image is skipped
    [sent] = client.get("/admin/whatsapp/outbox").json()
    assert sent["to"] == "15125550177"
    assert sent["messaging_product"] == "whatsapp"
    assert "Greet the customer" in sent["text"]["body"]
    assert client.get("/conversations/wa:15125550177").json()["lead"]["channel"] == "whatsapp"


def test_whatsapp_verification_handshake(client):
    params = {"hub.mode": "subscribe", "hub.verify_token": "bookwise-dev", "hub.challenge": "42"}
    assert client.get("/webhook/whatsapp", params=params).text == "42"
    assert client.get("/webhook/whatsapp", params={**params, "hub.verify_token": "x"}).status_code == 403


def test_api_key_is_enforced_when_set(client, monkeypatch):
    monkeypatch.setattr(api.settings, "app_api_key", "secret")
    assert client.get("/admin/leads").status_code == 401
    assert client.get("/admin/leads", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.post("/webhook/whatsapp", json={"entry": []}).status_code == 200  # Meta sends no key
