"""Fill a running API with demo conversations for the screenshots (real agent, scripted customer).

Start the API on a fresh database with the eval's frozen clock first:
    DB_PATH=data/demo.db CLOCK=2026-10-19T09:00:00 uv run uvicorn app.api:app --port 8000
Then:
    uv run python docs/portfolio/demo_seed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

API = "http://localhost:8000"
OUT = Path(__file__).parent / "demo_ids.json"

SCRIPTS = {
    "booked": ["Hi! We're thinking about redoing our kitchen, it's from the 90s.",
               "We're in 78704, would love to start in about a month",
               "Probably around 40k. And yes, my wife and I own the house",
               "Mark Daniels, 512-555-0199",
               "Afternoons are better for us actually",
               "The Thursday one works"],
    "declined": ["hello, do you do kitchens in Dallas? zip 75201",
                 "oh ok, thanks anyway"],
    "handoff": ["Your crew was supposed to finish my bathroom last week and nobody has called me back."],
}
WHATSAPP = {"entry": [{"changes": [{"value": {
    "contacts": [{"profile": {"name": "Rosa"}, "wa_id": "15125550177"}],
    "messages": [{"from": "15125550177", "id": "wamid.demo1", "type": "text",
                  "text": {"body": "Hola! Is a bathroom remodel for around 15k possible in 78745?"}}]}}]}]}


def main() -> None:
    http = httpx.Client(base_url=API, timeout=300)
    ids = {}
    for name, messages in SCRIPTS.items():
        cid = None
        for text in messages:
            turn = http.post("/chat", json={"message": text, "conversation_id": cid}).raise_for_status().json()
            cid = turn["conversation_id"]
            print(f"[{name}] > {text}\n  < {turn['reply']}\n  ({turn['action']}: {turn['events']})")
        ids[name] = cid
    handoff = next(h for h in http.get("/admin/handoffs").json() if h["conversation_id"] == ids["handoff"])
    http.post(f"/admin/handoffs/{handoff['id']}/reply",
              json={"text": "Hi, this is Dana, the project manager. I'm so sorry about this. I'm calling you in 5 "
                            "minutes to set a finish date."}).raise_for_status()
    http.post("/webhook/whatsapp", json=WHATSAPP).raise_for_status()
    ids["whatsapp"] = "wa:15125550177"
    OUT.write_text(json.dumps(ids, indent=1), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
