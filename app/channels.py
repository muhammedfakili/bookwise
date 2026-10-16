"""Where replies go, and the WhatsApp Cloud API message format.

The web chat reads replies from the HTTP response, so its channel does nothing.
WhatsApp is different: Meta calls our webhook with the customer's message and we send
the answer back with a separate API call. `MockWhatsApp` keeps those outgoing
payloads in memory (exactly what would be POSTed), `WhatsAppCloud` really sends them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel


class WebChannel:
    def send(self, to: str, text: str) -> None:
        return None


def whatsapp_text_payload(to: str, text: str) -> dict:
    """Body of POST /{phone-number-id}/messages in the WhatsApp Cloud API."""
    return {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to,
            "type": "text", "text": {"preview_url": False, "body": text}}


@dataclass
class MockWhatsApp:
    outbox: list[dict] = field(default_factory=list)

    def send(self, to: str, text: str) -> None:
        self.outbox.append(whatsapp_text_payload(to, text))


class WhatsAppCloud:
    """Real sender. Needs a Meta app, a phone number id and a permanent access token."""

    def __init__(self, phone_number_id: str, token: str, api_version: str = "v21.0") -> None:
        self._url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
        self._http = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=20)

    def send(self, to: str, text: str) -> None:
        self._http.post(self._url, json=whatsapp_text_payload(to, text)).raise_for_status()


class InboundText(BaseModel):
    sender: str  # the customer's phone number, in international format without "+"
    name: str | None
    text: str
    message_id: str


def parse_whatsapp_webhook(body: dict) -> list[InboundText]:
    """Pull the text messages out of a Cloud API webhook call. Statuses and media are skipped."""
    found = []
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            names = {c.get("wa_id"): c.get("profile", {}).get("name") for c in value.get("contacts", [])}
            for msg in value.get("messages", []):
                if msg.get("type") != "text":
                    continue
                found.append(InboundText(sender=msg["from"], name=names.get(msg["from"]),
                                         text=msg["text"]["body"], message_id=msg["id"]))
    return found
