"""Interfaces between the agent and the outside world.

The agent only knows these Protocols. The SQLite classes in app/store.py implement
them for the demo; Google Calendar, HubSpot or the WhatsApp Cloud API would be one
new class each, with no change to the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Slot:
    start: datetime
    end: datetime

    def label(self) -> str:
        return self.start.strftime("%A %b %d, %I:%M %p").replace(" 0", " ")


@dataclass(frozen=True)
class Appointment:
    id: int
    conversation_id: str
    name: str
    phone: str
    start: datetime
    end: datetime
    status: str


class SlotUnavailable(RuntimeError):
    """The slot is taken, in the past, or not one of the bookable times."""


class CalendarPort(Protocol):
    def find_free_slots(
        self, limit: int = 3, part_of_day: str | None = None, weekdays: list[str] | None = None,
        exclude: list[datetime] | None = None,
    ) -> list[Slot]: ...

    def book(self, start: datetime, conversation_id: str, name: str, phone: str) -> Appointment: ...

    def cancel(self, appointment_id: int) -> None: ...


class CrmPort(Protocol):
    def upsert_lead(self, conversation_id: str, channel: str, fields: dict) -> dict: ...

    def get_lead(self, conversation_id: str) -> dict | None: ...


class HandoffPort(Protocol):
    def open(self, conversation_id: str, reason: str, summary: str) -> int: ...

    def active(self, conversation_id: str) -> dict | None: ...


class ChannelPort(Protocol):
    """Delivers the agent's reply to the customer (web chat: nothing to do; WhatsApp: an API call)."""

    def send(self, to: str, text: str) -> None: ...
