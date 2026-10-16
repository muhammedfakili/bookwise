"""SQLite implementations of the ports, plus the transcript store.

One file holds leads, appointments, handoffs and messages. A partial UNIQUE index on
appointments makes a double booking impossible at the database level, whatever the
LLM decides.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from zlib import crc32

from app import business
from app.ports import Appointment, Slot, SlotUnavailable

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY, channel TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('customer', 'assistant', 'human_agent', 'system')),
    text TEXT NOT NULL, created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conv ON messages (conversation_id, id);
CREATE TABLE IF NOT EXISTS leads (
    conversation_id TEXT PRIMARY KEY, channel TEXT NOT NULL,
    name TEXT, phone TEXT, zip_code TEXT, service TEXT, other_service TEXT,
    budget_usd INTEGER, timeline_weeks INTEGER, decision_maker INTEGER,
    status TEXT NOT NULL DEFAULT 'new', status_reason TEXT, notes TEXT,
    created TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
    name TEXT NOT NULL, phone TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'booked', created TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_booking_per_slot ON appointments (start) WHERE status = 'booked';
CREATE TABLE IF NOT EXISTS handoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
    reason TEXT NOT NULL, summary TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
    created TEXT NOT NULL, resolved TEXT
);
CREATE TABLE IF NOT EXISTS baseline_threads (
    conversation_id TEXT PRIMARY KEY, messages TEXT NOT NULL
);
"""

LEAD_FIELDS = ("name", "phone", "zip_code", "service", "other_service", "budget_usd", "timeline_weeks",
               "decision_maker", "status", "status_reason", "notes")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


class Database:
    """A thread-safe SQLite connection (FastAPI runs sync endpoints in a thread pool)."""

    def __init__(self, path: Path | str, clock: Callable[[], datetime] = datetime.now) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.clock = clock
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def now(self) -> str:
        return self.clock().isoformat(timespec="seconds")

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def rows(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def close(self) -> None:
        self._conn.close()


class Transcripts:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure_conversation(self, conversation_id: str, channel: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO conversations VALUES (?, ?, ?)",
                        (conversation_id, channel, self.db.now()))

    def add(self, conversation_id: str, role: str, text: str) -> None:
        self.db.execute("INSERT INTO messages (conversation_id, role, text, created) VALUES (?, ?, ?, ?)",
                        (conversation_id, role, text, self.db.now()))

    def history(self, conversation_id: str, limit: int = 200) -> list[dict]:
        rows = self.db.rows(
            "SELECT role, text, created FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit))
        return rows[::-1]

    def conversations(self) -> list[dict]:
        return self.db.rows("""
            SELECT c.id, c.channel, c.created, COUNT(m.id) AS messages, MAX(m.created) AS last_message
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id ORDER BY last_message DESC""")


class SqliteCrm:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_lead(self, conversation_id: str, channel: str, fields: dict) -> dict:
        unknown = set(fields) - set(LEAD_FIELDS)
        if unknown:
            raise ValueError(f"unknown lead fields: {sorted(unknown)}")
        now = self.db.now()
        self.db.execute("INSERT OR IGNORE INTO leads (conversation_id, channel, created, updated) VALUES (?, ?, ?, ?)",
                        (conversation_id, channel, now, now))
        if fields:
            values = {k: (int(v) if isinstance(v, bool) else v) for k, v in fields.items()}
            sets = ", ".join(f"{k} = :{k}" for k in values)
            self.db.execute(f"UPDATE leads SET {sets}, updated = :updated WHERE conversation_id = :cid",
                            {**values, "updated": now, "cid": conversation_id})
        return self.get_lead(conversation_id) or {}

    def get_lead(self, conversation_id: str) -> dict | None:
        rows = self.db.rows("SELECT * FROM leads WHERE conversation_id = ?", (conversation_id,))
        if not rows:
            return None
        lead = rows[0]
        if lead["decision_maker"] is not None:
            lead["decision_maker"] = bool(lead["decision_maker"])
        return lead

    def leads(self) -> list[dict]:
        return self.db.rows("SELECT * FROM leads ORDER BY updated DESC")


class SqliteCalendar:
    """Free consultation slots on a weekday grid; some are pre-taken by other customers."""

    def __init__(self, db: Database, seed_busy: bool = True) -> None:
        self.db = db
        if seed_busy:
            self._seed_busy()

    def _grid(self) -> list[Slot]:
        now = self.db.clock()
        slots = []
        for day in range(1, business.BOOKING_HORIZON_DAYS + 1):
            date = (now + timedelta(days=day)).date()
            if date.weekday() >= 5:
                continue
            for t in business.SLOT_TIMES:
                start = datetime.combine(date, t)
                slots.append(Slot(start, start + timedelta(minutes=business.SLOT_MINUTES)))
        return slots

    def _seed_busy(self) -> None:
        """Pretend about a third of the grid is already booked (deterministic for a given clock)."""
        if self.db.rows("SELECT 1 FROM appointments WHERE conversation_id = 'existing-customer' LIMIT 1"):
            return
        for slot in self._grid():
            if crc32(slot.start.isoformat().encode()) % 3 == 0:
                self.db.execute(
                    "INSERT OR IGNORE INTO appointments (conversation_id, name, phone, start, end, created) "
                    "VALUES ('existing-customer', 'Existing customer', '-', ?, ?, ?)",
                    (slot.start.isoformat(), slot.end.isoformat(), self.db.now()))

    def _taken(self) -> set[str]:
        return {r["start"] for r in self.db.rows("SELECT start FROM appointments WHERE status = 'booked'")}

    def find_free_slots(
        self, limit: int = 3, part_of_day: str | None = None, weekdays: list[str] | None = None,
        exclude: list[datetime] | None = None,
    ) -> list[Slot]:
        taken = self._taken() | {e.isoformat() for e in exclude or []}
        wanted_days = {WEEKDAYS.index(d.lower()) for d in weekdays or [] if d.lower() in WEEKDAYS}
        free = []
        for slot in self._grid():
            if slot.start.isoformat() in taken:
                continue
            if part_of_day == "morning" and slot.start.hour >= 12:
                continue
            if part_of_day == "afternoon" and slot.start.hour < 12:
                continue
            if wanted_days and slot.start.weekday() not in wanted_days:
                continue
            free.append(slot)
        # Spread the offer over different days and times of day, so the customer gets a real choice.
        by_day: dict = {}
        for slot in free:
            by_day.setdefault(slot.start.date(), []).append(slot)
        picked = [day[i % len(day)] for i, day in enumerate(by_day.values())][:limit]
        return picked

    def book(self, start: datetime, conversation_id: str, name: str, phone: str) -> Appointment:
        grid = {s.start: s for s in self._grid()}
        if start not in grid:
            raise SlotUnavailable(f"{start.isoformat()} is not a bookable consultation time")
        slot = grid[start]
        try:
            cur = self.db.execute(
                "INSERT INTO appointments (conversation_id, name, phone, start, end, created) VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, name, phone, slot.start.isoformat(), slot.end.isoformat(), self.db.now()))
        except sqlite3.IntegrityError as exc:
            raise SlotUnavailable(f"{start.isoformat()} is already booked") from exc
        return Appointment(cur.lastrowid, conversation_id, name, phone, slot.start, slot.end, "booked")

    def cancel(self, appointment_id: int) -> None:
        self.db.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,))

    def appointments(self, include_existing: bool = False) -> list[dict]:
        where = "" if include_existing else "WHERE conversation_id != 'existing-customer'"
        return self.db.rows(f"SELECT * FROM appointments {where} ORDER BY start")

    def for_conversation(self, conversation_id: str) -> list[dict]:
        return self.db.rows("SELECT * FROM appointments WHERE conversation_id = ? AND status = 'booked'",
                            (conversation_id,))


class SqliteHandoffs:
    def __init__(self, db: Database) -> None:
        self.db = db

    def open(self, conversation_id: str, reason: str, summary: str) -> int:
        existing = self.active(conversation_id)
        if existing:
            return existing["id"]
        cur = self.db.execute(
            "INSERT INTO handoffs (conversation_id, reason, summary, created) VALUES (?, ?, ?, ?)",
            (conversation_id, reason, summary, self.db.now()))
        return cur.lastrowid

    def active(self, conversation_id: str) -> dict | None:
        rows = self.db.rows("SELECT * FROM handoffs WHERE conversation_id = ? AND status = 'open'",
                            (conversation_id,))
        return rows[0] if rows else None

    def resolve(self, handoff_id: int) -> None:
        self.db.execute("UPDATE handoffs SET status = 'resolved', resolved = ? WHERE id = ?",
                        (self.db.now(), handoff_id))

    def get(self, handoff_id: int) -> dict | None:
        rows = self.db.rows("SELECT * FROM handoffs WHERE id = ?", (handoff_id,))
        return rows[0] if rows else None

    def all(self) -> list[dict]:
        return self.db.rows("SELECT * FROM handoffs ORDER BY status = 'resolved', created DESC")

    def for_conversation(self, conversation_id: str) -> list[dict]:
        return self.db.rows("SELECT * FROM handoffs WHERE conversation_id = ?", (conversation_id,))


class BaselineThreads:
    """The baseline agent's raw tool-calling history (the graph agent uses LangGraph checkpoints)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def load(self, conversation_id: str) -> list[dict]:
        rows = self.db.rows("SELECT messages FROM baseline_threads WHERE conversation_id = ?", (conversation_id,))
        return json.loads(rows[0]["messages"]) if rows else []

    def save(self, conversation_id: str, messages: list[dict]) -> None:
        self.db.execute("INSERT OR REPLACE INTO baseline_threads VALUES (?, ?)",
                        (conversation_id, json.dumps(messages)))
