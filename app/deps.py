"""Everything an agent needs, wired once: storage, ports, LLM, clock."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.llm import LLMProvider, get_llm
from app.store import BaselineThreads, Database, SqliteCalendar, SqliteCrm, SqliteHandoffs, Transcripts


@dataclass
class Deps:
    db: Database
    llm: LLMProvider
    transcripts: Transcripts = field(init=False)
    crm: SqliteCrm = field(init=False)
    calendar: SqliteCalendar = field(init=False)
    handoffs: SqliteHandoffs = field(init=False)
    threads: BaselineThreads = field(init=False)

    def __post_init__(self) -> None:
        self.transcripts = Transcripts(self.db)
        self.crm = SqliteCrm(self.db)
        self.calendar = SqliteCalendar(self.db)
        self.handoffs = SqliteHandoffs(self.db)
        self.threads = BaselineThreads(self.db)

    def now(self) -> datetime:
        return self.db.clock()

    @property
    def checkpoint_path(self) -> str:
        path = str(self.db.path)
        return ":memory:" if path == ":memory:" else str(Path(path).with_suffix(".checkpoints.db"))


def make_clock() -> Callable[[], datetime]:
    """The frozen clock from settings (demo, eval) or real time."""
    frozen = settings.clock
    if frozen is None:
        return datetime.now
    return lambda: frozen


def build_deps(db_path: Path | str | None = None, llm: LLMProvider | None = None,
               clock: Callable[[], datetime] | None = None) -> Deps:
    db = Database(db_path or settings.db_path, clock or make_clock())
    return Deps(db=db, llm=llm or get_llm())
