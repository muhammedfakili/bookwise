from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime

import pytest

from app.agents.graph_agent import GraphAgent
from app.deps import Deps, build_deps
from app.llm import LLMAnswer, ToolTurn
from app.service import ChatService

MONDAY = datetime(2026, 10, 19, 9, 0)


def analysis(**fields) -> dict:
    """A TurnAnalysis as the LLM would return it; defaults to a calm, confident 'share_info'."""
    return {"intent": "share_info", "sentiment": "neutral", "confidence": 0.9, **fields}


class FakeLLM:
    """Scripted model. Structured calls pop `analyses`; text calls echo the instruction; tool turns pop `turns`."""

    name = "fake"
    model = "fake-model"

    def __init__(self, analyses: list[dict] | None = None, turns: list[ToolTurn] | None = None,
                 fail: Callable[[], Exception] | None = None) -> None:
        self.analyses = list(analyses or [])
        self.turns = list(turns or [])
        self.fail = fail
        self.prompts: list[str] = []

    def generate_structured(self, system: str, prompt: str, schema: type) -> LLMAnswer:
        if self.fail:
            raise self.fail()
        self.prompts.append(prompt)
        return LLMAnswer(json.dumps(self.analyses.pop(0)), self.model, 10, 5)

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        # Echo the instruction so tests can see what the agent was told to say.
        return LLMAnswer("REPLY: " + prompt.split("INSTRUCTION for your next message:\n")[-1], self.model, 10, 5)

    def chat_tools(self, system, messages, tools) -> ToolTurn:
        self.prompts.append(messages[-1].text)
        return self.turns.pop(0)


@pytest.fixture
def make_deps() -> Callable[..., Deps]:
    def _make(llm: FakeLLM | None = None) -> Deps:
        return build_deps(":memory:", llm=llm or FakeLLM(), clock=lambda: MONDAY)
    return _make


@pytest.fixture
def graph_service(make_deps) -> Callable[[list[dict]], ChatService]:
    def _make(analyses: list[dict]) -> ChatService:
        deps = make_deps(FakeLLM(analyses))
        return ChatService(deps, GraphAgent(deps))
    return _make
