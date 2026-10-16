"""Application settings, loaded from environment / .env file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Gemini Flash-Lite is free and fast (500 requests/day), Claude is the paid upgrade,
# NVIDIA's hosted NIM models are free with no daily cap (the evaluation runs on them).
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-opus-5",
    "nvidia": "nvidia/nemotron-3-super-120b-a12b",
}

Provider = Literal["gemini", "anthropic", "nvidia"]
AgentKind = Literal["graph", "baseline"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Provider = "gemini"
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    nvidia_api_key: str | None = None
    # Leave unset to take the default model for the selected provider.
    llm_model: str | None = None

    # "graph" = the LangGraph agent (rules in code). "baseline" = one prompt + free tool calling.
    agent: AgentKind = "graph"

    # SQLite file for leads, bookings, handoffs and transcripts; graph checkpoints sit next to it.
    db_path: Path = DATA_DIR / "leadpilot.db"

    # Freeze the clock (ISO datetime) for demos and the evaluation, so free slots are reproducible.
    clock: datetime | None = None

    # Protects the HTTP API. Unset = open (fine on localhost; set it before exposing the API).
    app_api_key: str | None = None
    # WhatsApp Cloud API webhook verification token (GET /webhook/whatsapp handshake).
    whatsapp_verify_token: str = "leadpilot-dev"

    @property
    def api_key_required(self) -> bool:
        return bool(self.app_api_key)

    def api_key(self, provider: Provider) -> str | None:
        return {
            "gemini": self.google_api_key,
            "anthropic": self.anthropic_api_key,
            "nvidia": self.nvidia_api_key,
        }[provider]

    @property
    def llm_model_resolved(self) -> str:
        """`llm_model` if set, else the default model for the provider."""
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]


settings = Settings()
