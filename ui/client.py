"""Talks to the BookWise API. The UI never imports app/ -- it could run on another machine."""

from __future__ import annotations

import os

import httpx

API_URL = os.environ.get("BOOKWISE_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("APP_API_KEY")


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str = API_URL, api_key: str | None = API_KEY) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        # One chat turn is two model calls; on the free tier that can take half a minute.
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=180)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            r = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"API not reachable at {self._http.base_url}: {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"{r.status_code}: {detail}")
        return r

    def health(self) -> dict:
        try:
            return self._request("GET", "/health").json()
        except ApiError as exc:
            return {"status": "down", "detail": str(exc)}

    def chat(self, message: str, conversation_id: str | None) -> dict:
        return self._request("POST", "/chat", json={"message": message, "conversation_id": conversation_id}).json()

    def conversation(self, conversation_id: str) -> dict:
        return self._request("GET", f"/conversations/{conversation_id}").json()

    def admin(self, what: str) -> list[dict]:
        return self._request("GET", f"/admin/{what}").json()

    def reply(self, handoff_id: int, text: str) -> None:
        self._request("POST", f"/admin/handoffs/{handoff_id}/reply", json={"text": text})

    def resolve(self, handoff_id: int) -> None:
        self._request("POST", f"/admin/handoffs/{handoff_id}/resolve")
