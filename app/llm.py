"""LLM providers -- bring your own key.

Copied from InvoiceIQ and extended with multi-turn tool calling. Three calls cover
everything the project needs:

    generate(system, prompt)                    plain text (replies, the simulated customers)
    generate_structured(system, prompt, schema) JSON that follows a pydantic schema
    chat_tools(system, messages, tools)         native function calling (the baseline agent)

Each provider returns the same small dataclasses, so callers stay provider-free.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel

from app.config import DEFAULT_MODELS, Provider, settings

_MAX_ATTEMPTS = 6  # the free tier answers 503 "high demand" in bursts
_BACKOFF_SECONDS = 8
_TIMEOUT_MS = 90_000  # per request; a chat turn normally takes 1-5 s


@dataclass(frozen=True)
class LLMAnswer:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema of the arguments


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Msg:
    """One message of a tool-calling conversation, in a provider-neutral shape.

    `raw` keeps the provider's own JSON for an assistant turn. Gemini needs it back
    verbatim (its function calls carry thought signatures that must be replayed).
    """

    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    raw: dict | None = None


@dataclass(frozen=True)
class ToolTurn:
    text: str
    tool_calls: list[ToolCall]
    model: str
    input_tokens: int
    output_tokens: int
    raw: dict | None = None

    def as_message(self) -> Msg:
        return Msg("assistant", self.text, list(self.tool_calls), raw=self.raw)


class LLMRefused(RuntimeError):
    """The provider declined to answer (safety filter, blocked prompt)."""


class ProviderNotConfigured(RuntimeError):
    """No API key for the selected provider."""


def _provider_errors() -> tuple[type[Exception], ...]:
    import anthropic
    import openai
    from google.genai import errors

    return (errors.APIError, openai.OpenAIError, anthropic.AnthropicError)


# Everything a model call can raise once retries are used up: provider API errors, network
# timeouts, a refusal, or JSON that does not match the schema (pydantic's ValidationError
# is a ValueError). Callers catch this tuple and degrade gracefully.
LLM_ERRORS: tuple[type[Exception], ...] = (
    LLMRefused, ProviderNotConfigured, httpx.HTTPError, ValueError, *_provider_errors(),
)


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, prompt: str) -> LLMAnswer: ...

    def generate_structured(self, system: str, prompt: str, schema: type[BaseModel]) -> LLMAnswer:
        """Return JSON text that follows `schema`."""
        ...

    def chat_tools(self, system: str, messages: list[Msg], tools: list[ToolSpec]) -> ToolTurn:
        """One model turn: text and/or tool calls. The caller runs the tools and calls again."""
        ...


class GeminiProvider:
    """Google Gemini. Free tier needs no credit card and no billing account.

    Get a key at https://aistudio.google.com/apikey and put it in .env as GOOGLE_API_KEY.
    """

    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import types

        self.model = model
        # Without a timeout one stuck request hangs forever (seen on the free tier).
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=_TIMEOUT_MS))

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        from google.genai import types

        response = self._call(prompt, types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=4_000,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ))
        return self._answer(response)

    def generate_structured(self, system: str, prompt: str, schema: type[BaseModel]) -> LLMAnswer:
        from google.genai import types

        # Structured output: the decoder is constrained to this JSON schema.
        response = self._call(prompt, types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=4_000,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ))
        return self._answer(response)

    def chat_tools(self, system: str, messages: list[Msg], tools: list[ToolSpec]) -> ToolTurn:
        from google.genai import types

        contents: list[types.Content] = []
        for m in messages:
            if m.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m.text)]))
            elif m.role == "assistant":
                if m.raw is not None:
                    contents.append(types.Content.model_validate(m.raw))
                    continue
                parts = [types.Part.from_text(text=m.text)] if m.text else []
                parts += [types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args))
                          for c in m.tool_calls]
                contents.append(types.Content(role="model", parts=parts))
            else:
                part = types.Part.from_function_response(name=m.tool_name, response=json.loads(m.text))
                # Results of one model turn go back together in one user turn.
                if contents and contents[-1].role == "user" and contents[-1].parts[0].function_response:
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=4_000,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description,
                                          parameters_json_schema=t.parameters)
                for t in tools
            ])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._call(contents, config)
        candidate = (response.candidates or [None])[0]
        content = getattr(candidate, "content", None)
        parts = (content.parts if content else None) or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [ToolCall(p.function_call.id or f"call_{uuid.uuid4().hex[:8]}", p.function_call.name,
                          dict(p.function_call.args or {}))
                 for p in parts if p.function_call]
        if not text and not calls:
            raise LLMRefused(_gemini_refusal_detail(response))
        usage = response.usage_metadata
        return ToolTurn(
            text=text, tool_calls=calls, model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            raw=content.model_dump(mode="json", exclude_none=True),
        )

    def _call(self, contents: object, config: object) -> object:
        from google.genai import errors

        # The free tier regularly answers 503 ("high demand") or 429 (per-minute
        # quota). Both are temporary, so wait and retry instead of failing a run.
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self._client.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.APIError as exc:
                if exc.code not in (429, 500, 503) or attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_BACKOFF_SECONDS * (attempt + 1))
            except httpx.TimeoutException:
                if attempt >= 1:  # two hung requests in a row: give up, let the caller report it
                    raise
        raise RuntimeError("unreachable")

    def _answer(self, response: object) -> LLMAnswer:
        if not response.text:
            raise LLMRefused(_gemini_refusal_detail(response))
        usage = response.usage_metadata
        return LLMAnswer(
            text=response.text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )


def _gemini_refusal_detail(response: object) -> str:
    """Turn an empty Gemini response into something a user can act on."""
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return f"Gemini blocked the prompt: {feedback.block_reason}."
    candidates = getattr(response, "candidates", None) or []
    if candidates and getattr(candidates[0], "finish_reason", None) is not None:
        return f"Gemini returned no text (finish_reason={candidates[0].finish_reason})."
    return "Gemini returned no text."


class AnthropicProvider:
    """Anthropic Claude. Paid API -- set ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=_MAX_ATTEMPTS, timeout=_TIMEOUT_MS / 1000)

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        response = self._client.messages.create(
            model=self.model, max_tokens=4_000, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise LLMRefused("Claude declined to answer.")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMAnswer(text, response.model, response.usage.input_tokens, response.usage.output_tokens)

    def generate_structured(self, system: str, prompt: str, schema: type[BaseModel]) -> LLMAnswer:
        # messages.parse sends the pydantic model as a JSON schema (structured outputs).
        response = self._client.messages.parse(
            model=self.model, max_tokens=4_000, system=system,
            messages=[{"role": "user", "content": prompt}], output_format=schema,
        )
        if response.stop_reason == "refusal":
            raise LLMRefused("Claude declined to answer.")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMAnswer(text, response.model, response.usage.input_tokens, response.usage.output_tokens)

    def chat_tools(self, system: str, messages: list[Msg], tools: list[ToolSpec]) -> ToolTurn:
        wire: list[dict] = []
        for m in messages:
            if m.role == "user":
                wire.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                blocks = [{"type": "text", "text": m.text}] if m.text else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in m.tool_calls]
                wire.append({"role": "assistant", "content": blocks})
            else:
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.text}
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
        response = self._client.messages.create(
            model=self.model, max_tokens=4_000, system=system, messages=wire,
            tools=[{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        return ToolTurn(text, calls, response.model, response.usage.input_tokens, response.usage.output_tokens)


class NvidiaProvider:
    """NVIDIA's hosted models (build.nvidia.com). Free for development -- set NVIDIA_API_KEY.

    The API speaks the OpenAI format, so the official `openai` client works with a
    different base URL.
    """

    name = "nvidia"
    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(base_url=self.BASE_URL, api_key=api_key, max_retries=_MAX_ATTEMPTS,
                              timeout=_TIMEOUT_MS / 1000 * 2)

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=8_000,  # room for the thinking plus the answer
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMRefused(f"{self.model} returned no text (finish_reason={response.choices[0].finish_reason}).")
        usage = response.usage
        return LLMAnswer(text, self.model, usage.prompt_tokens if usage else 0,
                         usage.completion_tokens if usage else 0)

    def generate_structured(self, system: str, prompt: str, schema: type[BaseModel]) -> LLMAnswer:
        hint = f"\n\nAnswer with JSON only, following this schema:\n{json.dumps(schema.model_json_schema())}"
        answer = self.generate(system, prompt + hint)
        text = answer.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return LLMAnswer(text, answer.model, answer.input_tokens, answer.output_tokens)

    def chat_tools(self, system: str, messages: list[Msg], tools: list[ToolSpec]) -> ToolTurn:
        wire: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                wire.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                msg: dict = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    msg["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                                         for c in m.tool_calls]
                wire.append(msg)
            else:
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.text})
        response = self._client.chat.completions.create(
            model=self.model, messages=wire, max_tokens=8_000,
            tools=[{"type": "function", "function": {"name": t.name, "description": t.description,
                                                     "parameters": t.parameters}} for t in tools],
        )
        message = response.choices[0].message
        calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}"))
                 for c in (message.tool_calls or [])]
        usage = response.usage
        return ToolTurn((message.content or "").strip(), calls, self.model,
                        usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)


_MISSING_KEY_HELP = {
    "gemini": (
        "GOOGLE_API_KEY is not set. Get a free key (no credit card) at "
        "https://aistudio.google.com/apikey and add it to .env"
    ),
    "anthropic": (
        "ANTHROPIC_API_KEY is not set. Get a key at https://console.anthropic.com "
        "and add it to .env, or set LLM_PROVIDER=gemini to use the free tier"
    ),
    "nvidia": (
        "NVIDIA_API_KEY is not set. Get a free key at https://build.nvidia.com "
        "and add it to .env"
    ),
}


PROVIDERS = {"gemini": GeminiProvider, "anthropic": AnthropicProvider, "nvidia": NvidiaProvider}


def get_llm(provider: Provider | None = None, model: str | None = None) -> LLMProvider:
    """Build the provider named by LLM_PROVIDER (or `provider`). Raises if its key is missing."""
    if provider is None:
        provider, model = settings.llm_provider, model or settings.llm_model_resolved
    key = settings.api_key(provider)
    if not key:
        raise ProviderNotConfigured(_MISSING_KEY_HELP[provider])
    return PROVIDERS[provider](key, model or DEFAULT_MODELS[provider])
